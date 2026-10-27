"""s3 多源调查：文档之外还有业务记录，用前一步查到的 ID 决定下一步查什么。"""

from collections.abc import Callable
from dataclasses import dataclass

from flowpay.core.agent import AgentTurn, Budget
from flowpay.core.data_sources.reader import SourceResult
from flowpay.core.data_sources.tools import LogToolSpec, SqlToolSpec, log_tool, sql_tool
from flowpay.core.execution.contracts import TurnInput, TurnResult
from flowpay.core.execution.runtime import AgentRuntime
from flowpay.core.execution.stage import StageCard
from flowpay.core.json import JsonObject
from flowpay.core.knowledge.hybrid_tools import (
    HybridDocumentSearchArguments,
    HybridDocumentSearchResult,
    hybrid_document_search_tool,
)
from flowpay.core.knowledge.tools import (
    DocumentReadArguments,
    DocumentReadResult,
    DocumentSearchArguments,
    DocumentSearchResult,
    document_read_tool,
    document_search_tool,
)
from flowpay.core.llm.chat import ChatModel
from flowpay.core.llm.chat.messages import ModelOutputError
from flowpay.core.prompts import prompt
from flowpay.core.tools import ToolInputError

SYSTEM = prompt("s3", "system")

SQL_TOOLS = (
    SqlToolSpec(
        "query_payments",
        prompt("tools", "query_payments_description"),
        "payments",
        ("account_id", "environment"),
        ("payment_id", "status", "created_by_request_id", "customer_id"),
    ),
    SqlToolSpec(
        "query_events",
        prompt("tools", "query_events_description"),
        "events",
        ("account_id", "environment"),
        ("event_id", "object_id", "event_type", "caused_by_request_id"),
    ),
    SqlToolSpec(
        "query_deliveries",
        prompt("tools", "query_deliveries_description"),
        "webhook_deliveries",
        ("account_id", "environment"),
        ("event_id", "delivery_id", "endpoint_id", "status"),
    ),
    SqlToolSpec(
        "query_endpoints",
        prompt("tools", "query_endpoints_description"),
        "webhook_endpoints",
        ("account_id", "environment"),
        ("endpoint_id", "status"),
        require_filter=False,
    ),
)

LOG_TOOLS = (
    LogToolSpec(
        "get_api_request",
        prompt("tools", "get_api_request_description"),
        "api_requests",
        ("account_id", "environment", "request_id"),
    ),
    LogToolSpec(
        "list_delivery_attempts",
        prompt("tools", "list_delivery_attempts_description"),
        "delivery_attempts",
        ("delivery_id",),
    ),
    LogToolSpec(
        "list_incidents",
        prompt("tools", "list_incidents_description"),
        "service_incidents",
        ("service", "region", "status"),
        window=True,
    ),
)


@dataclass(frozen=True)
class ToolAction:
    """本轮实际执行的一次工具调用；零命中也计入预算。"""

    tool: str
    target: str
    row_count: int


class S3Workflow:
    """在 s2 的循环上换一组工具：文档之外还有四张表和三份日志。"""

    id: str = "flowpay-multi-source-rag"
    version: str = "5"
    max_tool_calls: int = 8
    max_model_calls: int = 10
    card: StageCard = StageCard(
        label="s3 · 多源调查",
        description="结合文档规则与账户实际记录，按前一步的事实决定下一步查什么",
        examples=[
            "acct_bg_001 生产环境的 pay_bg_001_0003 成功了，但一直没收到通知，为什么？",
            "请求 req_demo_503 返回了 503，这笔支付到底创建了没有？",
            "我们的 Webhook 端点最近全部失败，是平台故障还是我们这边的问题？",
        ],
        reply_hint="业务查询需要账户与环境；每轮最多 8 次工具调用。",
    )

    def __init__(self, model: ChatModel, prepare_query: Callable[[str], str]) -> None:
        """接收已装配的模型与查询格式函数，不创建外部连接。"""
        self.model: ChatModel = model
        self.prepare_query: Callable[[str], str] = prepare_query

    async def run(self, turn: TurnInput, runtime: AgentRuntime) -> TurnResult:
        """手写 Agent Loop：一次回复里的多个动作按顺序执行，依赖关系由模型自己安排。"""
        agent = AgentTurn.start(
            turn, runtime, SYSTEM, self.model, Budget(self.max_model_calls, self.max_tool_calls)
        )
        actions: list[ToolAction] = []
        done: set[str] = set()

        async def search_documents(arguments: DocumentSearchArguments) -> DocumentSearchResult:
            """语义检索文档；与业务查询共用同一份工具预算。"""
            query = " ".join(arguments.query.split())
            if refused := self.claim(agent, done, "search", query):
                return DocumentSearchResult(query, [], refused)
            hits = await agent.search(self.prepare_query(query), limit=arguments.limit)
            actions.append(ToolAction("search_documents", query, len(hits)))
            return DocumentSearchResult(
                query, hits, prompt("tools", "no_search_results") if not hits else None
            )

        async def hybrid_search_documents(
            arguments: HybridDocumentSearchArguments,
        ) -> HybridDocumentSearchResult:
            """原始查询进入双路召回；与所有工具共用预算，单独记录查询去重键。"""
            query = " ".join(arguments.query.split())
            if refused := self.claim(agent, done, "hybrid", query):
                return HybridDocumentSearchResult(query, [], refused)
            result = await agent.search_hybrid(query, limit=arguments.limit)
            actions.append(ToolAction("hybrid_search_documents", query, len(result.hits)))
            return HybridDocumentSearchResult(
                query,
                result.hits,
                prompt("tools", "no_search_results") if not result.hits else None,
            )

        async def read_document(arguments: DocumentReadArguments) -> DocumentReadResult:
            """整篇读回一份文档。"""
            doc_id = arguments.doc_id
            if refused := self.claim(agent, done, "read", doc_id):
                return DocumentReadResult(doc_id, [], refused)
            passages = await agent.read(doc_id)
            actions.append(ToolAction("read_document", doc_id, len(passages)))
            return DocumentReadResult(
                doc_id, passages, prompt("tools", "document_missing") if not passages else None
            )

        async def run_sql(source: str, scope: JsonObject, filters: JsonObject) -> SourceResult:
            """业务查询与文档检索共用同一份预算；同样的条件组合不再打一次库。"""
            target = " ".join(f"{k}={v}" for k, v in sorted({**scope, **filters}.items()))
            if refused := self.claim(agent, done, source, target):
                raise ToolInputError(refused)
            result = await runtime.query_sql(source, scope, filters)
            actions.append(ToolAction(source, target, len(result["rows"])))
            return result

        async def run_logs(
            source: str,
            *,
            filters: JsonObject | None = None,
            start: str | None = None,
            end: str | None = None,
        ) -> SourceResult:
            """日志查询走同一套预算与去重；时间窗口也计入条件。"""
            conditions = {**(filters or {}), "start": start or "", "end": end or ""}
            target = " ".join(f"{k}={v}" for k, v in sorted(conditions.items()) if v)
            if refused := self.claim(agent, done, source, target):
                raise ToolInputError(refused)
            result = await runtime.query_logs(source, filters=filters, start=start, end=end)
            actions.append(ToolAction(source, target, len(result["rows"])))
            return result

        agent.offer(
            document_search_tool(search_documents),
            *([hybrid_document_search_tool(hybrid_search_documents)] if runtime.hybrid else []),
            document_read_tool(read_document),
            # 取值封闭的字段（环境、区域、服务、各种状态）由策略声明，写进工具的
            # JSON Schema。模型因此填不出登记之外的值，不会拿到一个"问法不对"
            # 却看起来像"没有记录"的空结果。
            *(
                sql_tool(spec, run_sql, runtime.field_values("sql", spec.source))
                for spec in SQL_TOOLS
            ),
            *(
                log_tool(spec, run_logs, runtime.field_values("logs", spec.source))
                for spec in LOG_TOOLS
            ),
        )
        while agent.may_think():
            reply = await agent.think()
            if reply.is_answer:
                if await agent.disputed(reply):
                    continue
                return agent.accept(reply, actions=actions)
            for action in reply.actions:
                agent.observe(await agent.perform(action))
        raise ModelOutputError("s3 在模型调用预算内未生成有效回答")

    @staticmethod
    def claim(agent: AgentTurn, done: set[str], tool: str, target: str) -> str | None:
        """申领一次工具调用：获准则记账并返回 None，否则返回交回模型的说明。

        文档检索、SQL 与日志共用一份预算和一份去重记录，因此"换个工具再问同一件事"
        同样会被拦下，调查成本是可预期的。
        """
        key = f"{tool}:{target.casefold()}"
        if key in done:
            return "这个查询已经执行过，请使用已有结果或换一个条件。"
        if not agent.budget.has_tool_calls_left():
            return prompt("tools", "budget_exhausted")
        agent.budget.spend_tool_call()
        done.add(key)
        return None
