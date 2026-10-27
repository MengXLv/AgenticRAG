"""s4 多轮协作：条件跨轮累积，缺项显式追问，用户改口则旧结论作废。"""

from __future__ import annotations

import re
from collections.abc import Awaitable, Callable, Iterable
from dataclasses import dataclass, field
from typing import ClassVar, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from flowpay.core.agent import AgentTurn, Budget
from flowpay.core.data_sources.reader import SourceResult
from flowpay.core.data_sources.tools import log_tool, sql_tool
from flowpay.core.execution.contracts import TurnInput, TurnResult
from flowpay.core.execution.runtime import AgentRuntime
from flowpay.core.execution.stage import StageCard
from flowpay.core.json import JsonObject, require_list, require_object, require_text
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
from flowpay.core.tools import Tool, ToolInputError
from flowpay.workflows.s3 import LOG_TOOLS, SQL_TOOLS

SYSTEM = prompt("s4", "system")

ConditionField = Literal[
    "account_id",
    "environment",
    "payment_id",
    "request_id",
    "event_id",
    "delivery_id",
    "endpoint_id",
    "customer_id",
]

CONDITION_FIELDS: tuple[ConditionField, ...] = (
    "account_id",
    "environment",
    "payment_id",
    "request_id",
    "event_id",
    "delivery_id",
    "endpoint_id",
    "customer_id",
)

# 账户和环境圈定了记录集合，其余编号都只在这个集合里有意义：换了账户或环境，
# 旧编号指向的记录在新集合里根本不存在，留着只会被当成仍然可用的条件。
SCOPE_FIELDS: tuple[ConditionField, ...] = ("account_id", "environment")

# 业务编号自带前缀，前缀本身就指明了它是哪个字段，不必再问模型这是什么。
FIELD_BY_PREFIX: dict[str, ConditionField] = {
    "acct": "account_id",
    "pay": "payment_id",
    "req": "request_id",
    "evt": "event_id",
    "del": "delivery_id",
    "we": "endpoint_id",
    "cus": "customer_id",
}
IDENTIFIER = re.compile(rf"\b({'|'.join(FIELD_BY_PREFIX)})_[A-Za-z0-9_]+\b")

# 环境没有编号形式，只能按说法认；取值与业务工具 schema 里的 enum 一致。
ENVIRONMENT_WORDS: dict[str, tuple[str, ...]] = {
    "production": ("production", "prod", "生产", "线上", "正式"),
    "sandbox": ("sandbox", "沙箱", "沙盒", "测试环境"),
}


def conditions_in(text: str) -> dict[ConditionField, str]:
    """从一句用户消息里读出可以直接当查询条件用的字段。

    只认两种写法：带前缀的业务编号，和明确说出的环境。两种环境同时出现时不取值——
    分不清就交给模型去问，比记下一个可能错的条件好。认不出来也不是错误：
    模型仍然可以调 ask_user，这里只负责把用户已经说过的记下来，省掉重复索要。
    """
    found: dict[ConditionField, str] = {}
    for match in IDENTIFIER.finditer(text):
        found[FIELD_BY_PREFIX[match.group(1)]] = match.group(0)
    lowered = text.casefold()
    named = [
        value for value, words in ENVIRONMENT_WORDS.items() if any(w in lowered for w in words)
    ]
    if len(named) == 1:
        found["environment"] = named[0]
    return found


@dataclass(frozen=True)
class Revision:
    """本轮用户消息给条件带来的变化。

    added 是补齐，此前的结论仍然成立；changed 是改口，依赖旧值的结论要作废；
    dropped 是随账户或环境一起失效的旧编号。changed 记的是旧值，新值已经写进 Case。
    """

    added: dict[ConditionField, str] = field(default_factory=dict)
    changed: dict[ConditionField, str] = field(default_factory=dict)
    dropped: dict[ConditionField, str] = field(default_factory=dict)


@dataclass
class Case:
    """本会话已经确定的查询条件，跨轮累积。

    模型每轮都能看到这份条件，因此不必重复索要；用户改口时旧值被替换。
    本类只记录条件，不判断条件够不够发起某个查询——那是模型按工具必填项决定的。
    """

    known: dict[ConditionField, str] = field(default_factory=dict)

    @classmethod
    def restore(cls, state: object) -> Case:
        """从检查点恢复；没有检查点时按空条件开始，不认识的字段一律丢弃。"""
        if state is None:
            return cls()
        stored = require_object(require_object(state)["known"])
        return cls(
            {name: require_text(stored[name]) for name in CONDITION_FIELDS if name in stored}
        )

    def apply(self, found: dict[ConditionField, str]) -> Revision:
        """把本轮读到的条件并入；同一字段换了值即视为用户改口。"""
        added: dict[ConditionField, str] = {}
        changed: dict[ConditionField, str] = {}
        for name, value in found.items():
            previous = self.known.get(name)
            if previous == value:
                continue
            if previous is None:
                added[name] = value
            else:
                changed[name] = previous
            self.known[name] = value
        return Revision(added, changed, self.discard_out_of_scope(changed, found))

    def discard_out_of_scope(
        self, changed: dict[ConditionField, str], found: dict[ConditionField, str]
    ) -> dict[ConditionField, str]:
        """换了账户或环境时丢掉此前登记的编号，本条消息里刚给的那些除外。

        用户改口说"是另一个账户"时，上一个账户的支付、事件、投递编号在新集合里
        查不到任何东西。留着它们，模型下一轮会当成仍然有效的条件接着用。
        """
        if not any(name in changed for name in SCOPE_FIELDS):
            return {}
        stale: dict[ConditionField, str] = {}
        for name in CONDITION_FIELDS:
            if name in self.known and name not in SCOPE_FIELDS and name not in found:
                stale[name] = self.known.pop(name)
        return stale

    def missing(self, fields: Iterable[ConditionField]) -> list[ConditionField]:
        """这些字段里还没拿到的那些，顺序与传入一致。"""
        return [name for name in fields if name not in self.known]


def briefing(case: Case, revision: Revision) -> str | None:
    """本轮开头交给模型的条件说明；一个条件都还没有时不追加消息。

    这段话是"不重复索要"和"改口后重查"两条要求的唯一依据，因此把已知条件逐条列全，
    并把改动单独说一遍——模型只看消息，不看 Case 对象。
    """
    if not case.known:
        return None
    lines = [prompt("s4", "context_heading")]
    lines += [
        prompt("s4", "context_item", name=name, value=value) for name, value in case.known.items()
    ]
    if revision.changed:
        moved = "；".join(
            prompt("s4", "condition_change_item", name=name, old=old, new_value=case.known[name])
            for name, old in revision.changed.items()
        )
        lines.append(prompt("s4", "conditions_changed", moved=moved))
    if revision.dropped:
        gone = "；".join(
            prompt("s4", "discarded_condition_item", name=name, value=value)
            for name, value in revision.dropped.items()
        )
        lines.append(prompt("s4", "conditions_discarded", gone=gone))
    return "\n".join(lines)


class AskUserArguments(BaseModel):
    """模型发起追问时要说清缺哪几项，以及用什么话去问。

    missing 限定在已登记的条件字段上：追问是为了拿到能填进工具的值，
    不是自由提问，因此字段名进 JSON Schema 的 enum，模型填不出别的。
    """

    model_config: ClassVar[ConfigDict] = ConfigDict(extra="forbid")
    missing: list[ConditionField] = Field(min_length=1, max_length=len(CONDITION_FIELDS))
    question: str = Field(min_length=1, max_length=500, pattern=r"\S")


@dataclass(frozen=True)
class AskUserResult:
    """一次追问的去向：提给用户，或因为条件其实已知而被退回。"""

    missing: list[ConditionField]
    question: str
    notice: str | None = None


AskUserHandler = Callable[[AskUserArguments], Awaitable[AskUserResult]]


def ask_user_tool(handler: AskUserHandler) -> Tool:
    """把追问包成模型可见的工具；只做协议校验与序列化。"""

    async def ask(arguments: JsonObject) -> JsonObject:
        """验证参数后交给处理函数，结果原样回填，让模型知道追问是否真的发出去了。"""
        try:
            validated = AskUserArguments.model_validate(arguments)
        except ValidationError as error:
            raise ToolInputError(str(error)) from error
        result = await handler(validated)
        payload: JsonObject = {"missing": list(result.missing), "question": result.question}
        if result.notice:
            payload["notice"] = result.notice
        return payload

    return Tool(
        name="ask_user",
        description=(prompt("tools", "ask_user_description")),
        parameters=AskUserArguments.model_json_schema(),
        execute=ask,
    )


@dataclass(frozen=True)
class ToolAction:
    """本轮实际执行的一次查询；零命中也计入预算。"""

    tool: str
    target: str
    row_count: int


class S4Workflow:
    """在 s3 的调查循环上加一层跨轮状态：已知条件、已执行查询、待补充的缺项。"""

    id: str = "flowpay-multi-turn-rag"
    version: str = "1"
    max_tool_calls: int = 8
    max_model_calls: int = 10
    card: StageCard = StageCard(
        label="s4 · 多轮协作",
        description="条件跨轮累积，缺项时先问再查，用户改口后按新条件重查",
        examples=[
            "有笔支付成功了但一直没收到通知，帮我看看",
            "生产环境，账户 acct_bg_001，支付 pay_bg_001_0003",
            "说错了，是 acct_bm_030，请求编号 req_bm_030_01",
        ],
        reply_hint="条件不全时会先问缺哪几项；补齐后接着查，不重复索要已给过的字段。",
    )

    def __init__(self, model: ChatModel, prepare_query: Callable[[str], str]) -> None:
        """接收已装配的模型与查询格式函数，不创建外部连接。"""
        self.model: ChatModel = model
        self.prepare_query: Callable[[str], str] = prepare_query

    async def run(self, turn: TurnInput, runtime: AgentRuntime) -> TurnResult:
        """先合并本轮带来的条件，再跑调查循环；模型选择追问时本轮就此结束。"""
        case = Case.restore(turn.saved_state.get("case"))
        revision = case.apply(conditions_in(turn.user_message))
        done = restore_queried(turn.saved_state.get("queried"))
        agent = AgentTurn.start(
            turn, runtime, SYSTEM, self.model, Budget(self.max_model_calls, self.max_tool_calls)
        )
        if note := briefing(case, revision):
            agent.messages.add_system_message(note)
        actions: list[ToolAction] = []
        # 追问记在列表里而不是一个可空变量：赋值发生在闭包内，
        # 用列表既省掉 nonlocal，也让"推迟到回合结束"这件事看得见。
        pending: list[AskUserResult] = []

        async def ask_user(arguments: AskUserArguments) -> AskUserResult:
            """记下这次追问；用户已经给过的字段不算缺项，退回去让模型接着查。

            这里是"不重复索要已提供字段"的机制保证：靠提示词提醒会漏，靠这道检查不会。
            """
            known = {name: case.known[name] for name in arguments.missing if name in case.known}
            if known:
                still = case.missing(arguments.missing)
                given = "，".join(
                    prompt("s4", "known_condition_item", name=name, value=value)
                    for name, value in known.items()
                )
                rest = (
                    prompt("s4", "conditions_still_missing", missing="、".join(still))
                    if still
                    else prompt("s4", "conditions_complete")
                )
                return AskUserResult(
                    still,
                    arguments.question,
                    prompt("s4", "conditions_already_known", given=given, rest=rest),
                )
            if pending:
                return AskUserResult(
                    arguments.missing,
                    arguments.question,
                    prompt("s4", "question_pending"),
                )
            pending.append(AskUserResult(arguments.missing, arguments.question))
            return AskUserResult(
                arguments.missing, arguments.question, prompt("s4", "question_sent")
            )

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
            """业务关系查询；与文档检索共用预算和去重记录。"""
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
            ask_user_tool(ask_user),
            document_search_tool(search_documents),
            *([hybrid_document_search_tool(hybrid_search_documents)] if runtime.hybrid else []),
            document_read_tool(read_document),
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
            state: dict[str, object] = {"case": case, "queried": sorted(done), "actions": actions}
            if reply.is_answer:
                if await agent.disputed(reply):
                    continue
                return agent.accept(reply, **state)
            for action in reply.actions:
                agent.observe(await agent.perform(action))
            # 追问要等本轮其余工具结果全部回填之后才结束，否则检查点里会留下
            # 没有配对的 tool_calls，下一轮请求会被供应商拒绝。
            if pending:
                return agent.pause(pending[0].question, **state, waiting_for=pending[0].missing)
        raise ModelOutputError("s4 在模型调用预算内未生成有效回答")

    @staticmethod
    def claim(agent: AgentTurn, done: set[str], tool: str, target: str) -> str | None:
        """申领一次工具调用：获准则记账并返回 None，否则返回交回模型的说明。

        去重记录跨轮保留：同一条件的同一查询，上一轮问过这一轮就不必再打一次库，
        结果还在消息历史里。条件改了会生成不同的键，因此重查不会被这道检查挡住。
        """
        key = f"{tool}:{target.casefold()}"
        if key in done:
            return prompt("tools", "duplicate_session_query")
        if not agent.budget.has_tool_calls_left():
            return prompt("tools", "budget_exhausted")
        agent.budget.spend_tool_call()
        done.add(key)
        return None


def restore_queried(state: object) -> set[str]:
    """恢复此前已执行过的查询键；没有检查点时从空开始。"""
    if state is None:
        return set()
    return {require_text(item) for item in require_list(state)}
