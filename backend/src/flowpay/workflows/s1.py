"""s1 朴素 RAG：带对话背景检索一次，再直接生成 Markdown 回答。"""

from collections.abc import Callable

from flowpay.core.agent import AgentTurn, Budget
from flowpay.core.chat.context import conversation_history
from flowpay.core.execution.contracts import TurnInput, TurnResult
from flowpay.core.execution.runtime import AgentRuntime
from flowpay.core.execution.stage import StageCard
from flowpay.core.llm.chat import ChatModel
from flowpay.core.prompts import prompt

SYSTEM = prompt("s1", "system")


class S1Workflow:
    """固定一次检索和一次生成，便于观察朴素 RAG 的效果与局限。"""

    id: str = "flowpay-document-rag"
    version: str = "4"
    top_k: int = 10
    card: StageCard = StageCard(
        label="s1 · 朴素 RAG",
        description="结合最近对话检索 10 个片段，生成带来源的 Markdown 回答",
        examples=[
            "Webhook 端点返回 HTTP 410 后，系统会怎样处理？",
            "重复使用幂等键，但请求参数不同，会发生什么？",
            "订单成功但没收到通知，文档能确认什么？",
        ],
        reply_hint="每轮检索一次；查询带最近对话，可展开本轮检索到的全部资料。",
    )

    def __init__(self, model: ChatModel, prepare_query: Callable[[str], str]) -> None:
        """接收模型与嵌入查询格式函数，阶段不创建外部连接。"""
        self.model: ChatModel = model
        self.prepare_query: Callable[[str], str] = prepare_query

    async def run(self, turn: TurnInput, runtime: AgentRuntime) -> TurnResult:
        """检索一次、生成一次；不重试、不补搜，作为后续阶段的对照基线。"""
        agent = AgentTurn.start(
            turn, runtime, SYSTEM, self.model, Budget(1, tool_calls=0), remember=False
        )
        passages = await agent.search(self.prepare_query(self.query(turn)), limit=self.top_k)
        if not passages:
            return agent.give_up("没有检索到可用的文档，当前资料无法支持回答。")
        agent.messages.add_system_message(
            prompt("s1", "sources_context") + agent.retrieved.serialize()
        )
        return agent.accept(await agent.think())

    def query(self, turn: TurnInput) -> str:
        """拼接最近两轮只补充指代背景，不增加模型调用；原问题仍保留在末尾。"""
        history = conversation_history(turn)[-4:]
        if not history:
            return turn.user_message
        background = "\n".join(f"{m['role']}: {m['content']}" for m in history)
        return prompt("s1", "conversation_query", background=background, question=turn.user_message)
