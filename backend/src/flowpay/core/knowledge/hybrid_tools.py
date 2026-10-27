"""混合检索的独立工具协议；排名不覆盖原有 Passage.score。"""

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import ClassVar

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from flowpay.core.json import JsonObject
from flowpay.core.tools import Tool, ToolInputError

from .hybrid_contracts import HybridHit
from .ranking import DEFAULT_LIMIT


class HybridDocumentSearchArguments(BaseModel):
    model_config: ClassVar[ConfigDict] = ConfigDict(extra="forbid")
    query: str = Field(min_length=1, max_length=1000, pattern=r"\S")
    limit: int = Field(default=DEFAULT_LIMIT, strict=True, ge=1, le=10)


@dataclass(frozen=True)
class HybridDocumentSearchResult:
    query: str
    hits: list[HybridHit]
    notice: str | None = None


HybridDocumentSearchHandler = Callable[
    [HybridDocumentSearchArguments], Awaitable[HybridDocumentSearchResult]
]


def hybrid_document_search_tool(handler: HybridDocumentSearchHandler) -> Tool:
    async def search(arguments: JsonObject) -> JsonObject:
        try:
            validated = HybridDocumentSearchArguments.model_validate(arguments)
        except ValidationError as error:
            raise ToolInputError(str(error)) from error
        result = await handler(validated)
        payload: JsonObject = {
            "query": result.query,
            "passages": [
                {
                    "id": hit.passage.id,
                    "title": hit.passage.title,
                    "text": hit.passage.text,
                    "rerank_score": hit.ranking.rerank_score,
                    "final_score": hit.ranking.final_score,
                }
                for hit in result.hits
            ],
        }
        if result.notice:
            payload["notice"] = result.notice
        return payload

    return Tool(
        name="hybrid_search_documents",
        description=(
            "结合语义与关键词检索文档，融合并重排后返回原文片段和引用 ID。"
            "适合错误码、字段名、版本号或需要同时匹配多项条件的问题。"
            "默认返回 6 条；分数仅用于本次结果排序，不是答案置信度。"
        ),
        parameters=HybridDocumentSearchArguments.model_json_schema(),
        execute=search,
    )
