"""混合检索的词法能力与融合候选；不扩展原有向量检索协议。"""

from dataclasses import dataclass
from typing import Protocol

from .contracts import Collection, Passage


@dataclass(frozen=True)
class LexicalHit:
    """词法分数独立于 Passage.score，不覆盖原有相似度语义。"""

    passage: Passage
    score: float


@dataclass(frozen=True)
class RankedCandidate:
    """融合候选保留两路的首次有效排名；缺席一路用 None 表示。"""

    passage: Passage
    dense_score: float | None = None
    dense_rank: int | None = None
    bm25_score: float | None = None
    bm25_rank: int | None = None
    rrf_score: float = 0.0


class PassageSourceReader(Protocol):
    """仅从已就绪的不可变集合读取索引素材。"""

    def get_collection(self, collection_id: str) -> Collection: ...

    def list_passages(self, collection_id: str) -> list[Passage]: ...


class LexicalRetriever(Protocol):
    """按指定集合进行词法召回，不负责模型工具或生成流程。"""

    def search(self, collection_id: str, query: str, limit: int = 50) -> list[LexicalHit]: ...
