"""同步推理适配边界，测试可替换而不安装模型依赖。"""

from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class BatchPrediction:
    scores: list[float]
    query_tokens: int
    query_tokens_used: int
    passage_tokens: list[int]
    passage_tokens_used: list[int]


class PairScorer(Protocol):
    def predict(self, query: str, passages: list[str]) -> BatchPrediction: ...
