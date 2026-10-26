"""rank-bm25 0.2.x 本项目使用的公共接口。"""

from typing import Protocol, SupportsFloat

class _Scores(Protocol):
    def __getitem__(self, index: int) -> SupportsFloat: ...

class BM25Okapi:
    def __init__(self, corpus: list[list[str]]) -> None: ...
    def get_scores(self, query: list[str]) -> _Scores: ...
