"""服务生命周期内按集合复用的进程内 BM25 召回。"""

from dataclasses import dataclass
from threading import Lock

from rank_bm25 import BM25Okapi

from .contracts import Passage
from .hybrid_contracts import LexicalHit, PassageSourceReader
from .tokenization import tokenize


@dataclass(frozen=True)
class _Index:
    passages: list[Passage]
    postings: dict[str, set[int]]
    model: BM25Okapi | None


class BM25Retriever:
    """首次访问集合时加载索引，后续查询不扫描数据库或重建分词结果。

    ready 集合不可改写，因此缓存可与实例同寿命。调用方应在服务生命周期内复用
    实例，并在线程中运行同步查询；本类不自行创建后台任务或数据库连接。
    """

    def __init__(self, source: PassageSourceReader) -> None:
        self.source: PassageSourceReader = source
        self._indexes: dict[str, _Index] = {}
        self._lock: Lock = Lock()

    def prepare(self, collection_id: str) -> None:
        """可在启动时显式预热；失败不缓存，下一次可重试。"""
        self._index(collection_id)

    def search(self, collection_id: str, query: str, limit: int = 50) -> list[LexicalHit]:
        """只返回至少匹配一个 token 的片段，允许零或负 BM25 分数。"""
        if type(limit) is not int or not 1 <= limit <= 100:
            raise ValueError("BM25 search limit must be 1..100")
        if not query.strip():
            raise ValueError("BM25 query must not be blank")
        index = self._index(collection_id)
        tokens = tokenize(query)
        matched: set[int] = set()
        for token in tokens:
            matched.update(index.postings.get(token, ()))
        if not matched or index.model is None:
            return []
        scores = index.model.get_scores(tokens)
        ordered = sorted(
            matched,
            key=lambda i: (
                -float(scores[i]),
                index.passages[i].source.doc_id,
                index.passages[i].source.chunk_no,
            ),
        )
        return [LexicalHit(index.passages[i], float(scores[i])) for i in ordered[:limit]]

    def _index(self, collection_id: str) -> _Index:
        # 初始化由同一把锁保护，避免并发首次查询重复读全量片段或构建半成品索引。
        with self._lock:
            if collection_id in self._indexes:
                return self._indexes[collection_id]
            collection = self.source.get_collection(collection_id)
            if collection["status"] != "ready":
                raise ValueError("Collection is not ready")
            passages = self.source.list_passages(collection_id)
            if any(p.source.collection_id != collection_id for p in passages):
                raise ValueError("Index passages belong to another collection")
            identities = [(p.source.doc_id, p.source.chunk_no) for p in passages]
            if len(set(identities)) != len(identities):
                raise ValueError("Index contains duplicate passages")
            corpus = [tokenize(p.text) for p in passages]
            postings: dict[str, set[int]] = {}
            for position, tokens in enumerate(corpus):
                for token in set(tokens):
                    postings.setdefault(token, set()).add(position)
            # BM25Okapi 不能构建全空语料；有空片段的非空语料仍保留原始下标。
            model = BM25Okapi(corpus) if postings else None
            index = _Index(passages, postings, model)
            self._indexes[collection_id] = index
            return index
