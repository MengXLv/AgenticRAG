"""两路召回、RRF 与独立重排；不改变原有知识检索客户端。"""

import asyncio
import math
import time
from collections.abc import Awaitable, Callable
from typing import override

from flowpay.core.event_log.contracts import EventSink
from flowpay.core.event_log.store import new_id
from flowpay.core.json import to_json_object

from .client import KnowledgeClient
from .fusion import fuse_rrf
from .hybrid_contracts import (
    DenseRetriever,
    HybridHit,
    HybridSearchResult,
    LexicalHit,
    LexicalRetriever,
    RankedCandidate,
    Reranker,
)
from .ranking import DEFAULT_LIMIT, DEFAULT_RERANK_WEIGHT, rank_candidates, validate_weight


async def _drain[T](task: asyncio.Future[T]) -> None:
    """等待已取消任务或同步线程回收；重复取消也不能跳过清理。"""
    while not task.done():
        try:
            await asyncio.shield(task)
        except asyncio.CancelledError:
            continue
        except Exception:
            break
    if not task.cancelled():
        task.exception()


class _JoinedKnowledgeClient(KnowledgeClient):
    """仅混合链路使用；继承原有校验和日志，在同步读库边界等待线程退出。"""

    @override
    async def _call_store[**P, T](
        self, operation: Callable[P, T], *args: P.args, **kwargs: P.kwargs
    ) -> T:
        worker = asyncio.create_task(asyncio.to_thread(operation, *args, **kwargs))
        try:
            return await asyncio.shield(worker)
        except asyncio.CancelledError:
            await _drain(worker)
            raise


class HybridKnowledgeClient:
    def __init__(
        self,
        dense: DenseRetriever,
        lexical: LexicalRetriever,
        reranker: Reranker,
        tracer: EventSink,
        prepare_query: Callable[[str], str],
        *,
        recall_limit: int = 50,
        rerank_limit: int = 30,
        rank_constant: int = 60,
        timeout_seconds: float = 30,
        rerank_weight: float = DEFAULT_RERANK_WEIGHT,
    ) -> None:
        if type(recall_limit) is not int or not 1 <= recall_limit <= 100:
            raise ValueError("recall_limit must be 1..100")
        if type(rerank_limit) is not int or not 10 <= rerank_limit <= 2 * recall_limit:
            raise ValueError("rerank_limit must be 10..2*recall_limit")
        if type(rank_constant) is not int or rank_constant < 1:
            raise ValueError("rank_constant must be a positive integer")
        if (
            isinstance(timeout_seconds, bool)
            or not math.isfinite(timeout_seconds)
            or timeout_seconds <= 0
        ):
            raise ValueError("timeout_seconds must be finite and positive")
        validate_weight(rerank_weight)
        self.rerank_weight: float = rerank_weight
        self.dense: DenseRetriever = (
            _JoinedKnowledgeClient(dense.store, dense.embeddings, dense.tracer)
            if isinstance(dense, KnowledgeClient)
            else dense
        )
        self.lexical: LexicalRetriever = lexical
        self.reranker: Reranker = reranker
        self.tracer: EventSink = tracer
        self.prepare_query: Callable[[str], str] = prepare_query
        self.recall_limit: int = recall_limit
        self.rerank_limit: int = rerank_limit
        self.rank_constant: int = rank_constant
        self.timeout_seconds: float = timeout_seconds

    async def search(
        self, collection_id: str, query: str, limit: int = DEFAULT_LIMIT
    ) -> HybridSearchResult:
        call_id = new_id("hybrid")
        await self.tracer.emit(
            "hybrid.request",
            {
                "collection_id": collection_id,
                "query": query,
                "limit": limit,
                "recall_limit": self.recall_limit,
                "rerank_limit": self.rerank_limit,
                "rank_constant": self.rank_constant,
                "timeout_seconds": self.timeout_seconds,
                "final_ranking": "normalized_score_blend",
                "rerank_weight": self.rerank_weight,
            },
            call_id,
        )
        try:
            if not collection_id.strip():
                raise ValueError("collection_id must not be blank")
            if not query.strip() or len(query) > 1000:
                raise ValueError("query must be nonblank and at most 1000 characters")
            if type(limit) is not int or not 1 <= limit <= 10:
                raise ValueError("limit must be 1..10")
            async with asyncio.timeout(self.timeout_seconds):
                result = await self._search(collection_id, query, limit, call_id)
            await self.tracer.emit("hybrid.response", to_json_object(result), call_id)
            return result
        except asyncio.CancelledError:
            await self.tracer.emit("hybrid.canceled", {}, call_id)
            raise
        except Exception as error:
            await self.tracer.emit(
                "hybrid.error", {"error_type": type(error).__name__, "message": str(error)}, call_id
            )
            raise

    async def _lexical(self, collection_id: str, query: str) -> list[LexicalHit]:
        worker = asyncio.create_task(
            asyncio.to_thread(self.lexical.search, collection_id, query, self.recall_limit)
        )
        try:
            return await asyncio.shield(worker)
        except asyncio.CancelledError:
            # Python 无法中断正在执行的线程，终态必须等读库/索引任务结束。
            await _drain(worker)
            raise

    async def _stage[T](self, name: str, operation: Awaitable[T], call_id: str) -> T:
        started = time.perf_counter()
        worker = asyncio.ensure_future(operation)
        try:
            try:
                result = await asyncio.shield(worker)
            except asyncio.CancelledError:
                worker.cancel()
                await _drain(worker)
                raise
        except BaseException as error:
            await self.tracer.emit(
                "hybrid.stage",
                {
                    "stage": name,
                    "duration_ms": (time.perf_counter() - started) * 1000,
                    "status": "canceled" if isinstance(error, asyncio.CancelledError) else "error",
                    "error_type": type(error).__name__,
                },
                call_id,
            )
            raise
        await self.tracer.emit(
            "hybrid.stage",
            to_json_object(
                {
                    "stage": name,
                    "duration_ms": (time.perf_counter() - started) * 1000,
                    "status": "ok",
                    "results": result,
                }
            ),
            call_id,
        )
        return result

    async def _search(
        self, collection_id: str, query: str, limit: int, call_id: str
    ) -> HybridSearchResult:
        dense_task = asyncio.create_task(
            self._stage(
                "dense",
                self.dense.search(collection_id, self.prepare_query(query), self.recall_limit),
                call_id,
            )
        )
        lexical_task = asyncio.create_task(
            self._stage("bm25", self._lexical(collection_id, query), call_id)
        )
        combined = asyncio.gather(dense_task, lexical_task)
        try:
            dense, lexical = await asyncio.shield(combined)
        except BaseException:
            dense_task.cancel()
            lexical_task.cancel()
            cleanup = asyncio.gather(dense_task, lexical_task, return_exceptions=True)
            await _drain(cleanup)
            await _drain(combined)
            raise
        if any(p.source.collection_id != collection_id for p in dense) or any(
            h.passage.source.collection_id != collection_id for h in lexical
        ):
            raise ValueError("Retrieved evidence belongs to another collection")
        started = time.perf_counter()
        fused = fuse_rrf(dense, lexical, self.rank_constant)
        candidates = fused[: self.rerank_limit]
        await self.tracer.emit(
            "hybrid.stage",
            to_json_object(
                {
                    "stage": "fusion",
                    "status": "ok",
                    "duration_ms": (time.perf_counter() - started) * 1000,
                    "results": fused,
                    "selected_count": len(candidates),
                }
            ),
            call_id,
        )
        if not candidates:
            return HybridSearchResult([])
        hits = await self._stage("rerank", self._rerank(query, candidates), call_id)
        return HybridSearchResult(hits[:limit])

    async def _rerank(self, query: str, candidates: list[RankedCandidate]) -> list[HybridHit]:
        ranks = await self.reranker.rank(query, [row.passage for row in candidates])
        return rank_candidates(candidates, ranks, rerank_weight=self.rerank_weight)
