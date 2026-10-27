"""可复用本地模型；线程内批量推理，取消后等待当前批完成。"""

import asyncio
import math
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass

from flowpay.core.event_log.contracts import EventSink
from flowpay.core.event_log.store import new_id
from flowpay.core.json import to_json_object
from flowpay.core.knowledge.contracts import Passage
from flowpay.core.knowledge.hybrid_contracts import RerankHit

from .contracts import BatchPrediction, PairScorer
from .settings import RerankingSettings


def _default_factory(settings: RerankingSettings) -> PairScorer:
    from .transformers_backend import TransformersPairScorer

    return TransformersPairScorer(settings)


async def _join[T](worker: asyncio.Task[T]) -> None:
    while not worker.done():
        try:
            await asyncio.shield(worker)
        except asyncio.CancelledError:
            continue
        except Exception:
            break
    if not worker.cancelled():
        worker.exception()


@dataclass(frozen=True)
class _Prediction:
    prediction: BatchPrediction
    model_loaded: bool
    load_ms: float
    inference_ms: float


class CrossEncoderReranker:
    def __init__(
        self,
        settings: RerankingSettings,
        tracer: EventSink,
        *,
        factory: Callable[[RerankingSettings], PairScorer] = _default_factory,
    ) -> None:
        self.settings: RerankingSettings = settings
        self.tracer: EventSink = tracer
        self._factory: Callable[[RerankingSettings], PairScorer] = factory
        self._backend: PairScorer | None = None
        self._lock: threading.Lock = threading.Lock()

    async def prepare(self) -> None:
        """显式预加载，不执行模型推理。"""
        await self._execute("", [])

    async def rank(self, query: str, passages: list[Passage]) -> list[RerankHit]:
        if not query.strip() or len(query) > 1000:
            raise ValueError("Rerank query must be nonblank and at most 1000 characters")
        if len(passages) > 100:
            raise ValueError("Rerank accepts at most 100 candidates")
        if not passages:
            return []
        result = await self._execute(query, [passage.text for passage in passages])
        return [RerankHit(i, score) for i, score in enumerate(result.prediction.scores)]

    async def _execute(self, query: str, texts: list[str]) -> _Prediction:
        call_id = new_id("rerank")
        await self.tracer.emit(
            "rerank.request",
            to_json_object(
                {
                    "settings": self.settings,
                    "query": query,
                    "candidate_count": len(texts),
                    "phase": "predict" if texts else "load",
                }
            ),
            call_id,
        )
        stop = threading.Event()
        worker = asyncio.create_task(asyncio.to_thread(self._predict, query, texts, stop))
        try:
            async with asyncio.timeout(self.settings.timeout_seconds):
                result = await asyncio.shield(worker)
        except asyncio.CancelledError:
            stop.set()
            await _join(worker)
            await self.tracer.emit("rerank.canceled", {}, call_id)
            raise
        except Exception as error:
            stop.set()
            await _join(worker)
            await self.tracer.emit(
                "rerank.error", {"error_type": type(error).__name__, "message": str(error)}, call_id
            )
            raise
        await self.tracer.emit(
            "rerank.response",
            to_json_object(
                {
                    **to_json_object(result.prediction),
                    "model_loaded": result.model_loaded,
                    "load_ms": result.load_ms,
                    "inference_ms": result.inference_ms,
                }
            ),
            call_id,
        )
        return result

    def _predict(self, query: str, texts: list[str], stop: threading.Event) -> _Prediction:
        # 可取消排队等待；同一实例串行使用模型，避免并发推理抢占显存。
        while not self._lock.acquire(timeout=0.05):
            if stop.is_set():
                raise RuntimeError("Rerank cancelled before inference")
        try:
            if stop.is_set():
                raise RuntimeError("Rerank cancelled before inference")
            loaded = self._backend is None
            started = time.perf_counter()
            if self._backend is None:
                self._backend = self._factory(self.settings)
            load_ms = (time.perf_counter() - started) * 1000 if loaded else 0.0
            scores: list[float] = []
            lengths: list[int] = []
            used_lengths: list[int] = []
            query_tokens = query_used = 0
            started = time.perf_counter()
            for offset in range(0, len(texts), self.settings.batch_size):
                if stop.is_set():
                    raise RuntimeError("Rerank cancelled between batches")
                batch = texts[offset : offset + self.settings.batch_size]
                prediction = self._backend.predict(query, batch)
                if (
                    len(prediction.scores) != len(batch)
                    or len(prediction.passage_tokens) != len(batch)
                    or len(prediction.passage_tokens_used) != len(batch)
                ):
                    raise ValueError("Rerank batch response count mismatch")
                if any(
                    isinstance(score, bool) or not math.isfinite(score)
                    for score in prediction.scores
                ):
                    raise ValueError("Rerank scores must be finite numbers")
                scores.extend(prediction.scores)
                lengths.extend(prediction.passage_tokens)
                used_lengths.extend(prediction.passage_tokens_used)
                query_tokens, query_used = prediction.query_tokens, prediction.query_tokens_used
            return _Prediction(
                BatchPrediction(scores, query_tokens, query_used, lengths, used_lengths),
                loaded,
                load_ms,
                (time.perf_counter() - started) * 1000,
            )
        finally:
            self._lock.release()
