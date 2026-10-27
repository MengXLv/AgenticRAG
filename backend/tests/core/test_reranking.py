"""真实重排适配器的生命周期与协议；默认测试不加载模型权重。"""

import asyncio
import threading
from dataclasses import replace
from pathlib import Path
from typing import override

import pytest

from flowpay.core.agent.trace import Tracer
from flowpay.core.event_log import EventStore
from flowpay.core.llm.reranking.client import CrossEncoderReranker
from flowpay.core.llm.reranking.contracts import BatchPrediction
from flowpay.core.llm.reranking.settings import RerankingSettings

from .test_lexical import passage


class FakeBackend:
    def __init__(self) -> None:
        self.batches: list[list[str]] = []

    def predict(self, query: str, passages: list[str]) -> BatchPrediction:
        _ = query
        self.batches.append(passages)
        return BatchPrediction(
            [float(text) for text in passages], 20, 10, [50] * len(passages), [30] * len(passages)
        )


def settings() -> RerankingSettings:
    return RerankingSettings(
        model_id="BAAI/bge-reranker-v2-m3",
        revision="a" * 40,
        device="cpu",
        dtype="float32",
        batch_size=2,
    )


async def test_batches_preserve_mapping_model_loads_once_and_trace_records_truncation(
    tmp_path: Path,
) -> None:
    backend = FakeBackend()
    loads: list[RerankingSettings] = []

    def factory(config: RerankingSettings) -> FakeBackend:
        loads.append(config)
        return backend

    store = EventStore(tmp_path / "trace.sqlite")
    client = CrossEncoderReranker(settings(), Tracer(store, "s", "r"), factory=factory)
    rows = [passage(str(i), str(i)) for i in range(5)]
    hits = await client.rank("query", rows)
    assert [(h.candidate_index, h.score) for h in hits] == [(i, float(i)) for i in range(5)]
    assert backend.batches == [["0", "1"], ["2", "3"], ["4"]]
    await client.rank("query", rows[:1])
    assert len(loads) == 1
    responses = [e["payload"] for e in store.read("s") if e["type"] == "rerank.response"]
    assert responses[0]["passage_tokens_used"] == [30] * 5
    assert responses[0]["query_tokens_used"] == 10
    assert responses[0]["model_loaded"] is True and responses[1]["model_loaded"] is False


async def test_empty_input_does_not_load_optional_dependencies(tmp_path: Path) -> None:
    def factory(config: RerankingSettings) -> FakeBackend:
        _ = config
        raise AssertionError("must not load")

    client = CrossEncoderReranker(
        settings(), Tracer(EventStore(tmp_path / "t.sqlite"), "s", "r"), factory=factory
    )
    assert await client.rank("query", []) == []


@pytest.mark.parametrize("scores", [[], [float("nan")], [float("inf")], [True]])
async def test_backend_invalid_scores_are_errors(tmp_path: Path, scores: list[float]) -> None:
    class InvalidBackend(FakeBackend):
        @override
        def predict(self, query: str, passages: list[str]) -> BatchPrediction:
            return BatchPrediction(scores, 1, 1, [1], [1])

    store = EventStore(tmp_path / "t.sqlite")
    client = CrossEncoderReranker(
        settings(), Tracer(store, "s", "r"), factory=lambda _: InvalidBackend()
    )
    with pytest.raises(ValueError):
        await client.rank("q", [passage("A", "1")])
    assert store.read("s")[-1]["type"] == "rerank.error"


async def test_load_failure_is_not_cached(tmp_path: Path) -> None:
    attempts: list[int] = []

    def factory(config: RerankingSettings) -> FakeBackend:
        _ = config
        attempts.append(1)
        if len(attempts) == 1:
            raise RuntimeError("model missing")
        return FakeBackend()

    client = CrossEncoderReranker(
        settings(), Tracer(EventStore(tmp_path / "t.sqlite"), "s", "r"), factory=factory
    )
    with pytest.raises(RuntimeError, match="model missing"):
        await client.rank("q", [passage("A", "1")])
    assert len(await client.rank("q", [passage("A", "1")])) == 1
    assert len(attempts) == 2


async def test_repeated_cancellation_waits_for_batch_and_stops_later_batches(
    tmp_path: Path,
) -> None:
    entered, release, finished = asyncio.Event(), threading.Event(), threading.Event()
    loop = asyncio.get_running_loop()
    calls: list[int] = []

    class BlockingBackend(FakeBackend):
        @override
        def predict(self, query: str, passages: list[str]) -> BatchPrediction:
            calls.append(1)
            loop.call_soon_threadsafe(entered.set)
            try:
                assert release.wait(3)
                return super().predict(query, passages)
            finally:
                finished.set()

    store = EventStore(tmp_path / "t.sqlite")
    client = CrossEncoderReranker(
        settings(), Tracer(store, "s", "r"), factory=lambda _: BlockingBackend()
    )
    task = asyncio.create_task(client.rank("q", [passage(str(i), str(i)) for i in range(5)]))
    try:
        await asyncio.wait_for(entered.wait(), 1)
        task.cancel()
        await asyncio.sleep(0)
        task.cancel()
        await asyncio.sleep(0)
        assert not task.done()
    finally:
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await task
    assert finished.is_set() and len(calls) == 1
    assert store.read("s")[-1]["type"] == "rerank.canceled"


async def test_timeout_is_reported_after_worker_cleanup(tmp_path: Path) -> None:
    entered, release, finished = asyncio.Event(), threading.Event(), threading.Event()
    loop = asyncio.get_running_loop()

    class BlockingBackend(FakeBackend):
        @override
        def predict(self, query: str, passages: list[str]) -> BatchPrediction:
            loop.call_soon_threadsafe(entered.set)
            try:
                assert release.wait(3)
                return super().predict(query, passages)
            finally:
                finished.set()

    store = EventStore(tmp_path / "t.sqlite")
    client = CrossEncoderReranker(
        replace(settings(), timeout_seconds=0.1),
        Tracer(store, "s", "r"),
        factory=lambda _: BlockingBackend(),
    )
    task = asyncio.create_task(client.rank("q", [passage("A", "1")]))
    try:
        await asyncio.wait_for(entered.wait(), 1)
        await asyncio.sleep(0.2)
        assert not task.done()
    finally:
        release.set()
    with pytest.raises(TimeoutError):
        await task
    assert finished.is_set()
    assert store.read("s")[-1]["type"] == "rerank.error"


@pytest.mark.parametrize(
    "changes",
    [
        {"revision": "main"},
        {"batch_size": 0},
        {"max_length": 16, "max_query_tokens": 16},
        {"device": "cpu", "dtype": "float16"},
        {"timeout_seconds": float("inf")},
    ],
)
def test_settings_reject_unpinned_or_invalid_runtime(changes: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        replace(settings(), **changes)


def test_standalone_configuration_resolves_cache_and_rejects_unknown_fields(tmp_path: Path) -> None:
    path = tmp_path / "reranking.toml"
    path.write_text(
        '[reranking]\nmodel_id="model"\nrevision="' + "a" * 40 + '"\ncache_dir="cache"\n',
        encoding="utf-8",
    )
    assert RerankingSettings.load(path).cache_dir == tmp_path / "cache"
    path.write_text(path.read_text(encoding="utf-8") + "unknown=1\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Unknown"):
        RerankingSettings.load(path)


async def test_concurrent_calls_share_one_loaded_model(tmp_path: Path) -> None:
    loads: list[int] = []

    def factory(config: RerankingSettings) -> FakeBackend:
        _ = config
        loads.append(1)
        return FakeBackend()

    client = CrossEncoderReranker(
        settings(), Tracer(EventStore(tmp_path / "t.sqlite"), "s", "r"), factory=factory
    )
    results = await asyncio.gather(*(client.rank("q", [passage("A", "1")]) for _ in range(5)))
    assert all(len(result) == 1 for result in results)
    assert len(loads) == 1
