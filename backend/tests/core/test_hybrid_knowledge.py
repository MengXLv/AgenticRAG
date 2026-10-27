"""混合召回编排；假重排器只证明流程，不代表真实模型质量。"""

import asyncio
import threading
from pathlib import Path
from typing import override

import pytest

from flowpay.core.agent.trace import Tracer
from flowpay.core.event_log import EventStore
from flowpay.core.knowledge import PostgresVectorStore
from flowpay.core.knowledge.client import KnowledgeClient
from flowpay.core.knowledge.contracts import Passage
from flowpay.core.knowledge.hybrid import HybridKnowledgeClient
from flowpay.core.knowledge.hybrid_contracts import LexicalHit, RerankHit
from flowpay.core.llm.embedding import EmbeddedBatch, EmbeddingClient, text_hash
from flowpay.core.llm.embedding.ollama import retrieval_input
from flowpay.core.llm.embedding.openai_embedding import OpenAiEmbeddingModel

from .test_embeddings import settings
from .test_lexical import MemorySource, passage


class Dense:
    def __init__(self, rows: list[Passage]) -> None:
        self.rows: list[Passage] = rows
        self.calls: list[tuple[str, str, int]] = []

    async def search(self, collection_id: str, text: str, limit: int = 5) -> list[Passage]:
        self.calls.append((collection_id, text, limit))
        return self.rows[:limit]


class Lexical:
    def __init__(self, rows: list[LexicalHit]) -> None:
        self.rows: list[LexicalHit] = rows
        self.calls: list[tuple[str, str, int]] = []

    def search(self, collection_id: str, query: str, limit: int = 50) -> list[LexicalHit]:
        self.calls.append((collection_id, query, limit))
        return self.rows[:limit]


class Rerank:
    def __init__(self, output: list[RerankHit] | None = None) -> None:
        self.output: list[RerankHit] | None = output
        self.calls: list[tuple[str, list[Passage]]] = []

    async def rank(self, query: str, passages: list[Passage]) -> list[RerankHit]:
        self.calls.append((query, passages))
        return (
            self.output
            if self.output is not None
            else [RerankHit(i, float(i)) for i in range(len(passages))]
        )


def make_client(
    tmp_path: Path,
    dense: Dense,
    lexical: Lexical,
    reranker: Rerank,
    *,
    rerank_limit: int = 30,
    timeout_seconds: float = 2,
) -> tuple[HybridKnowledgeClient, EventStore]:
    store = EventStore(tmp_path / "hybrid.sqlite")
    return HybridKnowledgeClient(
        dense,
        lexical,
        reranker,
        Tracer(store, "session", "run"),
        retrieval_input,
        rerank_limit=rerank_limit,
        timeout_seconds=timeout_seconds,
    ), store


async def test_branches_overlap_and_only_dense_query_has_instruction(tmp_path: Path) -> None:
    dense_entered = threading.Event()
    lexical_entered = threading.Event()

    class GatedDense(Dense):
        @override
        async def search(self, collection_id: str, text: str, limit: int = 5) -> list[Passage]:
            dense_entered.set()
            assert await asyncio.to_thread(lexical_entered.wait, 1)
            return await super().search(collection_id, text, limit)

    class GatedLexical(Lexical):
        @override
        def search(self, collection_id: str, query: str, limit: int = 50) -> list[LexicalHit]:
            lexical_entered.set()
            assert dense_entered.wait(1)
            return super().search(collection_id, query, limit)

    a = passage("A", "HTTP 410")
    dense, lexical, rerank = GatedDense([a]), GatedLexical([LexicalHit(a, 3)]), Rerank()
    client, store = make_client(tmp_path, dense, lexical, rerank)
    result = await client.search("ready", "HTTP 410")
    assert dense.calls == [("ready", retrieval_input("HTTP 410"), 50)]
    assert lexical.calls == [("ready", "HTTP 410", 50)]
    assert rerank.calls == [("HTTP 410", [a])]
    assert result.hits[0].passage is a
    ranking = result.hits[0].ranking
    assert ranking.dense_rank == ranking.bm25_rank == 1
    assert ranking.rrf_score == pytest.approx(2 / 61)
    assert ranking.rerank_score == 0
    events = store.read("session")
    assert events[0]["type"] == "hybrid.request" and events[-1]["type"] == "hybrid.response"
    stages = [e["payload"]["stage"] for e in events if e["type"] == "hybrid.stage"]
    assert set(stages) == {"dense", "bm25", "fusion", "rerank"}
    assert len({e["call_id"] for e in events}) == 1


async def test_fusion_candidate_limit_and_rerank_top_k(tmp_path: Path) -> None:
    rows = [passage(f"{i:03}", str(i)) for i in range(50)]
    rerank = Rerank()
    client, _ = make_client(tmp_path, Dense(rows), Lexical([]), rerank)
    result = await client.search("ready", "query", limit=5)
    assert len(rerank.calls[0][1]) == 30
    assert [hit.passage.source.doc_id for hit in result.hits] == ["029", "028", "027", "026", "025"]
    assert all(row.score == 0 for row in rows)


async def test_empty_candidates_skip_reranker(tmp_path: Path) -> None:
    rerank = Rerank()
    client, _ = make_client(tmp_path, Dense([]), Lexical([]), rerank)
    assert (await client.search("ready", "query")).hits == []
    assert not rerank.calls


async def test_default_client_uses_cross_encoder_scores_and_keeps_requested_limit(
    tmp_path: Path,
) -> None:
    source = [passage(name, name) for name in "ABC"]
    store = EventStore(tmp_path / "default.sqlite")
    client = HybridKnowledgeClient(
        Dense(source),
        Lexical([]),
        Rerank([RerankHit(0, 0), RerankHit(1, 1), RerankHit(2, 0.9)]),
        Tracer(store, "s", "r"),
        retrieval_input,
    )
    result = await client.search("ready", "query", limit=2)
    assert [h.passage.source.doc_id for h in result.hits] == ["B", "C"]
    assert store.read("s")[0]["payload"]["final_ranking"] == "cross_encoder"
    assert "rerank_weight" not in store.read("s")[0]["payload"]
    assert result.hits[0].ranking.final_score == result.hits[0].ranking.rerank_score


async def test_new_pipeline_default_is_six_but_explicit_five_is_respected(tmp_path: Path) -> None:
    source = [passage(str(i), str(i)) for i in range(8)]
    client, _ = make_client(tmp_path, Dense(source), Lexical([]), Rerank())
    assert len((await client.search("ready", "query")).hits) == 6
    assert len((await client.search("ready", "query", limit=5)).hits) == 5


async def test_tied_rerank_scores_use_stable_document_order(tmp_path: Path) -> None:
    a, b = passage("A", "a"), passage("B", "b")
    client, _ = make_client(
        tmp_path, Dense([b, a]), Lexical([]), Rerank([RerankHit(0, 1), RerankHit(1, 1)])
    )
    assert [h.passage.id for h in (await client.search("ready", "q")).hits] == [a.id, b.id]


@pytest.mark.parametrize(
    "output",
    [
        [],
        [RerankHit(1, 1)],
        [RerankHit(-1, 1)],
        [RerankHit(True, 1)],
        [RerankHit(0, float("nan"))],
        [RerankHit(0, float("inf"))],
        [RerankHit(0, 1), RerankHit(0, 2)],
    ],
)
async def test_invalid_reranker_mapping_is_rejected(
    tmp_path: Path, output: list[RerankHit]
) -> None:
    client, store = make_client(tmp_path, Dense([passage("A", "a")]), Lexical([]), Rerank(output))
    with pytest.raises(ValueError, match="Rerank"):
        await client.search("ready", "q")
    assert store.read("session")[-1]["type"] == "hybrid.error"


async def test_foreign_collection_is_rejected_even_if_both_lanes_agree(tmp_path: Path) -> None:
    a = passage("A", "a", "another")
    rerank = Rerank()
    client, _ = make_client(tmp_path, Dense([a]), Lexical([LexicalHit(a, 1)]), rerank)
    with pytest.raises(ValueError, match="collection"):
        await client.search("ready", "q")
    assert not rerank.calls


@pytest.mark.parametrize("limit", [True, 0, 11, 1.5])
async def test_invalid_limit_never_calls_dependencies(tmp_path: Path, limit: int) -> None:
    dense, lexical, rerank = Dense([]), Lexical([]), Rerank()
    client, _ = make_client(tmp_path, dense, lexical, rerank)
    with pytest.raises(ValueError, match="limit"):
        await client.search("ready", "query", limit)
    assert not dense.calls and not lexical.calls and not rerank.calls


@pytest.mark.parametrize("query", [" ", "x" * 1001])
async def test_invalid_query_never_calls_dependencies(tmp_path: Path, query: str) -> None:
    dense, lexical, rerank = Dense([]), Lexical([]), Rerank()
    client, _ = make_client(tmp_path, dense, lexical, rerank)
    with pytest.raises(ValueError, match="query"):
        await client.search("ready", query)
    assert not dense.calls and not lexical.calls and not rerank.calls


async def test_lexical_failure_cancels_and_awaits_dense_branch(tmp_path: Path) -> None:
    ended = asyncio.Event()

    class WaitingDense(Dense):
        @override
        async def search(self, collection_id: str, text: str, limit: int = 5) -> list[Passage]:
            try:
                await asyncio.Event().wait()
                return []
            finally:
                ended.set()

    class FailingLexical(Lexical):
        @override
        def search(self, collection_id: str, query: str, limit: int = 50) -> list[LexicalHit]:
            raise RuntimeError("lexical failed")

    client, store = make_client(tmp_path, WaitingDense([]), FailingLexical([]), Rerank())
    with pytest.raises(RuntimeError, match="lexical failed"):
        await client.search("ready", "q")
    assert ended.is_set()
    assert store.read("session")[-1]["type"] == "hybrid.error"


async def test_cancellation_waits_for_sync_worker_even_when_cancelled_twice(tmp_path: Path) -> None:
    entered, finished = asyncio.Event(), threading.Event()
    release = threading.Event()
    loop = asyncio.get_running_loop()

    class BlockingLexical(Lexical):
        @override
        def search(self, collection_id: str, query: str, limit: int = 50) -> list[LexicalHit]:
            loop.call_soon_threadsafe(entered.set)
            try:
                assert release.wait(3)
                return []
            finally:
                finished.set()

    client, store = make_client(tmp_path, Dense([]), BlockingLexical([]), Rerank())
    task = asyncio.create_task(client.search("ready", "q"))
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
    assert finished.is_set()
    assert store.read("session")[-1]["type"] == "hybrid.canceled"


async def test_reranker_timeout_is_an_error_and_cleans_up(tmp_path: Path) -> None:
    entered, finished = asyncio.Event(), asyncio.Event()

    class WaitingRerank(Rerank):
        @override
        async def rank(self, query: str, passages: list[Passage]) -> list[RerankHit]:
            entered.set()
            try:
                await asyncio.Event().wait()
                return []
            finally:
                finished.set()

    client, store = make_client(
        tmp_path, Dense([passage("A", "a")]), Lexical([]), WaitingRerank(), timeout_seconds=0.1
    )
    with pytest.raises(TimeoutError):
        await client.search("ready", "q")
    assert entered.is_set() and finished.is_set()
    assert store.read("session")[-1]["type"] == "hybrid.error"


async def test_dense_failure_waits_for_sync_worker_before_error_terminal(tmp_path: Path) -> None:
    entered = asyncio.Event()
    release = threading.Event()
    finished = threading.Event()
    loop = asyncio.get_running_loop()

    class FailingDense(Dense):
        @override
        async def search(self, collection_id: str, text: str, limit: int = 5) -> list[Passage]:
            await entered.wait()
            raise RuntimeError("dense failed")

    class BlockingLexical(Lexical):
        @override
        def search(self, collection_id: str, query: str, limit: int = 50) -> list[LexicalHit]:
            loop.call_soon_threadsafe(entered.set)
            try:
                assert release.wait(3)
                return []
            finally:
                finished.set()

    client, store = make_client(tmp_path, FailingDense([]), BlockingLexical([]), Rerank())
    task = asyncio.create_task(client.search("ready", "q"))
    try:
        await asyncio.wait_for(entered.wait(), 1)
        await asyncio.sleep(0)
        assert not task.done()
    finally:
        release.set()
    with pytest.raises(RuntimeError, match="dense failed"):
        await task
    assert finished.is_set()
    events = store.read("session")
    assert events[-1]["type"] == "hybrid.error"
    assert not any(event["type"] == "hybrid.response" for event in events)


async def test_reranker_failure_is_not_silently_replaced_by_rrf(tmp_path: Path) -> None:
    class FailingRerank(Rerank):
        @override
        async def rank(self, query: str, passages: list[Passage]) -> list[RerankHit]:
            raise RuntimeError("reranker unavailable")

    client, store = make_client(tmp_path, Dense([passage("A", "a")]), Lexical([]), FailingRerank())
    with pytest.raises(RuntimeError, match="reranker unavailable"):
        await client.search("ready", "q")
    assert store.read("session")[-1]["type"] == "hybrid.error"


async def test_incomplete_and_duplicate_rerank_indices_with_right_count(tmp_path: Path) -> None:
    client, _ = make_client(
        tmp_path,
        Dense([passage("A", "a"), passage("B", "b")]),
        Lexical([]),
        Rerank([RerankHit(0, 1), RerankHit(0, 2)]),
    )
    with pytest.raises(ValueError, match="duplicated"):
        await client.search("ready", "q")


def test_invalid_configuration_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="rerank_limit"):
        make_client(tmp_path, Dense([]), Lexical([]), Rerank(), rerank_limit=9)
    for timeout in (0, -1, float("nan"), float("inf"), True):
        with pytest.raises(ValueError, match="timeout_seconds"):
            make_client(tmp_path, Dense([]), Lexical([]), Rerank(), timeout_seconds=timeout)


async def test_repeated_parent_cancellation_does_not_interrupt_dense_cleanup(
    tmp_path: Path,
) -> None:
    entered, cleaning, release, finished = (asyncio.Event() for _ in range(4))

    class CleaningDense(Dense):
        @override
        async def search(self, collection_id: str, text: str, limit: int = 5) -> list[Passage]:
            entered.set()
            try:
                await asyncio.Event().wait()
                return []
            finally:
                cleaning.set()
                await release.wait()
                finished.set()

    client, store = make_client(tmp_path, CleaningDense([]), Lexical([]), Rerank())
    task = asyncio.create_task(client.search("ready", "q"))
    try:
        await asyncio.wait_for(entered.wait(), 1)
        task.cancel()
        await asyncio.wait_for(cleaning.wait(), 1)
        task.cancel()
        await asyncio.sleep(0)
    finally:
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await task
    assert finished.is_set()
    assert store.read("session")[-1]["type"] == "hybrid.canceled"


@pytest.mark.parametrize("boundary", ["get_collection", "search"])
async def test_real_knowledge_client_waits_for_its_store_thread(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, boundary: str
) -> None:
    entered, release, finished = asyncio.Event(), threading.Event(), threading.Event()
    loop = asyncio.get_running_loop()
    tracer = Tracer(EventStore(tmp_path / "dense.sqlite"), "dense", "run")
    model_settings = settings()
    embeddings = EmbeddingClient(OpenAiEmbeddingModel(model_settings), tracer)
    collection = MemorySource([]).get_collection("ready")
    collection["build_config"] = {"embedding": model_settings.identity()}
    # 构造真实存储类型但替换读边界，使用屏障模拟在途 SQL，无需外部数据库。
    vector_store = PostgresVectorStore.__new__(PostgresVectorStore)

    def get_collection(collection_id: str):
        _ = collection_id
        return collection

    monkeypatch.setattr(vector_store, "get_collection", get_collection)

    async def embed(texts: list[str]) -> EmbeddedBatch:
        return EmbeddedBatch(
            model_settings.identity(), [text_hash(t) for t in texts], [[1.0, 0.0, 0.0]], []
        )

    monkeypatch.setattr(embeddings, "embed", embed)

    def blocked(*args: object, **kwargs: object) -> object:
        _ = args, kwargs
        loop.call_soon_threadsafe(entered.set)
        try:
            assert release.wait(3)
            return collection if boundary == "get_collection" else []
        finally:
            finished.set()

    monkeypatch.setattr(vector_store, boundary, blocked)
    dense = KnowledgeClient(vector_store, embeddings, tracer)
    store = EventStore(tmp_path / "hybrid.sqlite")
    client = HybridKnowledgeClient(
        dense, Lexical([]), Rerank(), Tracer(store, "session", "run"), retrieval_input
    )
    task = asyncio.create_task(client.search("ready", "q"))
    try:
        await asyncio.wait_for(entered.wait(), 1)
        task.cancel()
        # 直接等终态会在旧实现立刻返回；带回收适配器时仍被线程屏障阻塞。
        done, _ = await asyncio.wait({task}, timeout=0.1)
        assert not done
        assert not any(e["type"] == "hybrid.canceled" for e in store.read("session"))
    finally:
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await task
    assert finished.is_set()
    assert store.read("session")[-1]["type"] == "hybrid.canceled"


async def test_repeated_parent_cancellation_does_not_interrupt_rerank_cleanup(
    tmp_path: Path,
) -> None:
    entered, cleaning, release, finished = (asyncio.Event() for _ in range(4))

    class CleaningRerank(Rerank):
        @override
        async def rank(self, query: str, passages: list[Passage]) -> list[RerankHit]:
            entered.set()
            try:
                await asyncio.Event().wait()
                return []
            finally:
                cleaning.set()
                await release.wait()
                finished.set()

    client, store = make_client(tmp_path, Dense([passage("A", "a")]), Lexical([]), CleaningRerank())
    task = asyncio.create_task(client.search("ready", "q"))
    try:
        await asyncio.wait_for(entered.wait(), 1)
        task.cancel()
        await asyncio.wait_for(cleaning.wait(), 1)
        task.cancel()
        await asyncio.sleep(0)
    finally:
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await task
    assert finished.is_set()
    assert store.read("session")[-1]["type"] == "hybrid.canceled"
