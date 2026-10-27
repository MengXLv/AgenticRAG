from pathlib import Path

import pytest

from flowpay.core.agent.trace import Tracer
from flowpay.core.event_log import EventStore
from flowpay.core.json import JsonObject
from flowpay.core.knowledge.hybrid_contracts import HybridHit, HybridRanking
from flowpay.core.knowledge.hybrid_tools import (
    HybridDocumentSearchArguments,
    HybridDocumentSearchResult,
    hybrid_document_search_tool,
)
from flowpay.core.llm.reranking.client import CrossEncoderReranker, RerankingModel
from flowpay.core.llm.reranking.settings import RerankingSettings
from flowpay.core.tools import ToolInputError

from .test_lexical import passage
from .test_reranking import FakeBackend, settings


async def test_projection_keeps_ranking_separate_from_similarity() -> None:
    async def handle(arguments: HybridDocumentSearchArguments) -> HybridDocumentSearchResult:
        assert arguments.limit == 6
        return HybridDocumentSearchResult(
            arguments.query,
            [HybridHit(passage("A", "text"), HybridRanking(0.8, 1, 2.0, 1, 0.03, -4.0, 1, 1, 0.9))],
        )

    payload = await hybrid_document_search_tool(handle).execute({"query": "query"})
    assert payload == {
        "query": "query",
        "passages": [
            {
                "id": "doc:A#c001",
                "title": "A",
                "text": "text",
                "rerank_score": -4.0,
                "final_score": 0.9,
            }
        ],
    }


@pytest.mark.parametrize(
    "arguments",
    [
        {"query": " "},
        {"query": "x" * 1001},
        {"query": "q", "limit": True},
        {"query": "q", "limit": 1.5},
        {"query": "q", "limit": 0},
        {"query": "q", "limit": 11},
        {"query": "q", "collection_id": "other"},
    ],
)
async def test_invalid_input_never_calls_handler(arguments: JsonObject) -> None:
    async def handle(_: HybridDocumentSearchArguments) -> HybridDocumentSearchResult:
        raise AssertionError("invalid input reached handler")

    with pytest.raises(ToolInputError):
        await hybrid_document_search_tool(handle).execute(arguments)


async def test_shared_model_preserves_per_run_trace(tmp_path: Path) -> None:
    loads: list[RerankingSettings] = []

    def factory(config: RerankingSettings) -> FakeBackend:
        loads.append(config)
        return FakeBackend()

    shared = RerankingModel(settings(), factory=factory)
    shared.prepare()
    store = EventStore(tmp_path / "events.sqlite")
    for run in ("one", "two"):
        client = CrossEncoderReranker(settings(), Tracer(store, run, run), model=shared)
        assert (await client.rank(run, [passage("A", "1")]))[0].score == 1.0
        assert [e["type"] for e in store.read(run)] == ["rerank.request", "rerank.response"]
        assert store.read(run)[0]["payload"]["query"] == run
    assert len(loads) == 1
