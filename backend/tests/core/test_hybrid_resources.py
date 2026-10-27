import json
from pathlib import Path

import pytest

from flowpay.core.agent.trace import Tracer
from flowpay.core.config import Settings, load_settings
from flowpay.core.event_log import EventStore
from flowpay.core.execution.resources import RuntimeResources
from flowpay.core.knowledge.hybrid import HybridKnowledgeClient
from flowpay.core.llm.reranking.client import CrossEncoderReranker
from flowpay.core.llm.reranking.settings import RerankingSettings

from .test_embeddings import settings as embedding_settings
from .test_lexical import MemorySource, passage
from .test_reranking import FakeBackend


class Store(MemorySource):
    def __init__(self) -> None:
        super().__init__([passage("A", "1")])
        self.closed: bool = False

    def close(self) -> None:
        self.closed = True


def config(tmp_path: Path) -> Path:
    path = tmp_path / "reranking.toml"
    path.write_text(
        '[reranking]\nmodel_id="model"\nrevision="' + "a" * 40 + '"\n', encoding="utf-8"
    )
    return path


def test_config_opt_in_relative_paths_and_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "local.toml"
    path.write_text('data_dir="state"\ndataset_path="data"\n', encoding="utf-8")
    assert load_settings(path).hybrid_reranking_config is None
    monkeypatch.setenv("FLOWPAY_HYBRID_RERANKING_CONFIG", "reranking.toml")
    loaded = load_settings(path)
    assert loaded.hybrid_reranking_config == tmp_path / "reranking.toml"
    assert json.loads(json.dumps(loaded.snapshot()))["hybrid_reranking_config"] == str(
        tmp_path / "reranking.toml"
    )


def test_disabled_does_not_load_model_or_index(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def forbidden(_: RerankingSettings) -> FakeBackend:
        raise AssertionError("optional model loaded while disabled")

    monkeypatch.setattr("flowpay.core.execution.resources.RerankingModel.prepare", forbidden)
    resources = RuntimeResources.from_settings(Settings(tmp_path, tmp_path))
    assert (
        resources.hybrid_document_retriever(Tracer(EventStore(tmp_path / "e.sqlite"), "s", "r"))
        is None
    )
    resources.close()


async def test_startup_prepares_once_and_clients_share_only_model_and_index(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = Store()
    monkeypatch.setenv("FLOWPAY_KNOWLEDGE_DSN", "fixture")

    def open_store(_: str) -> Store:
        return store

    monkeypatch.setattr("flowpay.core.knowledge.PostgresVectorStore", open_store)
    loads: list[RerankingSettings] = []

    def backend(settings: RerankingSettings) -> FakeBackend:
        loads.append(settings)
        return FakeBackend()

    monkeypatch.setattr(
        "flowpay.core.llm.reranking.transformers_backend.TransformersPairScorer", backend
    )
    settings = Settings(
        tmp_path,
        tmp_path,
        workflow="s2",
        knowledge_collection_id="ready",
        embedding=embedding_settings(),
        hybrid_reranking_config=config(tmp_path),
    )
    resources = RuntimeResources.from_settings(settings)
    events = EventStore(tmp_path / "e.sqlite")
    clients = [
        resources.hybrid_document_retriever(Tracer(events, run, run)) for run in ("one", "two")
    ]
    assert all(isinstance(client, HybridKnowledgeClient) for client in clients)
    first, second = clients
    assert isinstance(first, HybridKnowledgeClient) and isinstance(second, HybridKnowledgeClient)
    assert first.lexical is second.lexical
    assert isinstance(first.reranker, CrossEncoderReranker)
    assert isinstance(second.reranker, CrossEncoderReranker)
    assert first.reranker.model is second.reranker.model
    assert first.tracer is not second.tracer
    assert len(loads) == 1 and store.loads["ready"] == 1
    assert "hybrid_reranking" in await resources.snapshot()
    resources.close()
    assert store.closed


def test_enabled_missing_backend_fails_startup_and_closes_store(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = Store()
    monkeypatch.setenv("FLOWPAY_KNOWLEDGE_DSN", "fixture")

    def open_store(_: str) -> Store:
        return store

    monkeypatch.setattr("flowpay.core.knowledge.PostgresVectorStore", open_store)

    def missing(_: RerankingSettings) -> FakeBackend:
        raise ImportError("reranking dependencies unavailable")

    monkeypatch.setattr(
        "flowpay.core.llm.reranking.transformers_backend.TransformersPairScorer", missing
    )
    with pytest.raises(ImportError, match="dependencies unavailable"):
        RuntimeResources.from_settings(
            Settings(
                tmp_path,
                tmp_path,
                knowledge_collection_id="ready",
                embedding=embedding_settings(),
                hybrid_reranking_config=config(tmp_path),
            )
        )
    assert store.closed
