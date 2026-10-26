"""BM25 召回的匹配、集合边界与索引生命周期。"""

from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from datetime import UTC, datetime
from typing import override

import pytest

from flowpay.core.knowledge.contracts import Collection, Passage, PassageSource
from flowpay.core.knowledge.lexical import BM25Retriever
from flowpay.core.knowledge.tokenization import tokenize


def passage(doc_id: str, text: str, collection_id: str = "ready") -> Passage:
    return Passage(
        id=f"doc:{doc_id}#c001",
        title=doc_id,
        text=text,
        score=0.0,
        source=PassageSource(collection_id, doc_id, 1, f"{doc_id}.md", "a" * 64, {}, {}),
    )


class MemorySource:
    def __init__(self, rows: list[Passage]) -> None:
        self.rows: list[Passage] = rows
        self.loads: Counter[str] = Counter()
        self.status: dict[str, str] = {}

    def get_collection(self, collection_id: str) -> Collection:
        return Collection(
            collection_id=collection_id,
            dataset_release_id="fixture",
            dataset_manifest_sha256="a" * 64,
            build_config={},
            dimensions=3,
            status="building" if self.status.get(collection_id) == "building" else "ready",
            created_at=datetime.now(UTC),
            ready_at=None,
        )

    def list_passages(self, collection_id: str) -> list[Passage]:
        self.loads[collection_id] += 1
        return [p for p in self.rows if p.source.collection_id == collection_id]


def test_tokenization_preserves_technical_terms_and_segments_chinese() -> None:
    tokens = tokenize("支付失败 FP-4021，HTTP 410；account_id，v1.2.3")
    assert {"支付", "失败", "fp-4021", "http", "410", "account_id", "v1.2.3"} <= set(tokens)
    assert tokenize("  ，！？... ") == []
    assert tokenize("WEBHOOK") == tokenize("webhook")


@pytest.mark.parametrize(
    ("query", "target"),
    [
        ("支付失败", "PAY"),
        ("FP-4021", "ERROR"),
        ("HTTP 410", "HTTP"),
        ("account_id", "FIELD"),
        ("v1.2.3", "VERSION"),
    ],
)
def test_chinese_and_technical_queries_retrieve_target(query: str, target: str) -> None:
    source = MemorySource(
        [
            passage("PAY", "支付失败后查询支付状态"),
            passage("ERROR", "FP-4021 表示幂等键冲突"),
            passage("HTTP", "HTTP 410 表示端点已经不存在"),
            passage("FIELD", "account_id 是账户字段"),
            passage("VERSION", "v1.2.3 新增事件处理"),
        ]
    )
    hits = BM25Retriever(source).search("ready", query, limit=3)
    assert target in [hit.passage.source.doc_id for hit in hits]
    assert len(hits) <= 3
    assert all(hit.passage.score == 0.0 for hit in hits)


def test_no_match_returns_empty_and_negative_idf_does_not_drop_matches() -> None:
    retriever = BM25Retriever(MemorySource([passage("A", "webhook"), passage("B", "webhook")]))
    assert retriever.search("ready", "FP-9999") == []
    hits = retriever.search("ready", "webhook")
    assert [hit.passage.source.doc_id for hit in hits] == ["A", "B"]
    assert all(hit.score <= 0 for hit in hits)


def test_same_doc_id_is_isolated_by_collection_and_indexes_are_reused() -> None:
    source = MemorySource([passage("A", "FP-4021", "one"), passage("A", "account_id", "two")])
    retriever = BM25Retriever(source)
    assert retriever.search("one", "account_id") == []
    assert retriever.search("two", "FP-4021") == []
    assert retriever.search("one", "FP-4021")[0].passage.source.collection_id == "one"
    assert retriever.search("two", "account_id")[0].passage.source.collection_id == "two"
    assert source.loads == {"one": 1, "two": 1}


def test_building_collection_is_rejected_before_loading_and_can_retry() -> None:
    source = MemorySource([passage("A", "webhook")])
    source.status["ready"] = "building"
    retriever = BM25Retriever(source)
    with pytest.raises(ValueError, match="not ready"):
        retriever.search("ready", "webhook")
    assert not source.loads
    source.status["ready"] = "ready"
    assert len(retriever.search("ready", "webhook")) == 1


def test_concurrent_queries_build_index_once() -> None:
    source = MemorySource([passage("A", "webhook")])
    retriever = BM25Retriever(source)
    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = [pool.submit(retriever.search, "ready", "webhook") for _ in range(24)]
        results = [future.result() for future in futures]
    assert all(len(result) == 1 for result in results)
    assert source.loads == {"ready": 1}


def test_empty_collection_and_punctuation_only_content() -> None:
    source = MemorySource([passage("A", "... ！！！")])
    retriever = BM25Retriever(source)
    assert retriever.search("empty", "webhook") == []
    assert retriever.search("ready", "webhook") == []
    assert retriever.search("ready", "...") == []


def test_ties_are_stable_and_input_order_and_passages_are_not_modified() -> None:
    source = MemorySource([passage("B", "webhook"), passage("A", "webhook")])
    source.rows[0] = replace(source.rows[0], score=0.123)
    hits = BM25Retriever(source).search("ready", "webhook", limit=1)
    assert [hit.passage.source.doc_id for hit in hits] == ["A"]
    assert source.rows[0].score == 0.123
    assert [p.source.doc_id for p in source.rows] == ["B", "A"]


@pytest.mark.parametrize("limit", [True, 0, 101, 1.5])
def test_invalid_limit_is_rejected_before_loading(limit: int) -> None:
    source = MemorySource([])
    with pytest.raises(ValueError, match="limit"):
        BM25Retriever(source).search("ready", "webhook", limit=limit)
    assert not source.loads


def test_blank_query_is_rejected_before_loading() -> None:
    source = MemorySource([])
    with pytest.raises(ValueError, match="query"):
        BM25Retriever(source).search("ready", "  ")
    assert not source.loads


def test_source_failure_is_propagated_and_not_cached() -> None:
    class FailingSource(MemorySource):
        @override
        def list_passages(self, collection_id: str) -> list[Passage]:
            if not self.loads:
                self.loads[collection_id] += 1
                raise RuntimeError("database unavailable")
            return super().list_passages(collection_id)

    source = FailingSource([passage("A", "webhook")])
    retriever = BM25Retriever(source)
    with pytest.raises(RuntimeError, match="database unavailable"):
        retriever.search("ready", "webhook")
    assert len(retriever.search("ready", "webhook")) == 1


@pytest.mark.parametrize("duplicate", [False, True])
def test_malformed_source_cannot_mix_collections_or_duplicate_passages(duplicate: bool) -> None:
    class MalformedSource(MemorySource):
        @override
        def list_passages(self, collection_id: str) -> list[Passage]:
            row = passage("A", "webhook", collection_id if duplicate else "another")
            return [row, row] if duplicate else [row]

    with pytest.raises(ValueError, match="duplicate" if duplicate else "another collection"):
        BM25Retriever(MalformedSource([])).search("ready", "webhook")
