"""BM25 专项验证不调用向量服务，也不在重复查询时重读片段。"""

import pytest
from scripts.benchmark.lexical import measure

from tests.core.test_lexical import MemorySource, passage


def test_profile_reports_index_reuse_and_real_query_samples() -> None:
    source = MemorySource([passage("HTTP", "HTTP 410"), passage("FIELD", "account_id")])
    report = measure(source, "ready", ["HTTP 410", "account_id"], repeats=2)
    assert report["passage_count"] == 2
    assert report["index_load_calls"] == 1
    assert source.loads == {"ready": 1}
    assert report["timed_queries"] == 4
    assert report["samples"] == [
        {"query": "HTTP 410", "top_ids": ["doc:HTTP#c001"]},
        {"query": "account_id", "top_ids": ["doc:FIELD#c001"]},
    ]
    assert report["memory_scope"] == "tracemalloc allocations during prepare; not full process RSS"


@pytest.mark.parametrize("queries,repeats", [([], 2), ([" "], 2), (["HTTP"], 0)])
def test_invalid_profile_input(queries: list[str], repeats: int) -> None:
    with pytest.raises(ValueError):
        measure(MemorySource([]), "ready", queries, repeats)


def test_uninstrumented_run_does_not_report_memory_as_zero() -> None:
    report = measure(MemorySource([]), "ready", ["HTTP"], 1, track_memory=False)
    assert report["memory_instrumentation"] is False
    assert report["python_peak_bytes"] is None
    assert report["python_retained_bytes"] is None
