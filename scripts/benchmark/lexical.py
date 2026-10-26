"""只读验证指定 ready 集合的 BM25 初始化、索引复用和查询性能。"""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import statistics
import time
import tracemalloc
from pathlib import Path

from flowpay.core.json import JsonObject
from flowpay.core.knowledge import PostgresVectorStore
from flowpay.core.knowledge.contracts import Collection, Passage
from flowpay.core.knowledge.hybrid_contracts import PassageSourceReader
from flowpay.core.knowledge.lexical import BM25Retriever


class _MeasuredSource:
    def __init__(self, source: PassageSourceReader) -> None:
        self.source: PassageSourceReader = source
        self.load_calls: int = 0
        self.passage_count: int = 0

    def get_collection(self, collection_id: str) -> Collection:
        return self.source.get_collection(collection_id)

    def list_passages(self, collection_id: str) -> list[Passage]:
        passages = self.source.list_passages(collection_id)
        self.load_calls += 1
        self.passage_count = len(passages)
        return passages


def measure(
    source: PassageSourceReader,
    collection_id: str,
    queries: list[str],
    repeats: int = 20,
    *,
    track_memory: bool = True,
) -> JsonObject:
    """性能数据只描述当前集合和机器；不执行向量请求或修改知识内容。"""
    if not queries or any(not query.strip() for query in queries) or repeats < 1:
        raise ValueError("Provide nonblank queries and positive repeats")
    if track_memory and tracemalloc.is_tracing():
        raise RuntimeError("Memory measurement requires its own tracemalloc session")
    measured = _MeasuredSource(source)
    retriever = BM25Retriever(measured)
    if track_memory:
        tracemalloc.start()
    retained_bytes: int | None = None
    peak_bytes: int | None = None
    started = time.perf_counter()
    try:
        retriever.prepare(collection_id)
        initialization_ms = (time.perf_counter() - started) * 1000
        if track_memory:
            retained_bytes, peak_bytes = tracemalloc.get_traced_memory()
    finally:
        if track_memory:
            tracemalloc.stop()
    # 预热一次；热态计时关闭 tracemalloc，避免分配跟踪放大查询延迟。
    for query in queries:
        retriever.search(collection_id, query, limit=5)
    timings: list[float] = []
    samples: list[JsonObject] = []
    for query in queries:
        for _ in range(repeats):
            started = time.perf_counter()
            retriever.search(collection_id, query, limit=5)
            timings.append((time.perf_counter() - started) * 1000)
        hits = retriever.search(collection_id, query, limit=5)
        samples.append({"query": query, "top_ids": [hit.passage.id for hit in hits]})
    return {
        "collection_id": collection_id,
        "passage_count": measured.passage_count,
        "index_load_calls": measured.load_calls,
        "initialization_ms": initialization_ms,
        "memory_instrumentation": track_memory,
        "python_retained_bytes": retained_bytes,
        "python_peak_bytes": peak_bytes,
        "memory_scope": "tracemalloc allocations during prepare; not full process RSS",
        "python": platform.python_version(),
        "platform": platform.platform(),
        "timed_queries": len(timings),
        "warm_p50_ms": statistics.median(timings),
        "warm_p95_ms": sorted(timings)[math.ceil(len(timings) * 0.95) - 1],
        "samples": samples,
    }


class Arguments(argparse.Namespace):
    def __init__(self) -> None:
        super().__init__()
        self.collection: str = ""
        self.dsn_env: str = "FLOWPAY_KNOWLEDGE_DSN"
        self.query: list[str] = []
        self.repeats: int = 20
        self.output: Path = Path()
        self.no_memory: bool = False


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--collection", required=True)
    parser.add_argument("--dsn-env", default="FLOWPAY_KNOWLEDGE_DSN")
    parser.add_argument("--query", action="append", required=True)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--no-memory", action="store_true", help="独立进程测量未插桩初始化时间"
    )
    args = parser.parse_args(namespace=Arguments())
    store = PostgresVectorStore(os.environ[args.dsn_env])
    try:
        report = measure(
            store,
            args.collection,
            args.query,
            args.repeats,
            track_memory=not args.no_memory,
        )
    finally:
        store.close()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"BM25 verification report: {args.output}")


if __name__ == "__main__":
    main()
