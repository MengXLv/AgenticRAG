"""回放两路排名，输出 RRF Top-N 的保留、截断及标注证据损失。"""

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

from flowpay.core.json import JsonObject
from flowpay.core.knowledge.contracts import Passage
from flowpay.core.knowledge.fusion import fuse_rrf
from flowpay.core.knowledge.hybrid_contracts import LexicalHit
from pydantic import TypeAdapter


@dataclass(frozen=True)
class FusionCase:
    case_id: str
    query: str
    dense: list[Passage]
    lexical: list[LexicalHit]
    expected_ids: list[str] | None = None


def inspect(case: FusionCase, top_n: int = 30, rank_constant: int = 60) -> JsonObject:
    """无标注时保留未知状态；证据未召回和融合截断分别记录。"""
    if type(top_n) is not int or top_n < 1:
        raise ValueError("top_n must be a positive integer")
    ranked = fuse_rrf(case.dense, case.lexical, rank_constant)
    all_ids = {row.passage.id for row in ranked}
    selected = ranked[:top_n]
    selected_ids = {row.passage.id for row in selected}
    expected = set(case.expected_ids) if case.expected_ids is not None else None
    return {
        "case_id": case.case_id,
        "query": case.query,
        "rank_constant": rank_constant,
        "top_n": top_n,
        "dense_ids": [p.id for p in case.dense],
        "bm25_ids": [hit.passage.id for hit in case.lexical],
        "union_count": len(ranked),
        "selected_count": len(selected),
        "dropped_ids": [row.passage.id for row in ranked[top_n:]],
        "expected_ids": sorted(expected) if expected is not None else None,
        "expected_not_recalled": sorted(expected - all_ids)
        if expected is not None
        else None,
        "expected_lost_at_cutoff": (
            sorted((expected & all_ids) - selected_ids)
            if expected is not None
            else None
        ),
        "candidates": [
            {
                "id": row.passage.id,
                "collection_id": row.passage.source.collection_id,
                "dense_rank": row.dense_rank,
                "dense_score": row.dense_score,
                "bm25_rank": row.bm25_rank,
                "bm25_score": row.bm25_score,
                "rrf_score": row.rrf_score,
                "selected": position < top_n,
            }
            for position, row in enumerate(ranked)
        ],
    }


class Arguments(argparse.Namespace):
    def __init__(self) -> None:
        super().__init__()
        self.input: Path = Path()
        self.output: Path = Path()
        self.top_n: int = 30
        self.rank_constant: int = 60


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input", type=Path, required=True, help="FusionCase JSON 数组"
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--top-n", type=int, default=30)
    parser.add_argument("--rank-constant", type=int, default=60)
    args = parser.parse_args(namespace=Arguments())
    cases = TypeAdapter(list[FusionCase]).validate_json(
        args.input.read_text(encoding="utf-8")
    )
    reports = [inspect(case, args.top_n, args.rank_constant) for case in cases]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(reports, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"RRF replay report: {args.output}")


if __name__ == "__main__":
    main()
