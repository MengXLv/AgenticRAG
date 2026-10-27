"""在相同候选、相同模型分数和同一 K 预算下验证最终排序修复。"""

import argparse
import hashlib
import json
import sqlite3
import statistics
import time
from dataclasses import replace
from pathlib import Path

from flowpay.core.json import (
    JsonObject,
    parse_json,
    require_int,
    require_list,
    require_number,
    require_object,
    require_text,
)
from flowpay.core.knowledge.hybrid_contracts import RankedCandidate, RerankHit
from flowpay.core.knowledge.ranking import (
    DEFAULT_LIMIT,
    DEFAULT_RERANK_WEIGHT,
    rank_candidates,
)
from pydantic import TypeAdapter

from scripts.benchmark.reranking import EvaluationCase, metrics_at_k, validate_cases


def token_lengths(report_path: Path, report: JsonObject) -> dict[str, list[int]]:
    database = report_path.with_name("events.sqlite")
    if not database.exists():
        return {}
    connection = sqlite3.connect(database.resolve().as_uri() + "?mode=ro", uri=True)
    try:
        rows = TypeAdapter(list[tuple[str]]).validate_python(
            connection.execute(
                "SELECT data FROM events WHERE session_id=? ORDER BY seq",
                (require_text(report["event_session_id"]),),
            ).fetchall()
        )
    finally:
        connection.close()
    requests: dict[str, str] = {}
    result: dict[str, list[int]] = {}
    for (raw,) in rows:
        event = require_object(parse_json(raw))
        payload = require_object(event["payload"])
        if event["type"] == "rerank.request":
            requests[require_text(event["call_id"])] = require_text(payload["query"])
        elif event["type"] == "rerank.response":
            query = requests[require_text(event["call_id"])]
            lengths = [
                require_int(value) for value in require_list(payload["passage_tokens"])
            ]
            if query in result and result[query] != lengths:
                raise ValueError("Inconsistent token lengths across repeated queries")
            result[query] = lengths
    return result


def analyze(
    cases: list[EvaluationCase],
    scored_rows: list[JsonObject],
    *,
    rerank_weight: float = DEFAULT_RERANK_WEIGHT,
    limit: int = DEFAULT_LIMIT,
    lengths_by_query: dict[str, list[int]] | None = None,
) -> JsonObject:
    validate_cases(cases)
    scored = {require_text(row["question_id"]): row for row in scored_rows}
    if len(scored) != len(scored_rows) or set(scored) != {
        case.question_id for case in cases
    }:
        raise ValueError("Scored questions must match input exactly")
    rows: list[JsonObject] = []
    for case in cases:
        raw = scored[case.question_id]
        baseline = [passage.id for passage in case.candidates]
        if raw["rrf_ids"] != baseline:
            raise ValueError("Scored candidate order differs from frozen input")
        ranks = [
            RerankHit(
                require_int(require_object(item)["candidate_index"]),
                require_number(require_object(item)["score"]),
            )
            for item in require_list(raw["scores"])
        ]
        candidates = [RankedCandidate(passage) for passage in case.candidates]
        started = time.perf_counter()
        fixed = rank_candidates(candidates, ranks, rerank_weight=rerank_weight)
        ranking_ms = (time.perf_counter() - started) * 1000
        pure = rank_candidates(candidates, ranks, rerank_weight=1)
        pure_ids = [hit.passage.id for hit in pure]
        if pure_ids != raw["reranked_ids"]:
            raise ValueError("Cached model ranking cannot be reproduced")
        strategies = {
            "rrf": baseline,
            "cross_encoder": pure_ids,
            "fixed": [hit.passage.id for hit in fixed],
        }
        lengths = (lengths_by_query or {}).get(case.query)
        if lengths is not None and len(lengths) != len(baseline):
            raise ValueError("Token counts do not match candidate pool")
        tokens = (
            dict(zip(baseline, lengths, strict=True)) if lengths is not None else None
        )
        row: JsonObject = {
            "question_id": case.question_id,
            "split": case.split,
            "ranking_ms": ranking_ms,
        }
        for strategy, ids in strategies.items():
            row[strategy] = {
                "metrics": metrics_at_k(case, ids, limit),
                "ids": ids,
                "context_tokens": sum(tokens[pid] for pid in ids[:limit])
                if tokens
                else None,
                "context_chars": sum(
                    len(p.text) for p in case.candidates if p.id in ids[:limit]
                ),
            }
        rows.append(row)
    summary: JsonObject = {}
    for split in ("development", "holdout", "all"):
        selected = [row for row in rows if split == "all" or row["split"] == split]
        for strategy in ("rrf", "cross_encoder", "fixed"):
            metric_names = (
                f"designated_group_coverage{limit}",
                f"all_designated_groups{limit}",
                f"designated_positive_recall{limit}",
                f"known_positive_mrr{limit}",
                f"unjudged_top{limit}_count",
            )
            for name in metric_names:
                values = [
                    require_object(require_object(row[strategy])["metrics"])[name]
                    for row in selected
                ]
                numeric = [float(v) for v in values if isinstance(v, (int, float))]
                summary[f"{split}/{strategy}/{name}"] = (
                    statistics.mean(numeric) if numeric else None
                )
            counts = [
                require_object(row[strategy])["context_tokens"] for row in selected
            ]
            numeric = [float(value) for value in counts if isinstance(value, int)]
            summary[f"{split}/{strategy}/context_tokens"] = (
                statistics.mean(numeric) if numeric else None
            )
    return {
        "policy": "normalized_score_blend",
        "rerank_weight": rerank_weight,
        "k": limit,
        "summary": summary,
        "rows": rows,
        "annotation_scope": "designated evidence groups and partial labels; unknown is not irrelevant",
    }


class Arguments(argparse.Namespace):
    def __init__(self) -> None:
        super().__init__()
        self.input: Path = Path()
        self.scores: Path = Path()
        self.output: Path = Path()
        self.judgments: Path | None = None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--scores", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--judgments", type=Path)
    args = parser.parse_args(namespace=Arguments())
    raw = args.input.read_bytes()
    if (
        args.scores.with_name("input.sha256").read_text(encoding="utf-8").strip()
        != hashlib.sha256(raw).hexdigest()
    ):
        raise ValueError("Scores are not bound to this frozen candidate input")
    cases = TypeAdapter(list[EvaluationCase]).validate_json(raw.decode("utf-8-sig"))
    if args.judgments is not None:
        supplemental = TypeAdapter(dict[str, dict[str, int]]).validate_json(
            args.judgments.read_text(encoding="utf-8-sig"), strict=True
        )
        cases = [
            replace(case, grades=case.grades | supplemental.get(case.question_id, {}))
            for case in cases
        ]
    source = require_object(parse_json(args.scores.read_bytes()))
    report = analyze(
        cases,
        [require_object(row) for row in require_list(source["rows"])],
        lengths_by_query=token_lengths(args.scores, source),
    )
    report["input_sha256"] = hashlib.sha256(raw).hexdigest()
    report["scores_sha256"] = hashlib.sha256(args.scores.read_bytes()).hexdigest()
    report["judgments_sha256"] = (
        hashlib.sha256(args.judgments.read_bytes()).hexdigest()
        if args.judgments
        else None
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"Final-ranking verification: {args.output}")


if __name__ == "__main__":
    main()
