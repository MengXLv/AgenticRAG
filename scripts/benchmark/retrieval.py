"""同一查询、来源与 K 下对照四种检索策略，定位指定证据组丢失阶段。"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import statistics
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from flowpay.core.json import JsonObject
from flowpay.core.knowledge.contracts import Passage
from flowpay.core.knowledge.fusion import fuse_rrf
from flowpay.core.knowledge.hybrid_contracts import LexicalHit, RerankHit
from flowpay.core.knowledge.ranking import (
    DEFAULT_LIMIT,
    DEFAULT_RERANK_WEIGHT,
    rank_candidates,
)
from pydantic import TypeAdapter

from scripts.benchmark.reranking import EvaluationCase, metrics_at_k, validate_cases

Strategy = Literal["dense", "bm25", "rrf", "hybrid"]
STRATEGIES: tuple[Strategy, ...] = ("dense", "bm25", "rrf", "hybrid")
LossStage = Literal["recall_miss", "fusion_cutoff", "final_selection", "covered"]


@dataclass(frozen=True)
class RetrievalCase:
    evaluation: EvaluationCase
    dense: list[Passage]
    lexical: list[LexicalHit]
    ranks: list[RerankHit]
    # 只接收预热后真实执行的耗时；重复执行不增加质量评测样本数。
    warm_timings_ms: dict[Strategy, list[float]] = field(default_factory=dict)
    measured_rerank_weight: float | None = None
    measured_limit: int | None = None


def attribute_groups(
    groups: list[list[list[str]]],
    recalled: set[str],
    candidates: set[str],
    selected: set[str],
) -> list[LossStage]:
    """一个组有任一完整替代方案就算覆盖，不能把组内片段交集当成覆盖。"""
    statuses: list[LossStage] = []
    for group in groups:
        if not group or any(not alternative for alternative in group):
            raise ValueError("Evidence alternatives cannot be empty")
        if not any(set(alt) <= recalled for alt in group):
            statuses.append("recall_miss")
        elif not any(set(alt) <= candidates for alt in group):
            statuses.append("fusion_cutoff")
        elif not any(set(alt) <= selected for alt in group):
            statuses.append("final_selection")
        else:
            statuses.append("covered")
    return statuses


def _interval(families: dict[str, list[float]], samples: int) -> list[float] | None:
    if len(families) < 2:
        return None
    rng = random.Random(20261027)
    groups = list(families.values())
    estimates = sorted(
        statistics.mean(
            value for group in rng.choices(groups, k=len(groups)) for value in group
        )
        for _ in range(samples)
    )
    return [
        estimates[int(0.025 * (samples - 1))],
        estimates[math.ceil(0.975 * (samples - 1))],
    ]


def _timings(
    cases: list[RetrievalCase], rerank_weight: float, limit: int
) -> JsonObject | None:
    if not any(case.warm_timings_ms for case in cases):
        return None
    counts: set[int] = set()
    result: JsonObject = {}
    for case in cases:
        if set(case.warm_timings_ms) != set(STRATEGIES):
            raise ValueError(
                "Timing measurements must cover all four strategies and all questions"
            )
        if (
            case.measured_rerank_weight is None
            or case.measured_rerank_weight != rerank_weight
            or case.measured_limit != limit
        ):
            raise ValueError(
                "Warm timing policy must match the evaluated rerank weight and limit"
            )
        for values in case.warm_timings_ms.values():
            if len(values) < 3 or any(
                isinstance(value, bool) or not math.isfinite(value) or value < 0
                for value in values
            ):
                raise ValueError(
                    "Need at least three finite nonnegative warm measurements"
                )
            counts.add(len(values))
    if len(counts) != 1:
        raise ValueError("All strategies require the same number of warm measurements")
    for strategy in STRATEGIES:
        values = sorted(
            value for case in cases for value in case.warm_timings_ms[strategy]
        )
        result[strategy] = {
            "measurement_count": len(values),
            "warm_p50_ms": statistics.median(values),
            "warm_p95_ms": values[math.ceil(0.95 * len(values)) - 1],
        }
    return result


def _mean_metrics(measurements: list[JsonObject]) -> JsonObject:
    result: JsonObject = {}
    for key in measurements[0]:
        numbers: list[float] = []
        for row in measurements:
            value = row[key]
            if isinstance(value, (int, float)):
                numbers.append(float(value))
        result[key] = statistics.mean(numbers) if numbers else None
    return result


def analyze(
    cases: list[RetrievalCase],
    *,
    limit: int = DEFAULT_LIMIT,
    recall_limit: int = 50,
    rerank_limit: int = 30,
    rank_constant: int = 60,
    rerank_weight: float = DEFAULT_RERANK_WEIGHT,
    bootstrap_samples: int = 2000,
) -> JsonObject:
    validate_cases([case.evaluation for case in cases], allow_empty=True)
    if type(limit) is not int or not 1 <= limit <= 10:
        raise ValueError("Result limit must be 1..10")
    if type(recall_limit) is not int or not 1 <= recall_limit <= 100:
        raise ValueError("Recall limit must be 1..100")
    if type(rerank_limit) is not int or not 10 <= rerank_limit <= min(
        30, 2 * recall_limit
    ):
        raise ValueError("Rerank limit must be 10..min(30, 2*recall_limit)")
    if type(bootstrap_samples) is not int or bootstrap_samples < 100:
        raise ValueError("Use at least 100 bootstrap samples")
    if (
        len(
            {
                p.source.collection_id
                for case in cases
                for p in case.evaluation.candidates
            }
        )
        > 1
    ):
        raise ValueError("Comparison requires the same collection")
    timings = _timings(cases, rerank_weight, limit)
    rows: list[JsonObject] = []
    losses: Counter[LossStage] = Counter()
    metrics_by_strategy: dict[Strategy, list[JsonObject]] = {
        name: [] for name in STRATEGIES
    }
    for case in cases:
        if len(case.dense) > recall_limit or len(case.lexical) > recall_limit:
            raise ValueError("Captured recall exceeds the configured budget")
        for lane in (case.dense, [hit.passage for hit in case.lexical]):
            if len({p.id for p in lane}) != len(lane):
                raise ValueError("Captured recall must not contain duplicate ids")
        fused = fuse_rrf(case.dense, case.lexical, rank_constant)
        candidates = fused[:rerank_limit]
        if [
            candidate.passage for candidate in candidates
        ] != case.evaluation.candidates:
            raise ValueError("Replayed RRF candidates differ from frozen model input")
        final = rank_candidates(candidates, case.ranks, rerank_weight=rerank_weight)
        strategies: dict[Strategy, list[Passage]] = {
            "dense": case.dense[:limit],
            "bm25": [hit.passage for hit in case.lexical[:limit]],
            "rrf": [candidate.passage for candidate in fused[:limit]],
            "hybrid": [hit.passage for hit in final[:limit]],
        }
        row: JsonObject = {
            "question_id": case.evaluation.question_id,
            "scenario_family_id": case.evaluation.scenario_family_id,
            "split": case.evaluation.split,
            "query": case.evaluation.query,
        }
        for name, selected in strategies.items():
            ids = [p.id for p in selected]
            metrics = metrics_at_k(case.evaluation, ids, limit)
            # 不同策略的已判断片段池不同，这个条件值不能用于四策略对比。
            metrics.pop(f"judged_pool_ndcg{limit}", None)
            metrics_by_strategy[name].append(metrics)
            row[name] = {
                "ids": ids,
                "metrics": metrics,
                "context_chars": sum(len(p.text) for p in selected),
                "distinct_documents": len({p.source.doc_id for p in selected}),
            }
        loss_stages = attribute_groups(
            case.evaluation.evidence_groups,
            {candidate.passage.id for candidate in fused},
            {candidate.passage.id for candidate in candidates},
            {hit.passage.id for hit in final[:limit]},
        )
        row["hybrid_group_loss_stages"] = loss_stages
        losses.update(loss_stages)
        rows.append(row)
    summary: JsonObject = {}
    for name, measurements in metrics_by_strategy.items():
        summary[name] = _mean_metrics(measurements)
    comparisons: JsonObject = {}
    for baseline in ("dense", "rrf"):
        for metric in (
            f"designated_group_coverage{limit}",
            f"designated_positive_recall{limit}",
        ):
            families: dict[str, list[float]] = defaultdict(list)
            for case, proposed, original in zip(
                cases,
                metrics_by_strategy["hybrid"],
                metrics_by_strategy[baseline],
                strict=True,
            ):
                left, right = proposed[metric], original[metric]
                if isinstance(left, (int, float)) and isinstance(right, (int, float)):
                    families[case.evaluation.scenario_family_id].append(
                        float(left) - float(right)
                    )
            values = [value for family in families.values() for value in family]
            comparisons[f"hybrid-minus-{baseline}/{metric}"] = {
                "mean_difference": statistics.mean(values) if values else None,
                "family_bootstrap_95_interval": _interval(families, bootstrap_samples),
                "paired_question_count": len(values),
                "paired_family_count": len(families),
                "regressions": sum(value < 0 for value in values),
            }
    return {
        "question_count": len(cases),
        "scenario_family_count": len(
            {case.evaluation.scenario_family_id for case in cases}
        ),
        "policy": {
            "k": limit,
            "recall_limit": recall_limit,
            "rerank_limit": rerank_limit,
            "rank_constant": rank_constant,
            "rerank_weight": rerank_weight,
        },
        "bootstrap_unit": "scenario_family_id",
        "bootstrap_samples": bootstrap_samples,
        "annotation_scope": "specified evidence groups and known positives; unjudged remains unknown",
        "summary": summary,
        "paired_differences": comparisons,
        "hybrid_group_loss_stages": dict(losses),
        "performance": timings,
        "rows": rows,
    }


class Arguments(argparse.Namespace):
    input: Path = Path()
    output: Path = Path()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(namespace=Arguments())
    input_path = args.input
    output_path = args.output
    raw = input_path.read_bytes()
    cases = TypeAdapter(list[RetrievalCase]).validate_json(
        raw.decode("utf-8-sig"), strict=True
    )
    report = analyze(cases)
    report["input_sha256"] = hashlib.sha256(raw).hexdigest()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"Four-strategy retrieval report: {output_path}")


if __name__ == "__main__":
    main()
