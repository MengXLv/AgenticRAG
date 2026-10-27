"""固定候选池的真实重排诊断；部分标注不产生正式 nDCG。"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import importlib.metadata
import json
import math
import statistics
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

from flowpay.core.agent.trace import Tracer
from flowpay.core.event_log import EventStore
from flowpay.core.event_log.store import new_id
from flowpay.core.json import JsonObject, require_int, require_list, require_object
from flowpay.core.knowledge.contracts import Passage
from flowpay.core.llm.reranking.client import CrossEncoderReranker
from flowpay.core.llm.reranking.settings import RerankingSettings
from pydantic import TypeAdapter


@dataclass(frozen=True)
class EvaluationCase:
    question_id: str
    split: Literal["development", "holdout"]
    scenario_family_id: str
    query: str
    candidates: list[Passage]
    grades: dict[str, int]
    evidence_groups: list[list[list[str]]]
    judgments_complete: bool = False


def _ndcg(
    ids: list[str], grades: dict[str, int], ideal: list[int], k: int
) -> float | None:
    denominator = sum(
        (2**grade - 1) / math.log2(i + 2) for i, grade in enumerate(ideal[:k])
    )
    if denominator == 0:
        return None
    return (
        sum((2 ** grades[pid] - 1) / math.log2(i + 2) for i, pid in enumerate(ids[:k]))
        / denominator
    )


def metrics(case: EvaluationCase, ids: list[str]) -> JsonObject:
    k = 5
    top = ids[:k]
    selected = set(top)
    covered = sum(
        any(set(alternative) <= selected for alternative in group)
        for group in case.evidence_groups
    )
    positives = {pid for pid, grade in case.grades.items() if grade == 2}
    judged = [pid for pid in ids if pid in case.grades]
    ideal_judged = sorted((case.grades[pid] for pid in judged), reverse=True)
    # 这个条件指标只比较已标注候选的相对排序，不能当作实际返回前5的质量。
    conditional_ndcg = _ndcg(judged, case.grades, ideal_judged, k)
    formal_ndcg = None
    if case.judgments_complete and all(pid in case.grades for pid in ids):
        formal_ndcg = _ndcg(
            ids, case.grades, sorted(case.grades.values(), reverse=True), k
        )
    return {
        "designated_group_coverage5": covered / len(case.evidence_groups)
        if case.evidence_groups
        else None,
        "all_designated_groups5": covered == len(case.evidence_groups)
        if case.evidence_groups
        else None,
        "designated_positive_recall5": len(positives & selected) / len(positives)
        if positives
        else None,
        "known_positive_mrr5": next(
            (1 / (i + 1) for i, pid in enumerate(top) if pid in positives), 0.0
        ),
        "unjudged_top5_count": sum(pid not in case.grades for pid in top),
        "judged_pool_ndcg5": conditional_ndcg,
        "ndcg5": formal_ndcg,
    }


def validate_cases(cases: list[EvaluationCase]) -> None:
    families: dict[str, str] = {}
    identifiers: set[str] = set()
    if not cases:
        raise ValueError("Provide evaluation cases")
    for case in cases:
        if case.question_id in identifiers:
            raise ValueError("Duplicate question id")
        identifiers.add(case.question_id)
        if families.setdefault(case.scenario_family_id, case.split) != case.split:
            raise ValueError("Scenario family crosses development/holdout splits")
        if not case.candidates or len(case.candidates) > 30:
            raise ValueError("Each case needs 1..30 frozen RRF candidates")
        keys = [
            (p.source.collection_id, p.source.doc_id, p.source.chunk_no)
            for p in case.candidates
        ]
        if len(set(keys)) != len(keys) or len({key[0] for key in keys}) != 1:
            raise ValueError("Candidates must be unique and from one collection")
        if len({p.id for p in case.candidates}) != len(case.candidates):
            raise ValueError("Candidate ids must be unique")
        if any(
            type(grade) is not int or grade not in {0, 1, 2}
            for grade in case.grades.values()
        ):
            raise ValueError("Grades must be integers 0..2")
        if any(
            not group or any(not alt for alt in group) for group in case.evidence_groups
        ):
            raise ValueError("Evidence alternatives cannot be empty")


def summarize(rows: list[JsonObject]) -> JsonObject:
    report: JsonObject = {}
    for split in ("development", "holdout"):
        group = [row for row in rows if row["split"] == split]
        for strategy in ("rrf", "reranked"):
            for name in (
                "designated_group_coverage5",
                "all_designated_groups5",
                "designated_positive_recall5",
                "known_positive_mrr5",
                "judged_pool_ndcg5",
            ):
                values = [require_object(row[strategy])[name] for row in group]
                numbers = [
                    float(value) for value in values if isinstance(value, (int, float))
                ]
                report[f"{split}/{strategy}/{name}"] = (
                    statistics.mean(numbers) if numbers else None
                )
    return report


async def evaluate(
    cases: list[EvaluationCase],
    config: RerankingSettings,
    output: Path,
    repeats: int = 3,
) -> None:
    import torch

    validate_cases(cases)
    if repeats < 1:
        raise ValueError("repeats must be positive")
    output.mkdir(parents=True, exist_ok=True)
    events = EventStore(output / "events.sqlite")
    session_id = new_id("reranking-eval")
    client = CrossEncoderReranker(config, Tracer(events, session_id, new_id("run")))
    started = time.perf_counter()
    await client.prepare()
    load_ms = (time.perf_counter() - started) * 1000
    if config.device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    rows: list[JsonObject] = []
    timings: list[float] = []
    first_inference_ms: float | None = None
    for index, case in enumerate(cases):
        started = time.perf_counter()
        ranks = await client.rank(case.query, case.candidates)
        elapsed = (time.perf_counter() - started) * 1000
        if first_inference_ms is None:
            first_inference_ms = elapsed
        # 首次前向不混入热态分位数；每题再执行同样次数的热态推理。
        for _ in range(repeats):
            started = time.perf_counter()
            await client.rank(case.query, case.candidates)
            timings.append((time.perf_counter() - started) * 1000)
        ranks = sorted(
            ranks,
            key=lambda r: (
                -r.score,
                case.candidates[r.candidate_index].source.doc_id,
                case.candidates[r.candidate_index].source.chunk_no,
            ),
        )
        baseline = [p.id for p in case.candidates]
        reordered = [case.candidates[r.candidate_index].id for r in ranks]
        rows.append(
            {
                "question_id": case.question_id,
                "split": case.split,
                "rrf": metrics(case, baseline),
                "reranked": metrics(case, reordered),
                "rrf_ids": baseline,
                "reranked_ids": reordered,
                "scores": [asdict(rank) for rank in ranks],
            }
        )
        print(f"{index + 1}/{len(cases)} {case.question_id}", flush=True)
    response_events = [
        e for e in events.read(session_id) if e["type"] == "rerank.response"
    ]
    report: JsonObject = {
        "event_session_id": session_id,
        "settings": asdict(config) | {"cache_dir": str(config.cache_dir)},
        "versions": {
            name: importlib.metadata.version(name)
            for name in ("torch", "transformers", "tokenizers")
        },
        "model_loads": sum(
            e["payload"]["model_loaded"] is True for e in response_events
        ),
        "load_ms_excluding_torch_import": load_ms,
        "first_inference_ms": first_inference_ms,
        "warm_p50_ms": statistics.median(timings),
        "warm_p95_ms": sorted(timings)[math.ceil(0.95 * len(timings)) - 1],
        "gpu_peak_allocated_bytes": torch.cuda.max_memory_allocated()
        if config.device == "cuda"
        else None,
        "query_truncations": sum(
            require_int(e["payload"]["query_tokens"])
            > require_int(e["payload"]["query_tokens_used"])
            for e in response_events
        ),
        "truncated_passage_pairs": sum(
            sum(
                original != used
                for original, used in zip(
                    require_list(e["payload"]["passage_tokens"]),
                    require_list(e["payload"]["passage_tokens_used"]),
                    strict=True,
                )
            )
            for e in response_events
        ),
        "annotation_scope": "partial dev judgments; holdout is a family-disjoint diagnostic subset of dev, not official test",
        "ndcg_policy": "ndcg5 is null unless corpus judgments are complete; judged_pool_ndcg5 is conditional only",
        "summary": summarize(rows),
        "rows": rows,
    }
    (output / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )


class Arguments(argparse.Namespace):
    def __init__(self) -> None:
        super().__init__()
        self.input: Path = Path()
        self.config: Path = Path("backend/configs/reranking.toml")
        self.output: Path = Path()
        self.repeats: int = 3


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Arguments().config)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args(namespace=Arguments())
    raw = args.input.read_bytes()
    cases = TypeAdapter(list[EvaluationCase]).validate_json(raw)
    asyncio.run(
        evaluate(cases, RerankingSettings.load(args.config), args.output, args.repeats)
    )
    (args.output / "input.sha256").write_text(
        hashlib.sha256(raw).hexdigest(), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
