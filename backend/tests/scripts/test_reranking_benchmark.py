"""部分标注的诊断指标不能冒充正式检索评测。"""

import importlib.metadata
import json
import sys
from dataclasses import replace
from pathlib import Path
from types import ModuleType

import pytest
from scripts.benchmark import reranking as benchmark
from scripts.benchmark.reranking import EvaluationCase, metrics, summarize, validate_cases

from flowpay.core.event_log.contracts import EventSink
from flowpay.core.llm.reranking.client import CrossEncoderReranker
from flowpay.core.llm.reranking.settings import RerankingSettings
from tests.core.test_lexical import passage
from tests.core.test_reranking import FakeBackend, settings


def case() -> EvaluationCase:
    rows = [passage(name, name) for name in "ABC"]
    return EvaluationCase(
        "q1",
        "development",
        "family",
        "query",
        rows,
        {rows[0].id: 2, rows[1].id: 1},
        [[[rows[0].id]], [[rows[0].id, rows[1].id], [rows[2].id]]],
    )


def test_partial_labels_preserve_unknown_and_evidence_group_alternatives() -> None:
    value = case()
    report = metrics(value, [p.id for p in value.candidates])
    assert report["ndcg5"] is None
    assert report["unjudged_top5_count"] == 1
    assert report["designated_group_coverage5"] == 1
    assert report["all_designated_groups5"] is True
    assert report["known_positive_mrr5"] == 1
    assert report["judged_pool_ndcg5"] == 1


def test_known_positive_rank_and_recall_are_explicitly_designated() -> None:
    value = case()
    report = metrics(value, [value.candidates[1].id, value.candidates[0].id])
    assert report["known_positive_mrr5"] == 0.5
    assert report["designated_positive_recall5"] == 1
    assert report["ndcg5"] is None
    assert metrics(value, [value.candidates[1].id])["designated_group_coverage5"] == 0


def test_formal_ndcg_requires_complete_labels_and_known_returned_ids() -> None:
    value = replace(case(), judgments_complete=True)
    assert metrics(value, [p.id for p in value.candidates])["ndcg5"] is None
    assert metrics(value, [p.id for p in value.candidates[:2]])["ndcg5"] == 1


def test_split_validation_prevents_family_leakage() -> None:
    value = case()
    with pytest.raises(ValueError, match="family"):
        validate_cases([value, replace(value, question_id="q2", split="holdout")])


def test_unique_candidates_and_grades_are_validated() -> None:
    value = case()
    with pytest.raises(ValueError, match="unique"):
        validate_cases([replace(value, candidates=[value.candidates[0]] * 2)])
    with pytest.raises(ValueError, match="Grades"):
        validate_cases([replace(value, grades={"A": True})])
    with pytest.raises(ValueError, match="empty"):
        validate_cases([replace(value, evidence_groups=[[[]]])])


def test_summary_does_not_turn_missing_metrics_into_zero() -> None:
    result = summarize(
        [{"split": "development", "rrf": metrics(case(), []), "reranked": metrics(case(), [])}]
    )
    assert result["development/rrf/judged_pool_ndcg5"] is None
    assert result["holdout/rrf/designated_group_coverage5"] is None


async def test_reusing_output_directory_does_not_accumulate_model_load_events(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setitem(sys.modules, "torch", ModuleType("torch"))

    def version(name: str) -> str:
        _ = name
        return "fixture"

    def create(config: RerankingSettings, tracer: EventSink) -> CrossEncoderReranker:
        return CrossEncoderReranker(config, tracer, factory=lambda _: FakeBackend())

    monkeypatch.setattr(importlib.metadata, "version", version)
    monkeypatch.setattr(benchmark, "CrossEncoderReranker", create)
    value = replace(case(), candidates=[passage(str(i), str(i)) for i in range(3)])
    output = tmp_path / "output"
    await benchmark.evaluate([value], settings(), output, repeats=1)
    assert json.loads((output / "report.json").read_text(encoding="utf-8"))["model_loads"] == 1
    await benchmark.evaluate([value], settings(), output, repeats=1)
    assert json.loads((output / "report.json").read_text(encoding="utf-8"))["model_loads"] == 1
