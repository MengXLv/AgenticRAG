"""区分未召回、截断丢失与未标注，避免诊断报告虚报覆盖率。"""

from dataclasses import asdict

import pytest
from pydantic import TypeAdapter
from scripts.benchmark.fusion import FusionCase, inspect

from flowpay.core.knowledge.hybrid_contracts import LexicalHit
from tests.core.test_lexical import passage


def test_cutoff_distinguishes_retrieval_miss_and_fusion_loss() -> None:
    a, b, c = [passage(name, name) for name in "ABC"]
    case = FusionCase(
        "fixture",
        "test",
        [a, b],
        [LexicalHit(b, 3.0), LexicalHit(c, 1.0)],
        [a.id, c.id, "doc:missing#c001"],
    )
    report = inspect(case, top_n=1)
    assert report["union_count"] == 3 and report["selected_count"] == 1
    assert report["expected_not_recalled"] == ["doc:missing#c001"]
    assert report["expected_lost_at_cutoff"] == [a.id, c.id]
    assert report["dropped_ids"] == [a.id, c.id]


def test_unlabeled_and_explicitly_empty_labels_have_different_meanings() -> None:
    assert inspect(FusionCase("unlabeled", "test", [], []))["expected_lost_at_cutoff"] is None
    assert inspect(FusionCase("empty", "test", [], [], []))["expected_lost_at_cutoff"] == []


@pytest.mark.parametrize("top_n", [True, 0, -1, 1.5])
def test_cutoff_must_be_a_positive_integer(top_n: int) -> None:
    with pytest.raises(ValueError, match="top_n"):
        inspect(FusionCase("fixture", "test", [], []), top_n)


def test_case_json_roundtrip_keeps_source_and_separate_scores() -> None:
    a = passage("A", "a")
    case = FusionCase("fixture", "test", [a], [LexicalHit(a, -2.0)])
    restored = TypeAdapter(FusionCase).validate_python(asdict(case))
    assert inspect(restored) == inspect(case)
