from dataclasses import replace

import pytest
from scripts.benchmark.reranking import EvaluationCase
from scripts.benchmark.retrieval import RetrievalCase, analyze, attribute_groups

from flowpay.core.json import require_list, require_number, require_object
from flowpay.core.knowledge.fusion import fuse_rrf
from flowpay.core.knowledge.hybrid_contracts import LexicalHit, RerankHit
from tests.core.test_lexical import passage


def sample() -> RetrievalCase:
    rows = [passage(str(i), str(i)) for i in range(12)]
    lexical = [LexicalHit(p, float(12 - i)) for i, p in enumerate(rows)]
    candidates = [c.passage for c in fuse_rrf(rows, lexical)[:10]]
    evaluation = EvaluationCase(
        "q",
        "development",
        "family",
        "query",
        candidates,
        {rows[0].id: 2, rows[11].id: 2},
        [[[rows[0].id]], [[rows[11].id]]],
    )
    return RetrievalCase(evaluation, rows, lexical, [RerankHit(i, float(i)) for i in range(10)])


def test_comparison_uses_same_budget_and_keeps_unknown_labels_unknown() -> None:
    case = sample()
    report = analyze([case], rerank_limit=10, limit=6, bootstrap_samples=100)
    row = require_object(require_list(report["rows"])[0])
    for name in ("dense", "bm25", "rrf", "hybrid"):
        result = require_object(row[name])
        assert len(require_list(result["ids"])) == 6
        metrics = require_object(result["metrics"])
        assert metrics["ndcg6"] is None
        assert "judged_pool_ndcg6" not in metrics
        assert require_number(metrics["unjudged_top6_count"]) > 0
    assert report["question_count"] == 1 and report["scenario_family_count"] == 1


def test_default_cutoff_is_seven_and_explicit_six_keeps_same_prefix() -> None:
    case = sample()
    seven = analyze([case], rerank_limit=10, bootstrap_samples=100)
    six = analyze([case], rerank_limit=10, limit=6, bootstrap_samples=100)
    assert require_object(seven["policy"])["k"] == 7
    for name in ("dense", "bm25", "rrf", "hybrid"):
        seven_ids = require_list(
            require_object(require_object(require_list(seven["rows"])[0])[name])["ids"]
        )
        six_ids = require_list(
            require_object(require_object(require_list(six["rows"])[0])[name])["ids"]
        )
        assert len(seven_ids) == 7 and seven_ids[:6] == six_ids


def test_evidence_groups_distinguish_recall_cutoff_and_selection_losses() -> None:
    groups = [[["absent"]], [["late"]], [["candidate"]], [["selected"]], [["absent"], ["selected"]]]
    assert attribute_groups(
        groups, {"late", "candidate", "selected"}, {"candidate", "selected"}, {"selected"}
    ) == [
        "recall_miss",
        "fusion_cutoff",
        "final_selection",
        "covered",
        "covered",
    ]


def test_frozen_candidates_must_match_replayed_rrf_order() -> None:
    case = sample()
    wrong = replace(
        case,
        evaluation=replace(case.evaluation, candidates=list(reversed(case.evaluation.candidates))),
    )
    with pytest.raises(ValueError, match="frozen"):
        analyze([wrong], rerank_limit=10)


def test_rerank_budget_cannot_exceed_frozen_evaluation_contract() -> None:
    with pytest.raises(ValueError, match="30"):
        analyze([sample()], rerank_limit=31)


def test_conflicting_sources_and_duplicate_questions_are_rejected() -> None:
    case = sample()
    with pytest.raises(ValueError, match="Duplicate"):
        analyze([case, case], rerank_limit=10)
    wrong = replace(case.dense[0], text="conflicting original text")
    with pytest.raises(ValueError):
        analyze([replace(case, dense=[wrong, *case.dense[1:]])], rerank_limit=10)


def test_bootstrap_counts_families_not_repeated_measurements() -> None:
    first = sample()
    second = replace(first, evaluation=replace(first.evaluation, question_id="q2"))
    report = analyze([first, second], rerank_limit=10, bootstrap_samples=100)
    assert report["question_count"] == 2 and report["scenario_family_count"] == 1
    assert report["bootstrap_unit"] == "scenario_family_id"


def test_partial_evidence_alternative_is_not_counted_as_complete() -> None:
    assert attribute_groups([[["A", "B"]]], {"A", "B"}, {"A", "B"}, {"A"}) == ["final_selection"]


def test_zero_recall_stays_in_denominator_instead_of_being_dropped() -> None:
    case = sample()
    empty = replace(
        case, evaluation=replace(case.evaluation, candidates=[]), dense=[], lexical=[], ranks=[]
    )
    report = analyze([empty], rerank_limit=10)
    assert report["question_count"] == 1
    assert report["hybrid_group_loss_stages"] == {"recall_miss": 2}
    hybrid = require_object(require_object(report["summary"])["hybrid"])
    assert hybrid["designated_group_coverage7"] == 0.0


def test_measured_rounds_are_separate_from_quality_sample_size() -> None:
    case = replace(
        sample(),
        warm_timings_ms={name: [1.0, 2.0, 3.0] for name in ("dense", "bm25", "rrf", "hybrid")},
        measured_rerank_weight=1.0,
        measured_limit=7,
    )
    report = analyze([case], rerank_limit=10)
    assert report["question_count"] == 1
    performance = require_object(report["performance"])
    assert require_object(performance["hybrid"])["measurement_count"] == 3
    assert require_object(performance["hybrid"])["warm_p95_ms"] == 3.0
    with pytest.raises(ValueError, match="policy"):
        analyze([replace(case, measured_limit=6)], rerank_limit=10)
    with pytest.raises(ValueError, match="policy"):
        analyze([replace(case, measured_rerank_weight=0.75)], rerank_limit=10)
    with pytest.raises(ValueError, match="policy"):
        analyze([replace(case, measured_rerank_weight=None)], rerank_limit=10)
    with pytest.raises(ValueError, match="all four"):
        analyze([replace(case, warm_timings_ms={"dense": [1.0, 2.0, 3.0]})], rerank_limit=10)
    with pytest.raises(ValueError, match="three"):
        analyze(
            [
                replace(
                    case,
                    warm_timings_ms={
                        name: [1.0, 2.0] for name in ("dense", "bm25", "rrf", "hybrid")
                    },
                )
            ],
            rerank_limit=10,
        )
