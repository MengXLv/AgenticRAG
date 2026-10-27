"""最终排序同时保留 RRF 先验与模型分数，不扩大 Top-K 预算。"""

import math
from dataclasses import replace
from pathlib import Path

import pytest

from flowpay.core.json import parse_json, require_list, require_number, require_object, require_text
from flowpay.core.knowledge.hybrid_contracts import RankedCandidate, RerankHit
from flowpay.core.knowledge.ranking import rank_candidates

from .test_lexical import passage


def candidates() -> list[RankedCandidate]:
    return [RankedCandidate(passage(name, name)) for name in "ABC"]


def test_normalized_score_blend_preserves_evidence_and_separate_scores() -> None:
    source = candidates()
    hits = rank_candidates(
        source, [RerankHit(0, 0), RerankHit(1, 1), RerankHit(2, 0.9)], rerank_weight=0.5
    )
    assert [h.passage.source.doc_id for h in hits] == ["B", "A", "C"]
    assert hits[1].passage is source[0].passage
    assert hits[1].ranking.rerank_score == 0
    assert hits[1].ranking.rrf_rank == 1
    assert hits[1].ranking.rerank_rank == 3
    assert hits[1].ranking.final_score == pytest.approx(0.5)
    assert all(h.passage.score == 0 for h in hits)


def test_equal_model_scores_preserve_rrf_order() -> None:
    hits = rank_candidates(candidates(), [RerankHit(i, 1) for i in range(3)])
    assert [h.passage.source.doc_id for h in hits] == ["A", "B", "C"]


def test_extreme_weights_reproduce_single_strategy() -> None:
    rows = [RerankHit(2, 0), RerankHit(0, 1), RerankHit(1, 2)]
    assert [
        h.passage.source.doc_id for h in rank_candidates(candidates(), rows, rerank_weight=0)
    ] == ["A", "B", "C"]
    assert [
        h.passage.source.doc_id for h in rank_candidates(candidates(), rows, rerank_weight=1)
    ] == ["B", "A", "C"]


def test_extreme_finite_scores_do_not_overflow_normalization() -> None:
    hits = rank_candidates(
        candidates(), [RerankHit(0, -1e308), RerankHit(1, 1e308), RerankHit(2, 0)]
    )
    assert all(
        h.ranking.final_score is not None and math.isfinite(h.ranking.final_score) for h in hits
    )


def test_pure_model_endpoint_keeps_distinct_raw_scores_even_if_scaling_loses_precision() -> None:
    scores = [RerankHit(0, 0.0), RerankHit(1, 1e-300), RerankHit(2, 1e308)]
    assert [
        h.passage.source.doc_id for h in rank_candidates(candidates(), scores, rerank_weight=1)
    ] == ["C", "B", "A"]


@pytest.mark.parametrize("weight", [True, -0.1, 1.1, float("nan"), float("inf")])
def test_invalid_weight_is_rejected(weight: float) -> None:
    with pytest.raises(ValueError, match="weight"):
        rank_candidates([], [], rerank_weight=weight)


def test_empty_pool_and_invalid_mapping() -> None:
    assert rank_candidates([], []) == []
    with pytest.raises(ValueError, match="every candidate"):
        rank_candidates(candidates(), [])
    with pytest.raises(ValueError, match="duplicated"):
        rank_candidates(candidates(), [RerankHit(0, 1)] * 3)


def test_recorded_model_regressions_keep_both_critical_passages_in_default_budget() -> None:
    raw = Path(__file__).with_name("fixtures") / "ranking_regressions.json"
    for value in require_list(parse_json(raw.read_text(encoding="utf-8"))):
        fixture = require_object(value)
        source: list[RankedCandidate] = []
        for value in require_list(fixture["candidate_ids"]):
            pid = require_text(value)
            doc_id, chunk = pid.removeprefix("doc:").split("#c")
            p = passage(doc_id, doc_id)
            source.append(
                RankedCandidate(replace(p, id=pid, source=replace(p.source, chunk_no=int(chunk))))
            )
        scores = [
            RerankHit(i, require_number(score))
            for i, score in enumerate(require_list(fixture["scores"]))
        ]
        assert fixture["required_id"] not in [
            h.passage.id for h in rank_candidates(source, scores, rerank_weight=1)[:5]
        ]
        assert fixture["required_id"] in [h.passage.id for h in rank_candidates(source, scores)[:6]]
