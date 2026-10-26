"""RRF 只融合排名，保留各路原分数和原文身份。"""

from dataclasses import replace

import pytest

from flowpay.core.knowledge.fusion import fuse_rrf
from flowpay.core.knowledge.hybrid_contracts import LexicalHit

from .test_lexical import passage


def test_hand_calculated_two_lane_ranking() -> None:
    a, b, c = [passage(name, name) for name in "ABC"]
    result = fuse_rrf([a, b], [LexicalHit(b, 12.0), LexicalHit(c, 9.0)])
    assert [row.passage.source.doc_id for row in result] == ["B", "A", "C"]
    assert result[0].rrf_score == pytest.approx(1 / 62 + 1 / 61)
    assert (result[0].dense_rank, result[0].bm25_rank) == (2, 1)
    assert (result[0].dense_score, result[0].bm25_score) == (0.0, 12.0)
    assert result[1].rrf_score == pytest.approx(1 / 61)
    assert result[1].bm25_rank is None and result[1].bm25_score is None
    assert result[2].rrf_score == pytest.approx(1 / 62)
    assert result[2].dense_rank is None and result[2].dense_score is None


def test_empty_and_single_lane_results_keep_input_ranks() -> None:
    a, b = passage("A", "a"), passage("B", "b")
    assert fuse_rrf([], []) == []
    assert [r.passage.id for r in fuse_rrf([b, a], [])] == [b.id, a.id]
    result = fuse_rrf([], [LexicalHit(b, -1.0), LexicalHit(a, -2.0)])
    assert [r.passage.id for r in result] == [b.id, a.id]
    assert result[0].bm25_score == -1.0


def test_duplicates_do_not_add_votes_or_consume_distinct_rank_positions() -> None:
    a, b = passage("A", "a"), passage("B", "b")
    result = fuse_rrf([a, a, b], [LexicalHit(b, 5.0), LexicalHit(b, 5.0)])
    assert len(result) == 2
    assert result[0].passage.id == b.id
    assert result[0].dense_rank == 2
    assert result[0].rrf_score == pytest.approx(1 / 62 + 1 / 61)
    assert result[1].rrf_score == pytest.approx(1 / 61)


def test_equal_scores_have_stable_document_and_chunk_order() -> None:
    a, b = passage("A", "a"), passage("B", "b")
    first = fuse_rrf([b], [LexicalHit(a, 99.0)])
    second = fuse_rrf([a], [LexicalHit(b, -99.0)])
    assert [r.passage.id for r in first] == [a.id, b.id]
    assert [r.passage.id for r in second] == [a.id, b.id]
    a2 = replace(a, id="doc:A#c002", source=replace(a.source, chunk_no=2))
    assert [r.passage.id for r in fuse_rrf([a2], [LexicalHit(a, 0.0)])] == [a.id, a2.id]


def test_raw_scores_do_not_change_fusion_and_passages_are_not_mutated() -> None:
    dense = replace(passage("A", "a"), score=0.8)
    lexical = replace(dense, score=0.0)
    result = fuse_rrf([dense], [LexicalHit(lexical, -12.0)])
    assert result[0].passage is dense
    assert result[0].rrf_score == pytest.approx(2 / 61)
    assert result[0].dense_score == 0.8
    assert result[0].bm25_score == -12.0
    assert dense.score == 0.8 and lexical.score == 0.0


@pytest.mark.parametrize("constant", [True, 0, -1, 1.5])
def test_rank_constant_must_be_a_positive_integer(constant: int) -> None:
    with pytest.raises(ValueError, match="constant"):
        fuse_rrf([], [], rank_constant=constant)


def test_custom_rank_constant() -> None:
    a = passage("A", "a")
    assert fuse_rrf([a], [], rank_constant=1)[0].rrf_score == pytest.approx(0.5)


def test_mixing_collections_is_rejected_even_when_documents_have_same_id() -> None:
    a, other = passage("A", "a", "one"), passage("A", "a", "two")
    with pytest.raises(ValueError, match="collection"):
        fuse_rrf([a], [LexicalHit(other, 1.0)])
    with pytest.raises(ValueError, match="collection"):
        fuse_rrf([a, other], [])


@pytest.mark.parametrize("change", ["text", "title", "id", "source"])
def test_same_identity_cannot_point_to_conflicting_evidence(change: str) -> None:
    a = passage("A", "a")
    changes = {
        "text": replace(a, text="different"),
        "title": replace(a, title="different"),
        "id": replace(a, id="doc:other#c001"),
        "source": replace(a, source=replace(a.source, sha256="b" * 64)),
    }
    with pytest.raises(ValueError, match="Conflicting"):
        fuse_rrf([a], [LexicalHit(changes[change], 1.0)])


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_nonfinite_source_scores_are_rejected(value: float) -> None:
    a = passage("A", "a")
    with pytest.raises(ValueError, match="finite"):
        fuse_rrf([replace(a, score=value)], [])
    with pytest.raises(ValueError, match="finite"):
        fuse_rrf([], [LexicalHit(a, value)])


def test_top_n_is_a_prefix_of_the_full_fused_ranking() -> None:
    dense = [passage(str(i), str(i)) for i in range(50)]
    lexical = [LexicalHit(p, float(i)) for i, p in enumerate(reversed(dense))]
    result = fuse_rrf(dense, lexical)
    selected = result[:30]
    assert len(result) == 50 and len(selected) == 30
    assert {r.passage.id for r in selected} <= {p.id for p in dense}
    assert all(selected[i].rrf_score >= selected[i + 1].rrf_score for i in range(29))
