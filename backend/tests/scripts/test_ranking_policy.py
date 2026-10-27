"""统一预算、缓存身份及未知标签的修复验收。"""

import hashlib
import json
import sys
from dataclasses import asdict, replace
from pathlib import Path

import pytest
from scripts.benchmark.ranking_policy import analyze, main
from scripts.benchmark.reranking import EvaluationCase, metrics_at_k

from flowpay.core.knowledge.hybrid_contracts import RankedCandidate, RerankHit
from flowpay.core.knowledge.ranking import rank_candidates
from tests.core.test_lexical import passage


def fixture() -> tuple[EvaluationCase, list[dict[str, object]]]:
    rows = [passage(str(i), str(i)) for i in range(8)]
    case = EvaluationCase(
        "q", "holdout", "family", "query", rows, {rows[5].id: 2}, [[[rows[5].id]]]
    )
    scores = [RerankHit(i, float(i)) for i in range(8)]
    pure = rank_candidates([RankedCandidate(p) for p in rows], scores, rerank_weight=1)
    report: dict[str, object] = {
        "question_id": "q",
        "rrf_ids": [p.id for p in rows],
        "scores": [asdict(score) for score in scores],
        "reranked_ids": [h.passage.id for h in pure],
    }
    return case, [report]


def test_all_strategies_use_same_six_passage_budget_and_token_accounting() -> None:
    case, scored = fixture()
    report = analyze([case], scored, lengths_by_query={"query": [10] * 8})
    assert report["k"] == 6
    assert report["rerank_weight"] == 0.75
    from flowpay.core.json import require_list, require_object

    row = require_object(require_list(report["rows"])[0])
    for strategy in ("rrf", "cross_encoder", "fixed"):
        data = require_object(row[strategy])
        assert data["context_tokens"] == 60
        assert "ndcg6" in require_object(data["metrics"])
        assert require_object(data["metrics"])["ndcg6"] is None


def test_score_cache_must_match_candidate_order_and_question_ids() -> None:
    case, scored = fixture()
    with pytest.raises(ValueError, match="order"):
        analyze([replace(case, candidates=list(reversed(case.candidates)))], scored)
    with pytest.raises(ValueError, match="questions"):
        analyze([replace(case, question_id="another")], scored)


def test_cached_model_order_must_be_reproducible() -> None:
    case, scored = fixture()
    scored[0]["reranked_ids"] = []
    with pytest.raises(ValueError, match="reproduced"):
        analyze([case], scored)


def test_cutoff_metrics_preserve_top_five_compatibility_and_report_top_six() -> None:
    case, _ = fixture()
    ids = [p.id for p in case.candidates]
    assert metrics_at_k(case, ids, 5)["designated_group_coverage5"] == 0
    assert metrics_at_k(case, ids, 6)["designated_group_coverage6"] == 1
    with pytest.raises(ValueError):
        metrics_at_k(case, ids, True)


def test_cli_binds_cache_hash_and_accepts_bom_judgments(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    case, rows = fixture()
    input_path, score_path = tmp_path / "input.json", tmp_path / "report.json"
    labels, output = tmp_path / "labels.json", tmp_path / "output.json"
    raw = json.dumps([asdict(case)]).encode()
    input_path.write_bytes(raw)
    score_path.write_text(json.dumps({"rows": rows}), encoding="utf-8")
    score_path.with_name("input.sha256").write_text(
        hashlib.sha256(raw).hexdigest(), encoding="utf-8"
    )
    labels.write_text(json.dumps({"q": {case.candidates[0].id: 2}}), encoding="utf-8-sig")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "ranking_policy",
            "--input",
            str(input_path),
            "--scores",
            str(score_path),
            "--judgments",
            str(labels),
            "--output",
            str(output),
        ],
    )
    main()
    from flowpay.core.json import parse_json, require_object

    report = require_object(parse_json(output.read_bytes()))
    assert report["k"] == 6 and report["rerank_weight"] == 0.75
    score_path.with_name("input.sha256").write_text("0" * 64, encoding="utf-8")
    with pytest.raises(ValueError, match="frozen"):
        main()
