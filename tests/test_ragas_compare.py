"""RAGAS 对比门的纯函数守卫：delta、回退判定、分层选题。"""
from __future__ import annotations

import pytest

import compare_ragas as cr
import pick_eval_ids as pick


def doc(scores=None, rows=None):
    d = {"scores": scores or {}, "raw_results": rows or []}
    return d


def row(qid, **vals):
    return {"question_id": qid, "ragas": vals}


def test_delta_and_regression_threshold():
    base = doc({"faithfulness": 0.80, "answer_relevancy": 0.70,
                "context_precision": 0.60, "context_recall": 0.50})
    latest = doc({"faithfulness": 0.79, "answer_relevancy": 0.71,
                  "context_precision": 0.55, "context_recall": 0.50})
    r = cr.compare_docs(base, latest, tolerance=0.03)
    assert r["metrics"]["faithfulness"]["regressed"] is False       # -0.01 < 0.03
    assert r["metrics"]["context_precision"]["regressed"] is True  # -0.05 > 0.03
    assert r["metrics"]["context_precision"]["delta"] == pytest.approx(-0.05)


def test_top_level_scores_are_preferred_over_row_means():
    base = doc({"faithfulness": 0.9}, [row(1, faithfulness=0.0)])
    latest = doc({"faithfulness": 0.8}, [row(1, faithfulness=0.0)])
    r = cr.compare_docs(base, latest)
    assert r["metrics"]["faithfulness"]["baseline"] == 0.9
    assert r["metrics"]["faithfulness"]["latest"] == 0.8


def test_missing_top_level_score_falls_back_to_row_mean():
    base = doc({}, [row(1, context_recall=0.4), row(2, context_recall=0.6)])
    latest = doc({}, [row(1, context_recall=0.5), row(2, context_recall=0.5)])
    r = cr.compare_docs(base, latest)
    assert r["metrics"]["context_recall"]["baseline"] == 0.5
    assert r["metrics"]["context_recall"]["delta"] == 0.0


def test_per_question_regressions_use_common_ids_only():
    base = doc({}, [row(1, faithfulness=0.9), row(2, faithfulness=0.9)])
    latest = doc({}, [row(1, faithfulness=0.5), row(3, faithfulness=0.0)])
    r = cr.compare_docs(base, latest, tolerance=0.03)
    assert len(r["regressions"]) == 1
    assert r["regressions"][0]["question_id"] == "1"
    assert r["questions"]["common"] == 1
    assert r["questions"]["only_latest"] == ["3"]


def test_no_regression_when_metrics_improve():
    base = doc({"faithfulness": 0.5})
    latest = doc({"faithfulness": 0.7})
    r = cr.compare_docs(base, latest)
    assert not any(m["regressed"] for m in r["metrics"].values())


def test_pick_ids_is_stratified_and_skips_refusal():
    qs = [
        {"id": 1, "category": "numeric", "type": "precise_number"},
        {"id": 2, "category": "numeric", "type": "precise_number"},
        {"id": 3, "category": "numeric", "type": "precise_number"},
        {"id": 4, "category": "multihop", "type": "cross_doc_compare"},
        {"id": 5, "category": "hallucination", "type": "time_trend"},
        {"id": 6, "category": "refusal", "type": "should_refuse"},
    ]
    ids = pick.pick(qs, per_category=1)
    assert ids == [1, 4, 5], "每类各取前 1，且拒答题必须排除"
