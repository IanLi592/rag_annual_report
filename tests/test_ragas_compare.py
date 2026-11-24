"""RAGAS 对比门的纯函数守卫：delta、回退判定、分层选题，以及 ⚠ 在 GBK 控制台崩门。"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

import compare_ragas as cr
import pick_eval_ids as pick

REPO = Path(__file__).resolve().parents[1]


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


# ── ⚠ 与控制台编码：门必须真的能把结果打出来 ──────────────────────────────────
# 实测踩坑（2026-11-16）：Windows 控制台默认 GBK，报告里的 `⚠`（回退标记）让
# `print(render(report))` 抛 UnicodeEncodeError —— 评估早跑完了，门却永远打不出
# 对比、退出码还被 traceback 顶成 1。也就是「越该看的那次越看不到」。
# 用子进程 + PYTHONIOENCODING=gbk 造出同样的失败条件，不依赖本机 locale。

def _write_pair(tmp_path, latest_faithfulness=0.60):
    base = tmp_path / "base.json"
    latest = tmp_path / "latest.json"
    base.write_text(json.dumps(doc({"faithfulness": 0.80, "answer_relevancy": 0.70,
                                          "context_precision": 0.60, "context_recall": 0.50},
                                         [row(1, faithfulness=0.80)]), ensure_ascii=False),
                    encoding="utf-8")
    latest.write_text(json.dumps(doc({"faithfulness": latest_faithfulness,
                                             "answer_relevancy": 0.70,
                                             "context_precision": 0.60,
                                             "context_recall": 0.50},
                                         [row(1, faithfulness=latest_faithfulness)]),
                                 ensure_ascii=False), encoding="utf-8")
    return base, latest


def _run_cli(tmp_path, extra=(), latest_faithfulness=0.60):
    base, latest = _write_pair(tmp_path, latest_faithfulness)
    return subprocess.run(
        [sys.executable, str(REPO / "scripts" / "ci" / "compare_ragas.py"),
         "--baseline", str(base), "--latest", str(latest), *extra],
        capture_output=True, text=True, encoding="utf-8", errors="replace",
        env={**os.environ, "PYTHONIOENCODING": "gbk", "PYTHONUTF8": "0"},
    )


def test_warning_marker_only_appears_on_regression():
    """先钉住 ⚠ 的语义：它正是「出现回退」才有的那个非 ASCII 字符。"""
    regressed = cr.render(cr.compare_docs(doc({"faithfulness": 0.8}), doc({"faithfulness": 0.6})))
    clean = cr.render(cr.compare_docs(doc({"faithfulness": 0.8}), doc({"faithfulness": 0.79})))
    assert "⚠" in regressed and "⚠" not in clean


@pytest.mark.parametrize("extra,expect_rc", [
    ((), 0),                # 默认只报告，不拦人
    (("--strict",), 1),     # --strict 才因回退退出 1
], ids=["只报告", "strict拦人"])
def test_report_prints_on_gbk_console(tmp_path, extra, expect_rc):
    proc = _run_cli(tmp_path, extra)
    assert "UnicodeEncodeError" not in (proc.stderr or ""), proc.stderr[-400:]
    assert "⚠" in proc.stdout, f"报告没打出来：{proc.stdout[-400:]!r}"
    assert "Faithfulness" in proc.stdout and "-0.2000" in proc.stdout
    assert proc.returncode == expect_rc


def test_stdout_encoding_fix_does_not_alter_verdict(tmp_path):
    """修编码不许顺手改判读：无回退时 strict 也必须退出 0。"""
    proc = _run_cli(tmp_path, ("--strict",), latest_faithfulness=0.79)
    assert "⚠" not in proc.stdout
    assert proc.returncode == 0, proc.stdout + proc.stderr
