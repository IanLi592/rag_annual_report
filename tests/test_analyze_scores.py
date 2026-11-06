"""逐题归因工具 analyze_scores.py 的离线守卫。

重点守两件事：均值自检（能抓到 zip 回填错位）和"没有逐题分数就别装作有结论"。
"""
from __future__ import annotations

import json
import sys

import analyze_scores as asx
import pytest


def doc(rows, scores=None):
    return {"pipeline": "native", "scores": scores or {}, "raw_results": rows}


def row(qid, qtype, cp, cr=1.0, n_ctx=4):
    return {"question_id": qid, "question_type": qtype, "question": f"问题{qid}",
            "contexts": ["c"] * n_ctx,
            "ragas": {"faithfulness": 1.0, "answer_relevancy": 1.0,
                      "context_precision": cp, "context_recall": cr}}


def write(tmp_path, d):
    p = tmp_path / "native_20260101_000000.json"
    p.write_text(json.dumps(d, ensure_ascii=False), encoding="utf-8")
    return str(p)


def run(monkeypatch, path, top=8):
    monkeypatch.setattr(sys, "argv", ["analyze_scores.py", "--file", path, "--top", str(top)])
    return asx.main()


def test_self_consistent_report_passes(tmp_path, monkeypatch, capsys):
    rows = [row(1, "simple_fact", 0.5), row(2, "simple_fact", 0.25), row(3, "time_trend", 1.0)]
    mean_cp = (0.5 + 0.25 + 1.0) / 3
    d = doc(rows, {"faithfulness": 1.0, "answer_relevancy": 1.0,
                   "context_precision": mean_cp, "context_recall": 1.0})
    assert run(monkeypatch, write(tmp_path, d)) == 0
    out = capsys.readouterr().out
    assert "simple_fact" in out and "time_trend" in out
    assert "自洽" in out


def test_mean_mismatch_is_flagged(tmp_path, monkeypatch, capsys):
    """顶层均值被手改成 0.9 而逐题只有 0.5 —— 必须判 1，不能放过。"""
    rows = [row(1, "simple_fact", 0.5)]
    d = doc(rows, {"faithfulness": 1.0, "answer_relevancy": 1.0,
                   "context_precision": 0.9, "context_recall": 1.0})
    assert run(monkeypatch, write(tmp_path, d)) == 1
    out = capsys.readouterr().out
    assert "对不上" in out and "zip 回填疑似错位" in out


def test_rows_without_scores_are_skipped_not_zeroed(tmp_path, monkeypatch, capsys):
    """should_refuse 类没有 ragas —— 不能被当成 0 分拉低均值。"""
    rows = [row(1, "simple_fact", 0.5),
            {"question_id": 2, "question_type": "should_refuse", "question": "q", "contexts": []}]
    d = doc(rows, {"context_precision": 0.5})
    assert run(monkeypatch, write(tmp_path, d)) == 0
    out = capsys.readouterr().out
    assert "无分数（should_refuse 等）1 题" in out
    # 不断言对齐行的空白（列宽是实现细节），只验数值没被 should_refuse 拉低
    line = next(l for l in out.splitlines() if "context_precision" in l and "重算" in l)
    assert "0.5000" in line and "一致" in line


def test_no_scored_rows_is_skip_not_fail(tmp_path, monkeypatch, capsys):
    """旧产物（§13 之前）没有逐题分数 → 是"没数据"不是"数据坏了"，判 2 且不崩。"""
    d = doc([{"question_id": 1, "question_type": "simple_fact", "question": "q", "contexts": []}],
            {"context_precision": 0.5})
    assert run(monkeypatch, write(tmp_path, d)) == 2
    assert "没有逐题 ragas 分数" in capsys.readouterr().out


def test_missing_file_is_skip(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["analyze_scores.py", "--file", str(tmp_path / "nope.json")])
    assert asx.main() == 2
    assert "[CI][SKIP]" in capsys.readouterr().out


def test_worst_questions_sorted_by_precision(tmp_path, monkeypatch, capsys):
    rows = [row(1, "simple_fact", 0.9), row(2, "simple_fact", 0.1), row(3, "simple_fact", 0.5)]
    d = doc(rows, {"context_precision": 0.5})
    run(monkeypatch, write(tmp_path, d), top=2)
    out = capsys.readouterr().out
    lines = [l for l in out.splitlines() if "cp=" in l]
    assert lines[0].strip().startswith("id=  2") and lines[1].strip().startswith("id=  3")


# ── 自动发现分支（之前的测试全走 --file，把这条分支漏掉了，
#    结果一个 NameError 直接崩在真实数据上。以下三条补这个盲区。）──

def _setup_dir(monkeypatch, tmp_path):
    monkeypatch.setattr(asx, "RESULTS", tmp_path)
    monkeypatch.setattr(sys, "argv", ["analyze_scores.py"])


def test_autodiscovery_picks_file_with_scores(tmp_path, monkeypatch, capsys):
    old = tmp_path / "native_20260101_000000.json"
    old.write_text(json.dumps(doc([row(1, "simple_fact", 0.5)], {"context_precision": 0.5})),
                   encoding="utf-8")
    _setup_dir(monkeypatch, tmp_path)
    assert asx.main() == 0
    assert "native_20260101_000000.json" in capsys.readouterr().out


def test_autodiscovery_ignores_legacy_files_without_scores(tmp_path, monkeypatch, capsys):
    """§13 之前的产物没有逐题分数 —— 不能被自动选中，否则归因表是空的还装作有结论。"""
    legacy = tmp_path / "native_20251201_000000.json"
    legacy.write_text(json.dumps({
        "pipeline": "native", "scores": {"context_precision": 0.5},
        "raw_results": [{"question_id": 1, "question_type": "simple_fact",
                         "question": "q", "contexts": []}]}), encoding="utf-8")
    newer = tmp_path / "native_20260102_000000.json"
    newer.write_text(json.dumps(doc([row(2, "simple_fact", 0.25)], {"context_precision": 0.25})),
                     encoding="utf-8")
    _setup_dir(monkeypatch, tmp_path)
    assert asx.main() == 0
    out = capsys.readouterr().out
    assert "native_20260102_000000.json" in out and "20251201" not in out


def test_autodiscovery_only_legacy_is_skip(tmp_path, monkeypatch, capsys):
    legacy = tmp_path / "native_20251201_000000.json"
    legacy.write_text(json.dumps({
        "pipeline": "native", "scores": {},
        "raw_results": [{"question_id": 1, "question_type": "simple_fact",
                         "question": "q", "contexts": []}]}), encoding="utf-8")
    _setup_dir(monkeypatch, tmp_path)
    assert asx.main() == 2
    assert "没有带逐题 ragas 分数的产物" in capsys.readouterr().out


def test_autodiscovery_survives_a_corrupt_file(tmp_path, monkeypatch, capsys):
    """半截 JSON（进程被杀时常见）不该让整个分析崩掉。"""
    (tmp_path / "native_20260101_000001.json").write_text('{"raw_result', encoding="utf-8")
    (tmp_path / "native_20260101_000000.json").write_text(
        json.dumps(doc([row(1, "simple_fact", 0.5)], {"context_precision": 0.5})), encoding="utf-8")
    _setup_dir(monkeypatch, tmp_path)
    assert asx.main() == 0
    assert "跳过读不了的产物" in capsys.readouterr().out
