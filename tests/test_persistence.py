"""逐题 ragas 分数落盘的回归守卫。

背景：修复前 `raw_results` 每条只有 question/answer/contexts/ground_truth，
顶层只有四个均值 —— 于是"Context Precision 0.273 是被哪几题拖低的"根本没法回答
（我上一轮想做按题型拆解，取到全 0，因为键压根不存在）。

这里只测落盘这一半（不联网）；打分→回填那一半靠真实跑验证。
"""
from __future__ import annotations

import csv
import json

import evaluate


ROW_WITH = {
    "question_id": 3, "question_type": "cross_doc_compare", "question": "比较茅台和五粮液",
    "answer": "A", "contexts": ["c1", "c2"], "ground_truth": "G",
    "ragas": {"faithfulness": 0.5, "answer_relevancy": 0.9,
              "context_precision": 0.0, "context_recall": None},
}
ROW_WITHOUT = {          # should_refuse 类：被排除，不该有分数
    "question_id": 19, "question_type": "should_refuse", "question": "今天适合买入吗",
    "answer": "无法回答", "contexts": [], "ground_truth": "不该回答",
}


def _run(tmp_path, monkeypatch):
    monkeypatch.setattr(evaluate, "RESULT_DIR", tmp_path)
    evaluate.save_results([dict(ROW_WITH), dict(ROW_WITHOUT)], {"faithfulness": 0.5}, "native")
    js = next(tmp_path.glob("native_*.json"))
    cs = next(tmp_path.glob("native_*.csv"))
    return json.loads(js.read_text(encoding="utf-8")), list(
        csv.DictReader(cs.read_text(encoding="utf-8-sig").splitlines()))


def test_per_question_scores_land_in_json(tmp_path, monkeypatch):
    doc, _ = _run(tmp_path, monkeypatch)
    r0 = doc["raw_results"][0]["ragas"]
    assert r0["faithfulness"] == 0.5 and r0["context_precision"] == 0.0
    assert r0["context_recall"] is None, "None 要原样留着，不能被塞成 0 —— 那会污染均值口径"
    assert "ragas" not in doc["raw_results"][1], "没打分的题不该被凭空写零"


def test_csv_has_one_column_per_metric(tmp_path, monkeypatch):
    _, rows = _run(tmp_path, monkeypatch)
    for col in ("faithfulness", "answer_relevancy", "context_precision", "context_recall"):
        assert col in rows[0], f"CSV 少了 {col} 列"
    assert float(rows[0]["context_precision"]) == 0.0
    assert rows[1]["faithfulness"] in ("", "nan"), "未打分题在 CSV 里应是空值"


def test_means_and_per_question_coexist(tmp_path, monkeypatch):
    """顶层均值不能因为加了逐题分数而丢 —— compare_strategies 与历史结果都靠它。"""
    doc, _ = _run(tmp_path, monkeypatch)
    assert doc["scores"] == {"faithfulness": 0.5}
    assert doc["pipeline"] == "native"
