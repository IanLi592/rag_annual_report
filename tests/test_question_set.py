"""评测集 v2 的守卫：把独立校验器变成每次 push 都会跑的门禁。

为什么必须有：questions_v2.json 是"尺子"。尺子本身坏了（引用对不上原文、
答案数字没有出处），后面所有指标都是噪声 —— 而且不会报错，只会安静地给出错的结论。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import evaluate
import pytest
import verify_question_set as V

REPO = Path(__file__).resolve().parents[1]
QV2 = REPO / "evaluation" / "questions_v2.json"


@pytest.fixture(scope="module")
def doc():
    assert QV2.exists(), "先跑 python evaluation/build_question_set.py"
    return json.loads(QV2.read_text(encoding="utf-8"))


def test_independent_verifier_passes(capsys):
    """直接跑校验器的 main()，判据是退出码，不是它打印了什么。"""
    assert V.main() == 0, capsys.readouterr().out[-1500:]


def test_scale_and_distribution_meet_resume_claim(doc):
    """简历写"100+ 道、四类"，这里就是那条主张的守卫。"""
    qs = doc["questions"]
    assert len(qs) >= 105, f"题数 {len(qs)} 不足 105"
    from collections import Counter
    dist = Counter(q["category"] for q in qs)
    for cat, lo in (("numeric", 30), ("multihop", 30), ("hallucination", 25), ("refusal", 20)):
        assert dist.get(cat, 0) >= lo, f"{cat} 只有 {dist.get(cat,0)}，要求 ≥{lo}"
    assert set(dist) == {"numeric", "multihop", "hallucination", "refusal"}


def test_ids_unique_and_types_compatible(doc):
    """type 字段必须仍是旧五类之一 —— evaluate.py 与 compare_strategies.py 都按它分组。"""
    qs = doc["questions"]
    ids = [q["id"] for q in qs]
    assert len(ids) == len(set(ids)) == len(qs)
    assert set(q["type"] for q in qs) <= V.TYPES


def test_refusal_questions_carry_no_evidence(doc):
    """拒答题必须无 evidence、无 target_docs —— 否则等于假装"这题有出处"。"""
    for q in doc["questions"]:
        if q["category"] != "refusal":
            continue
        assert q["evidence"] == [] and q["target_docs"] == [], f"id={q['id']} 拒答题不该带出处"


def test_derived_questions_all_have_evidence(doc):
    for q in doc["questions"]:
        if q["category"] == "refusal":
            continue
        assert q["evidence"], f"id={q['id']} 派生题没有 evidence"


def test_partial_company_coverage_is_declared_not_hidden(doc):
    """平安目前 0 条派生题（它的摘要表排版碎片化，候选没过一致性锁）。

    这种情况**必须写在 coverage 里**，不能让"125 题"看起来覆盖了 5 家公司。
    守卫的是诚实性，不是覆盖率。
    """
    import build_question_set as B
    cov = doc.get("coverage")
    assert cov, "缺 coverage 字段：题集覆盖面必须显式声明"
    covered = set(cov["companies_with_derived_questions"])
    without = set(cov["companies_without"])
    assert covered | without == set(B.COMPANIES), \
        f"覆盖面没声明全部公司：{covered | without} vs {set(B.COMPANIES)}"
    assert not (covered & without), "同一家公司不该既被覆盖又未被覆盖"
    if without:
        assert cov.get("why"), f"有公司未被派生题覆盖（{without}）却没说明原因"
    # 声明必须与实际题目一致 —— 防止 coverage 写成漂亮话
    actual = {td.split("_")[0] for q in doc["questions"] for td in q.get("target_docs", [])}
    assert actual == covered, f"coverage 声明 {covered} 与实际出题公司 {actual} 不符"


def test_default_question_set_is_v2_and_switchable(monkeypatch):
    monkeypatch.delenv("RAGAS_QUESTION_SET", raising=False)
    name, path = evaluate.resolve_question_set(None)
    assert name == "v2" and path.name == "questions_v2.json"
    n1, p1 = evaluate.resolve_question_set("v1")
    assert n1 == "v1" and p1.name == "questions.json"
    with pytest.raises(ValueError):
        evaluate.resolve_question_set("v3")


def test_v1_set_still_loadable_for_old_ruler(monkeypatch):
    """旧口径必须还能跑 —— 换题集等于换尺子，没有退路就没法归因。"""
    qs = evaluate.load_questions(which="v1")
    assert len(qs) == 20
    assert {q["type"] for q in qs} == {"simple_fact", "precise_number",
                                       "cross_doc_compare", "time_trend", "should_refuse"}
