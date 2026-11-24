"""生成阶段完整性守卫（不联网、零 API 调用、不花钱）。

背景（2026-11-16 实测）：DashScope 账户欠费后，125 题里 96 题的生成全部抛异常，
旧实现把 `'ERROR: …'` 当答案继续送 RAGAS 打分 —— 白烧 1,464 次裁判调用、
产物照常落盘、退出码 0，调用方看不出这批是废的；全批失败时 ragas 内部还抛
IndexError 崩在最后。

守四条：
  1. 生成失败必须是结构化事实（gen_error），不能只藏在答案字符串里
  2. 失败的题不进打分 —— 打分才是花钱的那一步
  3. 失败率超阈值必须中止、退出码非 0、不落盘
  4. 产物里要留下「哪些题废了」，别让废产物和正常产物长得一样
"""
from __future__ import annotations

import json

import pytest

import evaluate
import rag_pipeline


def ok_row(qid):
    return {"question_id": qid, "question_type": "precise_number", "question": f"Q{qid}",
            "answer": f"第{qid}题的正常答案", "contexts": ["c1", "c2"],
            "ground_truth": "G", "gen_error": None}


def bad_row(qid, msg="Error code: 400 - Arrearage"):
    return {"question_id": qid, "question_type": "precise_number", "question": f"Q{qid}",
            "answer": f"ERROR: {msg}", "contexts": [], "ground_truth": "G",
            "gen_error": f"RuntimeError: {msg}"}


class FakePipeline:
    """只替 RAGPipeline 的 query：题目文本含 BOOM 就抛，复刻 API 报错的形状。"""

    def __init__(self, *a, **k):
        pass

    def query(self, question, verbose=False):
        if "BOOM" in question:
            raise RuntimeError("Error code: 400 - {'type': 'Arrearage'}")
        return {"answer": f"答案：{question}", "retrieved": [{"content": "c1"}]}


# ── 1. 失败要被结构化标记 ─────────────────────────────────────────────────────

def test_generation_failure_is_marked_structurally(monkeypatch):
    monkeypatch.setattr(rag_pipeline, "RAGPipeline", FakePipeline)
    rows = evaluate.run_native_rag([
        {"id": 1, "question": "五粮液净利润", "ground_truth": "G", "type": "precise_number"},
        {"id": 2, "question": "BOOM 会失败的题", "ground_truth": "G", "type": "precise_number"},
    ])
    assert rows[0]["gen_error"] is None, "成功题不该带失败标记"
    assert rows[1]["gen_error"].startswith("RuntimeError:"), rows[1]["gen_error"]
    assert rows[1]["contexts"] == []
    assert evaluate.generation_failures(rows) == [2]


# ── 2/3. 排除失败题 + 超阈值中止 ──────────────────────────────────────────────

def test_default_tolerance_is_pinned():
    """0.1 是写在 help 里的承诺，不能被悄悄改掉。"""
    assert evaluate.DEFAULT_MAX_GEN_FAILURE_RATE == pytest.approx(0.1)


def test_failed_rows_are_excluded_when_under_tolerance():
    rows = [ok_row(i) for i in range(1, 10)] + [bad_row(10)]
    scorable = evaluate.enforce_generation_integrity(rows)          # 1/10 = 10% ≤ 10%
    assert [r["question_id"] for r in scorable] == list(range(1, 10))
    assert all(not r.get("gen_error") for r in scorable)


def test_mass_failure_aborts_before_any_scoring_money():
    rows = [ok_row(i) for i in range(1, 23)] + [bad_row(i) for i in range(23, 26)]
    with pytest.raises(evaluate.GenerationIntegrityError) as e:
        evaluate.enforce_generation_integrity(rows)                  # 3/25 = 12% > 10%
    msg = str(e.value)
    assert "不进 RAGAS 打分" in msg and "23, 24, 25" in msg, msg


def test_zero_tolerance_is_the_canary_setting():
    rows = [ok_row(1), bad_row(2)]
    with pytest.raises(evaluate.GenerationIntegrityError):
        evaluate.enforce_generation_integrity(rows, max_failure_rate=0.0)


def test_escape_hatch_accepts_total_failure():
    rows = [bad_row(1), bad_row(2)]
    assert evaluate.enforce_generation_integrity(rows, max_failure_rate=1.0) == []


def test_empty_results_do_not_reach_ragas():
    """全批失败时旧代码会走到 ragas 里崩成 IndexError（20:30:25 实测）。"""
    with pytest.raises(evaluate.GenerationIntegrityError):
        evaluate.enforce_generation_integrity([])


# ── main() 的接线：中止必须发生在打分之前 ─────────────────────────────────────

@pytest.fixture
def stubbed(monkeypatch, request):
    """把生成/打分/落盘三个口子换成记录器；返回 (calls, set_fn)。"""
    calls = {"score": [], "save": []}
    rows = request.param

    monkeypatch.setattr(evaluate, "run_native_rag", lambda qs: list(rows))
    monkeypatch.setattr(evaluate, "resolve_judge",
                        lambda: {"name": "mimo", "model": "m", "key_env": "K", "base_url": "u"})
    monkeypatch.setattr(evaluate, "run_ragas_eval",
                        lambda r, name: calls["score"].append([x["question_id"] for x in r])
                        or {"faithfulness": 0.5})
    monkeypatch.setattr(evaluate, "save_results",
                        lambda raw, scores, name, failed=None: calls["save"].append(
                            ([x["question_id"] for x in raw], list(failed or []))))
    monkeypatch.setattr("sys.argv", ["evaluate.py", "--question-ids", "1,2"])
    return calls


@pytest.mark.parametrize("stubbed", [[bad_row(i) for i in range(1, 26)]], indirect=True)
def test_main_exits_nonzero_and_pays_nothing(stubbed):
    with pytest.raises(SystemExit) as e:
        evaluate.main()
    assert e.value.code == 2
    assert stubbed["score"] == [], "全批失败却仍然进了 RAGAS 打分 = 白烧裁判调用"
    assert stubbed["save"] == [], "废产物不许落盘"


def test_canary_skip_ragas_path_also_blocks(monkeypatch):
    """canary 用法：跑 1 题 + --skip-ragas + 零容忍，失败必须停在退出码 2。

    这是补跑 26–125 前那道「先探一题再放行整批」的闸门，所以要钉住：
    --skip-ragas 只跳过打分，不跳过完整性守卫。
    """
    calls = []
    monkeypatch.setattr(evaluate, "run_native_rag", lambda qs: [bad_row(1)])
    monkeypatch.setattr(evaluate, "resolve_judge",
                        lambda: {"name": "mimo", "model": "m", "key_env": "K", "base_url": "u"})
    monkeypatch.setattr(evaluate, "run_ragas_eval", lambda *a, **k: calls.append("score"))
    monkeypatch.setattr(evaluate, "save_results", lambda *a, **k: calls.append("save"))
    monkeypatch.setattr("sys.argv", ["evaluate.py", "--question-ids", "1",
                                     "--skip-ragas", "--max-gen-failure-rate", "0"])
    with pytest.raises(SystemExit) as e:
        evaluate.main()
    assert e.value.code == 2
    assert calls == []


@pytest.mark.parametrize("stubbed",
                         [[ok_row(i) for i in range(1, 10)] + [bad_row(10)]], indirect=True)
def test_main_scores_only_valid_rows_but_saves_all(stubbed):
    evaluate.main()
    assert stubbed["score"] == [list(range(1, 10))], "失败题必须排除在打分之外"
    assert len(stubbed["save"]) == 1
    saved_ids, failed_ids = stubbed["save"][0]
    assert saved_ids == list(range(1, 11)), "留档要留全，废在哪题要看得见"
    assert failed_ids == [10]


# ── 4. 产物里要记下哪些题废了 ─────────────────────────────────────────────────

def test_artifact_records_generation_failures(tmp_path, monkeypatch):
    monkeypatch.setattr(evaluate, "RESULT_DIR", tmp_path)
    evaluate.save_results([ok_row(6), bad_row(7)], {"faithfulness": 0.5}, "native", [7])
    doc = json.loads(next(tmp_path.glob("native_*.json")).read_text(encoding="utf-8"))
    assert doc["generation_failures"] == [7]
    assert doc["raw_results"][1]["gen_error"].startswith("RuntimeError:")


def test_artifact_key_present_even_when_all_good(tmp_path, monkeypatch):
    """没有失败也要有这个键（空列表），否则下游只能用 `.get()` 猜。"""
    monkeypatch.setattr(evaluate, "RESULT_DIR", tmp_path)
    evaluate.save_results([ok_row(1)], {"faithfulness": 0.5}, "native")
    doc = json.loads(next(tmp_path.glob("native_*.json")).read_text(encoding="utf-8"))
    assert doc["generation_failures"] == []
