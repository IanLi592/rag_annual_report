"""消融实验的指标计算（compare_strategies.py:177-223）回归守卫（P1-1）。

compute_metrics 内部会调 retrieve_* 打 API，所以这里 monkeypatch 掉检索、
只考它自己的数学：命中判定、MRR 倒数、top_k 截断、should_refuse 跳过。
零密钥、零网络。
"""
from __future__ import annotations

import compare_strategies as cs
import pytest


def fake_bm25_factory(per_question: dict[str, list[dict]]):
    """返回一个替身检索函数，并记录每次被请求的 k。"""
    calls: list[int] = []

    def _fake(query: str, meta_list: list, top_k: int = 10) -> list[dict]:
        calls.append(top_k)
        return per_question[query]

    return _fake, calls


def item(code: str, year: str) -> dict:
    return {"chunk_id": f"{code}_{year}_00000", "stock_code": code, "year": year, "content": "x"}


def test_hit_at_rank_one_gives_perfect_scores(monkeypatch):
    qs = [{"question": "q1", "target_docs": ["600519_2021"]}, {"question": "q2", "target_docs": ["000858_2023"]}]
    fake, calls = fake_bm25_factory({"q1": [item("600519", "2021")], "q2": [item("000858", "2023")]})
    monkeypatch.setattr(cs, "retrieve_bm25", fake)

    out = cs.compute_metrics(qs, index=None, meta_list=[], retrieval_mode="bm25_only", top_k=4)
    assert out["hit_rate"] == pytest.approx(1.0)
    assert out["mrr"] == pytest.approx(1.0)
    assert out["n_questions"] == 2
    assert calls == [8, 8]  # top_k * 2，召回池比评估口径大一倍


def test_mrr_uses_reciprocal_of_first_hit_rank(monkeypatch):
    qs = [{"question": "q1", "target_docs": ["600519_2021"]}]
    ranked = [item("000001", "2020"), item("000002", "2020"), item("600519", "2021"), item("600519", "2021")]
    fake, _ = fake_bm25_factory({"q1": ranked})
    monkeypatch.setattr(cs, "retrieve_bm25", fake)

    out = cs.compute_metrics(qs, None, [], "bm25_only", top_k=4)
    assert out["hit_rate"] == pytest.approx(1.0)
    assert out["mrr"] == pytest.approx(1 / 3)


def test_miss_counts_as_zero(monkeypatch):
    qs = [{"question": "q1", "target_docs": ["600519_2021"]}]
    fake, _ = fake_bm25_factory({"q1": [item("000001", "2020")]})
    monkeypatch.setattr(cs, "retrieve_bm25", fake)

    out = cs.compute_metrics(qs, None, [], "bm25_only", top_k=4)
    assert out["hit_rate"] == pytest.approx(0.0)
    assert out["mrr"] == pytest.approx(0.0)


def test_hit_beyond_top_k_is_not_counted(monkeypatch):
    """召回 top_k*2 条，但评分只看前 top_k 条（compare_strategies.py:204）。
    改坏这行截断会让 hit_rate 虚高，消融结论就废了。"""
    qs = [{"question": "q1", "target_docs": ["600519_2021"]}]
    ranked = [item("000001", "2020") for _ in range(4)] + [item("600519", "2021")]
    fake, calls = fake_bm25_factory({"q1": ranked})
    monkeypatch.setattr(cs, "retrieve_bm25", fake)

    out = cs.compute_metrics(qs, None, [], "bm25_only", top_k=4)
    assert calls == [8]  # 确实召回了 8 条
    assert out["hit_rate"] == pytest.approx(0.0)
    assert out["mrr"] == pytest.approx(0.0)


def test_questions_without_target_docs_are_skipped(monkeypatch):
    """should_refuse 类没有 target_docs，不能进分母，否则 hit_rate 被人为压低。"""
    qs = [
        {"question": "q1", "target_docs": ["600519_2021"]},
        {"question": "refuse1", "target_docs": []},
        {"question": "refuse2"},  # 连 target_docs 键都没有
    ]
    fake, calls = fake_bm25_factory({"q1": [item("600519", "2021")]})
    monkeypatch.setattr(cs, "retrieve_bm25", fake)

    out = cs.compute_metrics(qs, None, [], "bm25_only", top_k=4)
    assert out["n_questions"] == 1
    assert calls == [8]
    assert out["hit_rate"] == pytest.approx(1.0)


def test_multiple_target_docs_any_one_matches(monkeypatch):
    """对比题会有多个目标文档，命中任意一个即算 hit（:211 any(...)）。"""
    qs = [{"question": "q1", "target_docs": ["600519_2021", "000858_2021"]}]
    fake, _ = fake_bm25_factory({"q1": [item("000858", "2021")]})
    monkeypatch.setattr(cs, "retrieve_bm25", fake)

    out = cs.compute_metrics(qs, None, [], "bm25_only", top_k=4)
    assert out["hit_rate"] == pytest.approx(1.0)


def test_all_questions_skipped_yields_zeros(monkeypatch):
    """空分母不能让 np.mean 抛异常（源码用 `if hits else 0.0` 兜住）。"""
    qs = [{"question": "r1", "target_docs": []}]
    fake, calls = fake_bm25_factory({})
    monkeypatch.setattr(cs, "retrieve_bm25", fake)

    out = cs.compute_metrics(qs, None, [], "bm25_only", top_k=4)
    assert out == {"hit_rate": 0.0, "mrr": 0.0, "n_questions": 0}
    assert calls == []
