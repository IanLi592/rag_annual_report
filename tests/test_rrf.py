"""RRF 融合（rag_pipeline.py:164-193）的回归守卫（P1-1）。

RRF 是混合检索的核心，公式写错或 k 值被动过，向量/BM25 的互补关系就静默失效。
全部纯内存、零密钥。
"""
from __future__ import annotations

import rag_pipeline as rp


def test_rrf_score_matches_formula_with_k60():
    """score(d) = Σ 1/(k + rank)，k=60。手工算一遍钉死。"""
    vec = [{"chunk_id": "A"}, {"chunk_id": "B"}]   # A 排名1，B 排名2
    bm25 = [{"chunk_id": "B"}, {"chunk_id": "C"}]  # B 排名1，C 排名2
    out = rp.reciprocal_rank_fusion(vec, bm25)

    scores = {c["chunk_id"]: c["rrf_score"] for c in out}
    assert scores["A"] == 1 / 61
    assert scores["B"] == 1 / 62 + 1 / 61  # 两个列表里都有它 → 累加
    assert scores["C"] == 1 / 62
    assert [c["chunk_id"] for c in out] == ["B", "A", "C"]  # 分数降序


def test_rrf_dedupes_by_chunk_id():
    """同一 chunk 只能出现一次，否则上下文里会有重复段落喂给 LLM。"""
    out = rp.reciprocal_rank_fusion([{"chunk_id": "X"}], [{"chunk_id": "X"}])
    ids = [c["chunk_id"] for c in out]
    assert len(ids) == 1
    assert out[0]["rrf_score"] == 1 / 61 + 1 / 61


def test_rrf_later_list_metadata_wins_for_shared_chunk():
    """既有行为：两边都有的 chunk，最终 item 取 BM25 侧那份（源码 chunk_map 后写覆盖）。
    这条不是"我觉得对"，是把现状钉住 —— 要改得先让它变红、再显式决策。"""
    vec = [{"chunk_id": "B", "src_marker": "vec"}]
    bm25 = [{"chunk_id": "B", "src_marker": "bm25"}]
    out = rp.reciprocal_rank_fusion(vec, bm25)
    assert out[0]["src_marker"] == "bm25"


def test_rrf_ties_keep_insertion_order():
    """分数相同则按插入顺序（Python sort 稳定）。这条防的是"改用不稳定排序"。"""
    out = rp.reciprocal_rank_fusion([{"chunk_id": "A"}], [{"chunk_id": "B"}])
    assert [c["chunk_id"] for c in out] == ["A", "B"]


def test_rrf_with_empty_inputs():
    assert rp.reciprocal_rank_fusion([], []) == []


def test_rrf_k_is_configurable():
    out = rp.reciprocal_rank_fusion([{"chunk_id": "A"}], [], k=10)
    assert out[0]["rrf_score"] == 1 / 11
