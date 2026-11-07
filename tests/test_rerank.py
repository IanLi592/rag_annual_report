"""Rerank 漏斗的守卫（离线，用假 reranker，不加载 1.1 GB 模型）。

守的是三件容易悄悄坏掉的事：
  1. min_score 过滤真的在过滤（这是简历那条"削减上下文"的唯一实现基础）
  2. 不传 min_score 时行为与旧版一致（永远返回 top_k）—— 向后兼容
  3. 模型只加载一次（旧版每次请求都重新加载，实测 5.8 s/请求）
  4. 模型不可用时降级为截断，不抛异常
"""
from __future__ import annotations

from pathlib import Path

import pytest

import rag_pipeline as rp


class FakeReranker:
    def __init__(self, scores):
        self.scores = scores
        self.calls = 0

    def predict(self, pairs):
        self.calls += 1
        return self.scores


@pytest.fixture(autouse=True)
def isolate(monkeypatch):
    monkeypatch.setattr(rp, "_RERANKER", None)
    monkeypatch.setattr(rp, "_RERANKER_FAILED", False)
    yield


def cand(cid, content="x" * 50, **meta):
    return {"chunk_id": cid, "content": content, "rrf_score": 1.0, **meta}


def test_scores_are_written_and_sorted_desc(monkeypatch):
    monkeypatch.setattr(rp, "get_reranker", lambda: FakeReranker([0.1, 0.9, 0.5]))
    out = rp.rerank("q", [cand("a"), cand("b"), cand("c")], top_k=3)
    assert [c["chunk_id"] for c in out] == ["b", "c", "a"]
    assert out[0]["rerank_score"] == pytest.approx(0.9)


def test_without_floor_always_returns_top_k(monkeypatch):
    """旧行为：只排序+截断，一条都不滤。这条钉住"向后兼容"，
    也解释了为什么旧代码下"上下文削减 X%"恒为 0。"""
    monkeypatch.setattr(rp, "get_reranker", lambda: FakeReranker([0.99, 0.0, 0.0, 0.0]))
    out = rp.rerank("q", [cand(str(i)) for i in range(4)], top_k=4)
    assert len(out) == 4, "不传 min_score 却少返回了，等于偷偷改了线上行为"


def test_top_k_truncation_is_enforced(monkeypatch):
    """候选多于 top_k 时必须截断 —— 两条路径都要截。
    （上一版测试恰好用了"候选数 == top_k"，截断不可见，变异测试当场暴露了这个盲区）"""
    monkeypatch.setattr(rp, "get_reranker", lambda: FakeReranker([0.9, 0.8, 0.7, 0.6, 0.5, 0.4]))
    cands = [cand(str(i)) for i in range(6)]

    no_floor = rp.rerank("q", cands, top_k=3)
    assert len(no_floor) == 3, f"不传 min_score 时没截断到 top_k，返回 {len(no_floor)} 条"
    assert [c["rerank_score"] for c in no_floor] == [0.9, 0.8, 0.7]

    with_floor = rp.rerank("q", cands, top_k=2, min_score=0.5)
    assert len(with_floor) == 2, f"传了 min_score 时没截断到 top_k，返回 {len(with_floor)} 条"


def test_floor_drops_long_tail(monkeypatch):
    monkeypatch.setattr(rp, "get_reranker", lambda: FakeReranker([0.95, 0.6, 0.3, 0.1]))
    out = rp.rerank("q", [cand(str(i)) for i in range(4)], top_k=4, min_score=0.5)
    assert [c["rerank_score"] for c in out] == [0.95, 0.6]
    assert all(c["rerank_score"] >= 0.5 for c in out)


def test_floor_still_respects_top_k(monkeypatch):
    monkeypatch.setattr(rp, "get_reranker", lambda: FakeReranker([0.9, 0.9, 0.9]))
    out = rp.rerank("q", [cand(str(i)) for i in range(3)], top_k=2, min_score=0.5)
    assert len(out) == 2


def test_all_below_floor_returns_empty(monkeypatch):
    """全低分时返回空上下文而不是硬塞一条 —— 宁可让生成侧走"资料不足"分支。"""
    monkeypatch.setattr(rp, "get_reranker", lambda: FakeReranker([0.1, 0.2]))
    out = rp.rerank("q", [cand("a"), cand("b")], top_k=4, min_score=0.5)
    assert out == []


def test_reranker_is_loaded_once():
    """旧实现每次 rerank() 都 CrossEncoder(...)，实测模型加载 5.8 s/请求。
    这条守卫钉住单例：两次调用只构造一次。"""
    builds = []

    class FakeCE:
        def __init__(self, name):
            builds.append(name)

        def predict(self, pairs):
            return [0.5] * len(pairs)

    import sys, types
    mod = types.ModuleType("sentence_transformers")
    mod.CrossEncoder = FakeCE
    sys.modules["sentence_transformers"] = mod
    try:
        rp.get_reranker()
        rp.get_reranker()
        assert len(builds) == 1, f"模型被重复加载 {len(builds)} 次"
    finally:
        sys.modules.pop("sentence_transformers", None)


def test_degrades_to_truncation_when_unavailable(monkeypatch):
    monkeypatch.setattr(rp, "get_reranker", lambda: None)
    out = rp.rerank("q", [cand(str(i)) for i in range(6)], top_k=4)
    assert len(out) == 4, "模型不可用时应退回截断，且仍受 top_k 约束"
    assert [c["chunk_id"] for c in out] == ["0", "1", "2", "3"]


def test_load_failure_is_not_retried_every_request(monkeypatch):
    """加载失败要记住失败，否则每个请求都重付一次失败加载的代价。"""
    import sys, types
    boom = types.ModuleType("sentence_transformers")

    class Bad:
        def __init__(self, name):
            raise RuntimeError("模型文件损坏")
    boom.CrossEncoder = Bad
    sys.modules["sentence_transformers"] = boom
    try:
        assert rp.get_reranker() is None
        assert rp._RERANKER_FAILED is True
        assert rp.get_reranker() is None      # 第二次直接返回 None，不再尝试
    finally:
        sys.modules.pop("sentence_transformers", None)


# ── 接线守卫：能力存在但没接上，等于没做（BM25 过滤那次就是这么翻车的）──

def test_pipeline_query_forwards_min_score():
    """RAGPipeline.query 必须把 self.rerank_min_score 传给 rerank()，否则新参数永远是死的。"""
    import ast
    tree = ast.parse(Path(rp.__file__).read_text(encoding="utf-8"))
    cls = next(n for n in ast.walk(tree) if isinstance(n, ast.ClassDef) and n.name == "RAGPipeline")
    fn = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "query")
    calls = [n for n in ast.walk(fn)
             if isinstance(n, ast.Call) and getattr(n.func, "id", "") == "rerank"]
    assert calls, "query() 里没有 rerank 调用"
    forwarded = any(any(isinstance(a, ast.Attribute) and a.attr == "rerank_min_score"
                        for a in c.args) for c in calls)
    assert forwarded, "rerank() 没收到 rerank_min_score —— 长尾过滤在真实链路上是死代码"


def test_serve_enables_rerank_with_threshold():
    """serve.py 必须真的打开精排。旧版显式 use_rerank=False，
    于是"两阶段漏斗 + BGE 重排"在 HTTP 服务上从未生效。"""
    import ast
    import serve
    tree = ast.parse(Path(serve.__file__).read_text(encoding="utf-8"))
    call = next((n for n in ast.walk(tree)
                 if isinstance(n, ast.Call) and getattr(n.func, "attr", "") == "RAGPipeline"), None)
    assert call, "serve.py 里没有构造 RAGPipeline"
    kw = {k.arg: k.value for k in call.keywords}
    assert isinstance(kw.get("use_rerank"), ast.Constant) and kw["use_rerank"].value is True, \
        "use_rerank 没被打开"
    assert "rerank_min_score" in kw, "没配长尾过滤阈值"
    # 阈值改由 RERANK_MIN_SCORE 环境变量控制：默认 None（不误杀），
    # 要启用必须先按 100 题重定。这里允许 None 或 _env_float(...) 两种写法。
    value = kw["rerank_min_score"]
    assert isinstance(value, (ast.Constant, ast.Call))
    if isinstance(value, ast.Constant):
        assert value.value is None or (isinstance(value.value, (int, float)) and 0 < value.value < 1)
    else:
        assert getattr(value.func, "id", "") == "_env_float", \
            "rerank_min_score 既不是常量也不是 _env_float 读取"
