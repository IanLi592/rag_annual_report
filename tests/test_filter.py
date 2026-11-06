"""元数据过滤的回归守卫（本轮修复的命门）。

缺陷背景：BM25Store.search 原先不收 filter_meta，而 use_bm25=True 是默认路径，
导致 /query 里 year=2023 的请求仍会召回 600519_2022_00205（2022年报）。
修复后向量侧与 BM25 侧共用 match_filter 这一份谓词。

零密钥、零网络：用 6 条合成语料直接喂 BM25Okapi，不碰 10352 条真实索引。
"""
from __future__ import annotations

import pytest

import rag_pipeline as rp


# ── 谓词本身 ─────────────────────────────────────────────────────────────────

def test_match_filter_empty_passes_everything():
    assert rp.match_filter({"year": "2021"}, None) is True
    assert rp.match_filter({"year": "2021"}, {}) is True


def test_match_filter_requires_all_keys_to_match():
    item = {"stock_code": "600519", "year": "2023"}
    assert rp.match_filter(item, {"year": "2023"}) is True
    assert rp.match_filter(item, {"year": "2023", "stock_code": "600519"}) is True
    assert rp.match_filter(item, {"year": "2022"}) is False           # 单键不符
    assert rp.match_filter(item, {"year": "2023", "stock_code": "0"}) is False  # 多键里一个不符即 False


def test_match_filter_missing_key_compares_as_empty_string():
    """语义钉死：缺字段当 ""，所以按 stock_code 过滤会把没有该字段的块排除。"""
    assert rp.match_filter({"year": "2023"}, {"stock_code": "600519"}) is False
    assert rp.match_filter({"stock_code": ""}, {"stock_code": ""}) is True


def test_match_filter_coerces_both_sides_to_str():
    """索引里 year 是 str，调用方传 int 也应该匹上（两侧都 str()）。"""
    assert rp.match_filter({"page_num": "61"}, {"page_num": 61}) is True
    assert rp.match_filter({"page_num": 61}, {"page_num": "61"}) is True


# ── BM25 链路真的过滤了 ───────────────────────────────────────────────────────

CORPUS = [
    {"chunk_id": "600519_2023_00001", "content": "贵州茅台营业收入 亿元 增长", "stock_code": "600519", "year": "2023"},
    {"chunk_id": "600519_2022_00002", "content": "贵州茅台营业收入 亿元 增长 显著", "stock_code": "600519", "year": "2022"},
    {"chunk_id": "600519_2021_00003", "content": "贵州茅台营业总收入 提升", "stock_code": "600519", "year": "2021"},
    {"chunk_id": "000858_2023_00004", "content": "五粮液营业收入 亿元 增长", "stock_code": "000858", "year": "2023"},
    {"chunk_id": "000858_2022_00005", "content": "五粮液销售 收入 成本", "stock_code": "000858", "year": "2022"},
    {"chunk_id": "600519_2023_00006", "content": "贵州茅台 净利润 分红", "stock_code": "600519", "year": "2023"},
]


@pytest.fixture()
def store():
    """绕过 __init__（它会读 15 MB 真实 meta 并分词 10352 条），手搭一个小语料库。"""
    from rank_bm25 import BM25Okapi
    import jieba

    s = rp.BM25Store.__new__(rp.BM25Store)
    s.meta_list = CORPUS
    s.jieba = jieba
    s.bm25 = BM25Okapi([list(jieba.cut(d["content"])) for d in CORPUS])
    return s


def test_bm25_search_without_filter_returns_cross_year_hits(store):
    hits = store.search("贵州茅台营业收入", top_k=10)
    assert hits, "无过滤时应命中"
    assert {h["year"] for h in hits} != {"2023"}, "语料里 2021/2022 也该被召回（对照用）"


def test_bm25_search_respects_year_filter(store):
    hits = store.search("贵州茅台营业收入", top_k=10, filter_meta={"year": "2023"})
    assert hits
    assert all(h["year"] == "2023" for h in hits), [h["chunk_id"] for h in hits]


def test_bm25_search_respects_stock_and_year_together(store):
    hits = store.search("营业收入", top_k=10, filter_meta={"stock_code": "000858", "year": "2022"})
    assert [h["chunk_id"] for h in hits] == ["000858_2022_00005"]


def test_bm25_filter_removes_the_would_be_top_hit(store):
    """关键一条：过滤不是"排完序再丢几条"，而是让不合规的块根本进不了候选。
    2022 那条语料更长、词频更高，无过滤时排第一；加了 year=2023 后它必须消失。"""
    unfiltered = store.search("贵州茅台营业收入 亿元 增长 显著", top_k=3)
    filtered = store.search("贵州茅台营业收入 亿元 增长 显著", top_k=3, filter_meta={"year": "2023"})
    assert unfiltered[0]["chunk_id"] == "600519_2022_00002"
    assert all(h["chunk_id"] != "600519_2022_00002" for h in filtered)
    assert {h["year"] for h in filtered} == {"2023"}


def test_bm25_filter_matching_nothing_returns_empty(store):
    assert store.search("贵州茅台", top_k=5, filter_meta={"year": "1999"}) == []


def test_bm25_filtered_scores_are_preserved(store):
    """被保留下来的块分数不能被 np.where 改写。"""
    hits = store.search("五粮液营业收入", top_k=5, filter_meta={"stock_code": "000858"})
    assert all(h["bm25_score"] > 0 for h in hits)


def test_both_retrieval_paths_share_one_predicate(monkeypatch, store):
    """向量侧与 BM25 侧必须用同一个 match_filter —— 防止以后又各自长出一套口径。"""
    calls: list[tuple] = []
    real = rp.match_filter

    def spy(item, filter_meta):
        calls.append((item.get("chunk_id"), filter_meta))
        return real(item, filter_meta)

    monkeypatch.setattr(rp, "match_filter", spy)
    store.search("贵州茅台", top_k=3, filter_meta={"year": "2023"})

    assert calls, "BM25 链路没走共享谓词"
    assert {c[1]["year"] for c in calls} == {"2023"}


# ── 接线守卫：光测函数不够，得钉住"调用点真的把 filter 传下去了" ──────────────

def test_pipeline_call_sites_pass_filter_meta():
    """
    AST 静态守卫：RAGPipeline.query 里对 vec_store.search / bm25_store.search 的调用
    都必须带 filter_meta。

    为什么需要：只看 BM25Store.search 的实现，删掉调用点的第三个实参照样全绿 ——
    而那正是本次缺陷的原样（能力存在但没接上）。
    """
    import ast
    import inspect
    from pathlib import Path

    src = Path(inspect.getsourcefile(rp)).read_text(encoding="utf-8")
    tree = ast.parse(src)

    def is_self(node) -> bool:
        return (isinstance(node, ast.Name) and node.id == "self") or (
            isinstance(node, ast.Attribute) and node.attr == "self"
        )

    fn = next(
        n for n in ast.walk(tree)
        if isinstance(n, ast.FunctionDef) and n.name == "query"
        and n.args.args and n.args.args[0].arg == "self"
    )

    got: dict[str, int] = {}
    for node in ast.walk(fn):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "search"
            and isinstance(node.func.value, ast.Attribute)
            and is_self(node.func.value.value)
        ):
            got[node.func.value.attr] = len(node.args) + len(node.keywords)

    assert "vec_store" in got and "bm25_store" in got, f"没找到两路检索调用，只看到 {got}"
    # 3 个实参 = query, top_k, filter_meta
    assert got["vec_store"] >= 3, "向量侧调用丢了 filter_meta"
    assert got["bm25_store"] >= 3, "BM25 侧调用丢了 filter_meta —— 就是修前那个缺陷"


def test_debug_endpoint_passes_filter_meta_too():
    """/query/debug 的 BM25 调用必须和主链路口径一致，否则"逐步复现"是假的。"""
    import ast
    from pathlib import Path

    import serve

    src = Path(serve.__file__).read_text(encoding="utf-8")
    tree = ast.parse(src)
    fn = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "query_debug")

    calls = [
        n for n in ast.walk(fn)
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
        and n.func.attr == "search" and isinstance(n.func.value, ast.Attribute)
        and n.func.value.attr == "bm25_store"
    ]
    assert calls, "debug 端点里没找到 bm25_store.search 调用"
    for c in calls:
        assert len(c.args) + len(c.keywords) >= 3, "debug 端点的 BM25 调用丢了 filter_meta"
