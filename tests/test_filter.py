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
    """只测过滤语义，不测 BM25 排序结果。

    为什么：词表是进程级全局的（jieba 前缀词典），本文件可能在 test_domain_dict 之后跑，
    那时"营业收入"已经是一个 token 而不是"营业/收入"，命中集合会变。
    排序依赖分词状态，过滤不依赖 —— 守卫必须只断言它要测的那件事。
    """
    hits = store.search("五粮液 销售 收入 成本", top_k=10,
                        filter_meta={"stock_code": "000858", "year": "2022"})
    assert hits, "过滤后一条都不剩，说明过滤条件本身失效"
    assert all(h["stock_code"] == "000858" and h["year"] == "2022" for h in hits), \
        [h["chunk_id"] for h in hits]
    assert "000858_2022_00005" in {h["chunk_id"] for h in hits}


def test_bm25_filter_removes_the_would_be_top_hit(store):
    """关键一条：过滤不是"排完序再丢几条"，而是让不合规的块根本进不了候选。
    断言写成"存在被过滤掉的块"，不钉具体名次（同上，避免依赖分词状态）。"""
    unfiltered = store.search("贵州茅台营业收入 亿元 增长 显著", top_k=5)
    filtered = store.search("贵州茅台营业收入 亿元 增长 显著", top_k=5, filter_meta={"year": "2023"})
    assert unfiltered, "无过滤应该要有命中"
    assert {h["year"] for h in filtered} == {"2023"}
    dropped = {h["chunk_id"] for h in unfiltered} - {h["chunk_id"] for h in filtered}
    assert dropped, "过滤后候选集没变小 —— 过滤根本没起作用（或语料太单一，测试前提失效）"


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

    cls = next(n for n in ast.walk(tree)
               if isinstance(n, ast.ClassDef) and n.name == "RAGPipeline")
    retrieve = next(n for n in cls.body
                    if isinstance(n, ast.FunctionDef) and n.name == "retrieve")

    got: dict[str, int] = {}
    for node in ast.walk(retrieve):
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

    # query() 必须把 filter_meta 原样交给 retrieve()，否则入口处就丢了
    query = next(n for n in cls.body
                 if isinstance(n, ast.FunctionDef) and n.name == "query")
    calls = [n for n in ast.walk(query)
             if isinstance(n, ast.Call) and getattr(n.func, "attr", "") == "retrieve"]
    assert calls, "query() 没走 retrieve()"
    forwarded = any(any(isinstance(a, ast.Name) and a.id == "filter_meta" for a in c.args)
                    for c in calls)
    assert forwarded, "query() 调 retrieve() 时丢了 filter_meta"


def test_debug_endpoint_passes_filter_meta_too():
    """/query/debug 的检索必须和主链路口径一致，否则"逐步复现"是假的。"""
    import ast
    from pathlib import Path

    import serve

    src = Path(serve.__file__).read_text(encoding="utf-8")
    tree = ast.parse(src)
    fn = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "query_debug")

    # 不允许再自己复制一份检索逻辑
    direct = [
        n for n in ast.walk(fn)
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
        and n.func.attr == "search"
    ]
    assert not direct, "debug 端点又自己调了 store.search，应该统一走 pipeline.retrieve"

    calls = [n for n in ast.walk(fn)
             if isinstance(n, ast.Call) and getattr(n.func, "attr", "") == "retrieve"]
    assert calls, "debug 端点里没找到 pipeline.retrieve 调用"
    forwarded = any(any(isinstance(a, ast.Name) and a.id == "fm" for a in c.args)
                    for c in calls)
    assert forwarded, "debug 端点调 retrieve() 时丢了 filter_meta"


def test_bm25_ranking_is_deterministic_under_ties(store):
    """语料里有内容完全相同的块时，BM25 会同分。
    np.argsort(quicksort) 不稳定会让 top-k 截断随实现漂移 —— 实测同一份数据
    在两种聚合写法下 Hit@4 差 9 个百分点，所有 A/B 结论都不可信。
    这里钉住：重复查询、以及故意构造的同分语料，顺序必须完全一致。"""
    a = store.search("贵州茅台营业收入", top_k=10)
    b = store.search("贵州茅台营业收入", top_k=10)
    assert [x["chunk_id"] for x in a] == [x["chunk_id"] for x in b]

    from rank_bm25 import BM25Okapi
    import jieba
    # 2 份内容完全相同（制造真实同分）+ 4 份无关。
    # 三个坑都要避开：
    #   ① 查询词若出现在全部文档里，idf 退化为 0；
    #   ② rank_bm25 的老 idf 公式 log((N-df+0.5)/(df+0.5)) 在 df == N/2 时**恰好为 0**，
    #      所以重复份数必须小于语料一半，否则总分恒 0 被 1e-9 判据滤空；
    #   ③ 语料里的空格会被 jieba 切成独立 token 且 df=N，idf 为负，能把总分拖成负数。
    dup = rp.BM25Store.__new__(rp.BM25Store)
    dup.meta_list = (
        [{"chunk_id": f"d{i}", "content": "完全相同的内容，营业收入、净利润均增长",
          "stock_code": "600519", "year": str(2020 + i)} for i in range(2)]
        + [{"chunk_id": f"x{i}", "content": f"第{i}家其他公司，智能安防与视频监控出货",
            "stock_code": "002415", "year": str(2020 + i)} for i in range(4)]
    )
    dup.jieba = jieba
    dup.bm25 = BM25Okapi([list(jieba.cut(m["content"])) for m in dup.meta_list])
    r1 = [x["chunk_id"] for x in dup.search("营业收入、净利润", top_k=2)]
    r2 = [x["chunk_id"] for x in dup.search("营业收入、净利润", top_k=2)]
    assert r1 == r2 == ["d0", "d1"], \
        f"同分块顺序不稳定（应按下标 tie-break）: {r1} vs {r2}"
