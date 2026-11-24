"""跨公司比较的检索与覆盖守卫（2026-11-18 修复）。

修复的实测失效模式（v1 cross_doc_compare 4 题：11/12/13 全拒答、14 拿错年份资料
"自信地答错"）：

  1. 检索层从不按公司做配额召回 —— top-4 被语料里更相似的那一家占满，另一家
     一条证据都没有；
  2. `table_pool` 要求唯一 stock_code —— 跨公司题的表格补位在取数前就被整批挡掉；
  3. 没有覆盖校验 —— 模型在「只有一家资料」的半边证据上仍会被调用，于是编比较结论。

守卫按函数分层，**上层测试用 monkeypatch 固定下层输入**：每个突变只应打红它
自己那一层的守卫（命中数 =1），否则锚点无效。全部离线：不联网、不调 embedding。
"""
from __future__ import annotations

import rag_pipeline as rp

NAMES = {"600519": "贵州茅台", "000858": "五粮液"}


def chunk(stock: str, year: str, i: int, **kw) -> dict:
    base = {
        "chunk_id": f"{stock}_{year}_{i}",
        "stock_code": stock,
        "year": year,
        "page_num": 10,
        "section": "第二节 公司简介和主要财务指标",
        "content": f"{NAMES[stock]} {year} 年报正文块 {i}",
        "source_file": f"{stock}_{year}_{NAMES[stock]}_{year}年年度报告.json",
        "table_json": None,
    }
    base.update(kw)
    return base


def flat_meta() -> list[dict]:
    """两家各 6 块普通正文，顺序上 600519 在前（模拟真实语料里的偏置方向）。"""
    return ([chunk("600519", "2022", i) for i in range(6)]
            + [chunk("000858", "2022", i) for i in range(6)])


TABLE = [["项目", "2022年", "2021年"], ["总资产", "2,545,241,814,929.69", "2,184,088,141,646.84"]]


def table_meta() -> list[dict]:
    """两家各一块带 table_json 的摘要表。"""
    return [
        chunk("600519", "2022", 0, content="主要会计数据和财务指标", table_json=TABLE),
        chunk("000858", "2022", 0, content="主要会计数据和财务指标",
              table_json=[["项目", "2022年"], ["总资产", "152,724,487,378.53"]]),
    ]


def pipe_of(meta) -> rp.RAGPipeline:
    pipe = object.__new__(rp.RAGPipeline)
    pipe.vec_store = type("V", (), {"meta_list": meta})()
    pipe.parents = {}
    pipe.client = None
    return pipe


# ── L1：实体推断（跨公司题的唯一入口判据）────────────────────────────────────

def test_l1_infer_companies_lists_every_named_company():
    meta = flat_meta()
    assert rp.infer_companies_from_question(
        "比较贵州茅台和五粮液2022年的销售毛利率，哪家公司更高？", meta) == ["000858", "600519"]
    assert rp.infer_companies_from_question(
        "600519 与 000858 在 2022 年谁的总资产更高？", meta) == ["000858", "600519"]
    assert rp.infer_companies_from_question(
        "贵州茅台2022年的总资产是多少？", meta) == ["600519"]
    assert rp.infer_companies_from_question("今年的毛利率如何？", meta) == []


# ── L2：表格兜底池的多公司入口 ───────────────────────────────────────────────

def test_l2_table_pool_accepts_multiple_codes():
    pool = pipe_of(table_meta()).table_pool(question="", stock_codes=["000858", "600519"])
    assert sorted({p["stock_code"] for p in pool}) == ["000858", "600519"], \
        "跨公司题的池子只进一家 = 修复前的失效模式"


def test_l2_single_company_path_unchanged():
    pool = pipe_of(table_meta()).table_pool(question="贵州茅台2022年的总资产是多少")
    assert {p["stock_code"] for p in pool} == {"600519"}


def test_l2_still_refuses_without_any_company():
    """硬闸不变：没有高置信公司就不取数，防跨公司串数。"""
    assert pipe_of(table_meta()).table_pool(question="2022年的总资产是多少") == []


# ── L3：补位是否把全部被点名公司传给池子 ─────────────────────────────────────

def test_l3_backfill_forwards_all_named_companies(monkeypatch):
    monkeypatch.setattr(rp, "infer_companies_from_question",
                        lambda q, m: ["000858", "600519"])
    seen: dict = {}

    def fake_pool(filter_meta=None, question="", stock_codes=None):
        seen["codes"] = stock_codes
        return []

    pipe = pipe_of(table_meta())
    monkeypatch.setattr(pipe, "table_pool", fake_pool)
    assert pipe._table_context_items("2022年，贵州茅台和五粮液的总资产相比哪家更高？") == []
    assert seen["codes"] == ["000858", "600519"], "补位只带一家 = 比较题永远缺证据"


# ── L4：覆盖校验 ─────────────────────────────────────────────────────────────

def test_l4_missing_companies_detects_zero_evidence():
    items = [chunk("600519", "2022", 0)]
    assert rp.missing_companies(["000858", "600519"], items) == ["000858"]
    assert rp.missing_companies(["000858", "600519"], []) == ["000858", "600519"]
    both = [chunk("600519", "2022", 0), chunk("000858", "2022", 0)]
    assert rp.missing_companies(["000858", "600519"], both) == []


def test_l4_merged_backfill_item_counts_as_coverage():
    """表格补位把两家数值合并在**一条** content 里，字段只有第一家 —— 必须看正文。"""
    merged = {"chunk_id": "x", "stock_code": "000858", "year": "2022",
              "content": "【表格精确取数】600519 2022年报 · 总资产：2,545,241,814,929.69\n"
                         "【表格精确取数】000858 2022年报 · 总资产：152,724,487,378.53"}
    assert rp.missing_companies(["000858", "600519"], [merged]) == []


# ── L5 / L6：query 的覆盖闸 ──────────────────────────────────────────────────

def _stub_retrieve(pipe, final):
    def retrieve(question, filter_meta=None, verbose=False):
        return {"query": question, "vec_results": [], "bm25_results": [],
                "candidates": list(final), "final": list(final)}
    pipe.retrieve = retrieve
    return pipe


def _gate_pipe(monkeypatch, final, missing):
    monkeypatch.setattr(rp, "infer_companies_from_question",
                        lambda q, m: ["000858", "600519"])
    monkeypatch.setattr(rp, "missing_companies", lambda codes, items: list(missing))
    pipe = _stub_retrieve(pipe_of(flat_meta()), final)
    # 补位与本守卫无关，钉成空，避免真实 table_pool 影响闸的判定
    monkeypatch.setattr(pipe, "_table_context_items", lambda q, f=None: [])
    return pipe


def test_l5_refuses_without_calling_llm_when_a_company_has_no_evidence(monkeypatch):
    def boom(*a, **k):
        raise AssertionError("半边证据上不得调用 LLM")
    monkeypatch.setattr(rp, "call_llm", boom)
    final = [dict(chunk("600519", "2022", 0), vec_score=0.9)]
    pipe = _gate_pipe(monkeypatch, final, missing=["000858"])
    out = pipe.query("2022年，贵州茅台和五粮液的总资产相比哪家更高？")
    assert "000858" in out["answer"] and "无法完成跨公司比较" in out["answer"]
    assert out["citations"] == []
    assert len(out["retrieved"]) == 1, "拒答响应仍要带上当次检索到的上下文，便于排障"


def test_l6_answers_via_llm_when_every_company_has_evidence(monkeypatch):
    monkeypatch.setattr(rp, "call_llm", lambda q, ctx, c: "OK")
    final = [dict(chunk("600519", "2022", 0), vec_score=0.9),
             dict(chunk("000858", "2022", 0), vec_score=0.9)]
    pipe = _gate_pipe(monkeypatch, final, missing=[])
    assert pipe.query("2022年，贵州茅台和五粮液的总资产相比哪家更高？")["answer"] == "OK"


def test_l6b_single_company_questions_are_not_gated(monkeypatch):
    """覆盖闸只管跨公司题：单公司题即使检索到的公司"不匹配"也不该被新闸拦。"""
    monkeypatch.setattr(rp, "infer_companies_from_question", lambda q, m: ["600519"])
    monkeypatch.setattr(rp, "missing_companies", lambda codes, items: list(codes))
    monkeypatch.setattr(rp, "call_llm", lambda q, ctx, c: "OK")
    pipe = _stub_retrieve(pipe_of(flat_meta()), [dict(chunk("600519", "2022", 0), vec_score=0.9)])
    monkeypatch.setattr(pipe, "_table_context_items", lambda q, f=None: [])
    assert pipe.query("贵州茅台2022年的总资产是多少？")["answer"] == "OK"


# ── L7：多公司检索分支 ───────────────────────────────────────────────────────

def test_l7_multi_company_retrieve_covers_every_company(monkeypatch):
    monkeypatch.setattr(rp, "infer_companies_from_question",
                        lambda q, m: ["000858", "600519"])
    meta = flat_meta()

    def fake_search(query, top_k, filter_meta=None):
        if filter_meta and filter_meta.get("stock_code"):
            code = filter_meta["stock_code"]
            return [dict(it, vec_score=0.9) for it in meta if it["stock_code"] == code][:top_k]
        # 无过滤：更相似的一家整体排在前面 —— 这就是修复前 top-4 被一家占满的根因
        ordered = ([it for it in meta if it["stock_code"] == "600519"]
                   + [it for it in meta if it["stock_code"] == "000858"])
        return [dict(it, vec_score=0.9) for it in ordered][:top_k]

    pipe = pipe_of(meta)
    pipe.vec_store = type("V", (), {"meta_list": meta, "search": staticmethod(fake_search)})()
    pipe.use_bm25, pipe.bm25_store = False, None
    pipe.use_rerank, pipe.use_qr, pipe.rerank_min_score = False, False, None

    out = pipe.retrieve("2022年，贵州茅台和五粮液的总资产相比哪家更高？")
    codes = [it["stock_code"] for it in out["final"]]
    assert {"000858", "600519"} <= set(codes), f"仍有一家没进上下文：{codes}"
    assert len(out["final"]) <= rp.TOP_K_RERANK, "按公司分配不得让上下文条数膨胀"


def test_l7b_explicit_filter_meta_keeps_the_manual_path(monkeypatch):
    """调用方显式给了 filter_meta（serve 的 stock/year 参数）时不走多公司分支。"""
    monkeypatch.setattr(rp, "infer_companies_from_question",
                        lambda q, m: ["000858", "600519"])
    calls: list = []
    meta = flat_meta()

    def fake_search(query, top_k, filter_meta=None):
        calls.append(filter_meta)
        return [dict(it, vec_score=0.9) for it in meta][:top_k]

    pipe = pipe_of(meta)
    pipe.vec_store = type("V", (), {"meta_list": meta, "search": staticmethod(fake_search)})()
    pipe.use_bm25, pipe.bm25_store = False, None
    pipe.use_rerank, pipe.use_qr, pipe.rerank_min_score = False, False, None

    pipe.retrieve("贵州茅台和五粮液2022年总资产", filter_meta={"year": "2022"})
    assert calls == [{"year": "2022"}], "显式过滤被改写成按公司召回 = 越权"


# ── L8：生成规则里的资料一致性约束 ───────────────────────────────────────────

def test_l8_prompt_forbids_mismatched_company_or_year():
    """Q14 型失败的护栏：拿 2023 年报的 AI 描述冒充 2022 年作答。"""
    assert "公司或年份不符" in rp.SYSTEM_PROMPT
