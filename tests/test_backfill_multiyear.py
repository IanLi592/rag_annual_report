"""跨年趋势 / 多来源表格的补位守卫（2026-11-17 修复）。

钉住的行为，全部离线（不联网、不调 embedding、不花钱）：

  1. 多年题不再在取数前就被拒（旧闸：问题里出现多个年份 → 推断不出唯一 year →
     `return []`，实测 multihop 34 题里兜底通道一条都没进到上下文）
  2. 同一指标按 (公司, 年度) 分组各取一行，但**合并成一条**上下文 —— 槽位数不变
  3. 组内选行看表来源优先级：合并/摘要 > 母公司 > 附注 > 其他，而不是「块顺序谁在前」
  4. 摘要表与合并报表同档时保持池子原顺序 —— 这是为了让 numeric 那批本来答对的题
     不被无端改掉（实测改前改后 numeric 的证据覆盖 32/36 → 33/36）
  5. 单年题的输出字符串与修复前逐字一致
  6. 没有公司线索仍然不取数（跨公司串数的硬闸不变）。
     跨公司对比题的补位已于 2026-11-18 修复（table_pool 支持多公司代码集合），
     分层守卫见 tests/test_cross_doc.py
"""
from __future__ import annotations

import rag_pipeline as rp

SEC_FS = "第十节 财务报告 > 二、财务报表"
SEC_SUMMARY = "第二节 公司简介和主要财务指标 > 六、主要会计数据和财务指标"
SEC_NOTES = "第十节 财务报告 > 五、重要会计政策及会计估计"


def item(**kw):
    """一块表格。chunk_id 故意不给默认值：给默认值会让 doc() 里的 setdefault
    永远盖不上，同一年报的两块表就共用一个 chunk_id —— 而 `select_backfill_rows`
    正是按 chunk_id 回查来源块的，撞车后所有行都拿到同一个优先级（实测就是这里
    让「合并优先于母公司/附注」两条断言红了）。"""
    base = {"page_num": 63, "section": SEC_FS, "content": "", "table_json": None}
    base.update(kw)
    return base


def doc(stock, year, source, items):
    """一份年报的若干表格块。

    stock_code / year / source_file / chunk_id 在这里无条件覆盖（除非调用方
    显式传了 chunk_id）—— 之前用 setdefault 配上 item() 的默认值，三份 fixture
    全变成同一年度、同一块，分组只剩一组，测试假失败。
    """
    out = []
    for i, it in enumerate(items):
        m = dict(it)
        m["stock_code"] = stock
        m["year"] = year
        m["source_file"] = source
        m["chunk_id"] = m.get("chunk_id") or f"{stock}_{year}_{i}"
        out.append(m)
    ids = [m["chunk_id"] for m in out]
    assert len(set(ids)) == len(ids), f"同一份文档里 chunk_id 撞车：{ids}"
    return out


def pipe_of(meta):
    pipe = object.__new__(rp.RAGPipeline)
    pipe.vec_store = type("V", (), {"meta_list": meta})()
    return pipe


def summary_table(label, value, prev="1,000.00"):
    return [["项目", "2023年", "2022年"], [label, value, prev]]


# ── 1 + 2：多年题拿到三年，且仍然只占一条上下文 ──────────────────────────────

TREND_Q = "贵州茅台的总资产在2021到2023年整体是上升还是下降？各年分别是多少？"


def _three_year_meta():
    meta = []
    for year, val in (("2021", "255,168,195,159.90"),
                      ("2022", "254,364,804,995.25"),
                      ("2023", "272,699,660,092.25")):
        meta += doc("600519", year, f"600519_{year}_贵州茅台_{year}年年度报告.json", [
            item(section=SEC_SUMMARY, content=f"主要会计数据 {year}",
                 table_json=[["项目", f"{year}年", f"{int(year)-1}年"],
                             ["总资产", val, "200,000,000,000.00"]]),
        ])
    return meta


def test_three_year_question_gets_all_three_values():
    items = pipe_of(_three_year_meta())._table_context_items(TREND_Q)
    assert items, "跨年题被闸挡住 = 修复前的失效模式"
    blob = items[0]["content"]
    for v in ("255,168,195,159.90", "254,364,804,995.25", "272,699,660,092.25"):
        assert v in blob, f"{v} 没进上下文"


def test_multi_year_backfill_still_one_context_item():
    """槽位不动：三个年度合并进**一条**补位上下文。"""
    items = pipe_of(_three_year_meta())._table_context_items(TREND_Q)
    assert len(items) == 1
    assert items[0]["content"].count("【表格精确取数】") == 3
    assert items[0]["stock_code"] == "600519"


def test_backfill_row_cap_is_three_and_constant_is_pinned():
    assert rp.MAX_BACKFILL_ROWS == 3
    meta = []
    for year in ("2019", "2020", "2021", "2022", "2023"):
        meta += doc("600519", year, f"600519_{year}_贵州茅台_{year}年年度报告.json", [
            item(section=SEC_SUMMARY, content="主要会计数据",
                 table_json=[["项目", f"{year}年"], ["总资产", f"{year}00,000.00"]]),
        ])
    items = pipe_of(meta)._table_context_items(
        "贵州茅台的总资产在2019到2023年各年分别是多少？")
    assert items[0]["content"].count("【表格精确取数】") == 3, "上限必须钉死，防上下文失控"


# ── 3：组内按表来源优先级，而不是块顺序 ───────────────────────────────────────

def test_consolidated_wins_over_parent_even_when_later():
    """母公司表在池子里排前面，也要选合并表那一行。"""
    meta = doc("600519", "2023", "600519_2023_贵州茅台_2023年年度报告.json", [
        item(section=SEC_FS, content="母公司资产负债表",
             table_json=summary_table("总资产", "80,528,835,543.17")),
        item(section=SEC_FS, content="合并资产负债表",
             table_json=summary_table("总资产", "272,699,660,092.25")),
    ])
    blob = pipe_of(meta)._table_context_items("贵州茅台2023年的总资产是多少")[0]["content"]
    assert "272,699,660,092.25" in blob and "80,528,835,543.17" not in blob


def test_notes_row_does_not_override_consolidated():
    meta = doc("000858", "2021", "000858_2021_五粮液_2021年年度报告.json", [
        item(section=SEC_NOTES, content="重要会计政策及会计估计",
             table_json=summary_table("总资产", "113,893,139,013.72")),
        item(section=SEC_FS, content="合并资产负债表",
             table_json=summary_table("总资产", "135,620,812,221.13")),
    ])
    blob = pipe_of(meta)._table_context_items("五粮液2021年的总资产是多少")[0]["content"]
    assert "135,620,812,221.13" in blob and "113,893,139,013.72" not in blob


def test_summary_and_consolidated_same_tier_keeps_pool_order():
    """摘要表与合并报表同档，同分保持池子原顺序 —— 这是 numeric 现状的保护锁。"""
    meta = doc("600519", "2023", "600519_2023_贵州茅台_2023年年度报告.json", [
        item(section=SEC_SUMMARY, content="主要会计数据和财务指标",
             table_json=summary_table("总资产", "272,699,660,092.25")),
        item(section=SEC_FS, content="合并资产负债表",
             table_json=summary_table("总资产", "272,700,000,000.00")),
    ])
    blob = pipe_of(meta)._table_context_items("贵州茅台2023年的总资产是多少")[0]["content"]
    assert "272,699,660,092.25" in blob, "摘要被降档 = 会改掉一批本来答对的 numeric 题"


def test_table_source_rank_classes():
    assert rp.table_source_rank({"section": SEC_FS, "content": "合并资产负债表"}) == 0
    assert rp.table_source_rank({"section": SEC_SUMMARY, "content": "主要会计数据"}) == 0
    assert rp.table_source_rank({"section": SEC_FS, "content": "母公司资产负债表"}) == 1
    assert rp.table_source_rank({"section": SEC_NOTES, "content": "会计政策"}) == 2
    assert rp.table_source_rank({"section": "环境问题", "content": "排污信息"}) == 3


# ── 5：单年题与修复前逐字一致 ─────────────────────────────────────────────────

def test_single_year_output_is_verbatim_unchanged():
    meta = doc("600519", "2023", "600519_2023_贵州茅台_2023年年度报告.json", [
        item(chunk_id="k1", page_num=10, section=SEC_SUMMARY, content="主要会计数据",
             table_json=summary_table("总资产", "272,699,660,092.25")),
    ])
    items = pipe_of(meta)._table_context_items("贵州茅台2023年的总资产是多少")
    assert items[0]["content"] == (
        "【表格精确取数】600519 2023年报 · " + SEC_SUMMARY + "\n"
        "总资产：2023年: 272,699,660,092.25；2022年: 1,000.00"), items[0]["content"]
    assert items[0]["chunk_id"] == "k1" and items[0]["page_num"] == 10


# ── 6：保留的硬闸与已知边界 ───────────────────────────────────────────────────

def test_question_without_any_year_still_bails():
    """问题里一个年份都没有时不补位：连取哪年都不确定。"""
    meta = _three_year_meta()
    assert pipe_of(meta)._table_context_items("贵州茅台的总资产是多少") == []


def test_no_company_clue_still_refuses_to_fetch():
    meta = _three_year_meta()
    assert pipe_of(meta)._table_context_items("2021到2023年总资产各年是多少") == []
