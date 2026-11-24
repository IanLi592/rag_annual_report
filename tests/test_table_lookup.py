"""表格精确取数守卫（P1-2）。

`table_json` 的价值不在于"存了"，而在于检索命中表格后能直接读原始行，
不让 LLM 从 Markdown 里重新数格子。这里钉住匹配规则：
  - 只匹配首列（行标签），防止数值恰好等于标签造成误命中
  - 归一化空白，兼容 "其中：营业收入" / "营业收入 (元)"
  - 无 table_json / 空标签 / 无命中都返回空列表，不抛异常
"""
from __future__ import annotations

import rag_pipeline as rp


def item(**kw):
    base = {"chunk_id": "c1", "stock_code": "600519", "year": "2023",
            "page_num": 63, "section": "财务报告 > 合并利润表"}
    base.update(kw)
    return base


TABLE = [
    ["项目", "2023年", "2022年"],
    ["其中：营业收入", "147,693,604,994.14", "124,099,843,771.99"],
    ["营业成本", "22,473,402,516.29", "20,322,664,470.85"],
]


def test_exact_row_match_returns_full_row():
    rows = rp.lookup_table_value([item(table_json=TABLE)], "营业收入")
    assert len(rows) == 1
    assert rows[0]["row"] == ["其中：营业收入", "147,693,604,994.14", "124,099,843,771.99"]
    assert rows[0]["chunk_id"] == "c1" and rows[0]["page_num"] == 63


def test_label_with_whitespace_is_normalized():
    rows = rp.lookup_table_value([item(table_json=TABLE)], " 营业 收入 ")
    assert len(rows) == 1


def test_only_first_column_is_matched():
    """数值列里出现同名字符串不应命中。"""
    table = [["项目", "说明"], ["营收", "营业收入相关"]]
    rows = rp.lookup_table_value([item(table_json=table)], "营业收入相关")
    assert rows == [], "匹配到了非首列，说明取数会串行"


def test_multiple_tables_and_rows_are_all_returned():
    rows = rp.lookup_table_value(
        [item(chunk_id="a", table_json=TABLE),
         item(chunk_id="b", table_json=[["营业收入", "100"]])],
        "营业收入")
    assert [r["chunk_id"] for r in rows] == ["a", "b"]


def test_missing_table_json_is_empty():
    assert rp.lookup_table_value([item()], "营业收入") == []
    assert rp.lookup_table_value([], "营业收入") == []


def test_blank_label_is_empty():
    assert rp.lookup_table_value([item(table_json=TABLE)], "   ") == []


def test_ragged_and_none_cells_do_not_crash():
    table = [["营业收入", None, "100"], [], ["营业成本"]]
    rows = rp.lookup_table_value([item(table_json=table)], "营业收入")
    assert rows[0]["row"] == ["营业收入", "", "100"]


def test_label_longer_than_cell_still_matches():
    """问题里常写全称，如"营业收入（元）"，单元格只写"营业收入"。"""
    rows = rp.lookup_table_value([item(table_json=TABLE)], "营业收入（元）")
    assert len(rows) == 1


def test_long_paragraph_cell_is_not_a_row_label():
    """实测有整段审计说明的首列包含"营业收入"，不能被当成数据行。"""
    paragraph = "我们对营业收入执行了审计程序，包括检查合同与发票，评价收入确认政策"
    table = [["项目", "说明"], [paragraph, "1"], ["营业收入", "100"]]
    rows = rp.lookup_table_value([item(table_json=table)], "营业收入")
    assert [r["row"][0] for r in rows] == ["营业收入"]


def test_cross_column_heading_is_not_a_row():
    """跨列标题只有首列有字（"（一）营业收入的确认"），不是数据行。
    实测 /table/lookup 在 structured 索引上把它也返回了。"""
    table = [["项目", "本期金额", "上期金额"],
             ["（一）营业收入的确认", "", ""],
             ["营业收入", "100", "90"]]
    rows = rp.lookup_table_value([item(table_json=table)], "营业收入")
    assert [r["row"][0] for r in rows] == ["营业收入"]


def test_fallback_pool_is_used_when_candidates_miss():
    """向量检索 Top-10 里没有报表页时，按行标签扫收窄后的兜底池。
    实测“贵州茅台2023年利润表营业收入”Top-10 全是审计意见等叙述页。"""
    fallback = [item(chunk_id="income", table_json=TABLE)]
    rows = rp.lookup_table_value([], "营业收入", fallback_pool=fallback)
    assert [r["chunk_id"] for r in rows] == ["income"]


def test_fallback_pool_is_ignored_when_candidates_hit():
    """候选已命中就不该再扫全池，否则会混入同公司其它表的同名行。"""
    candidates = [item(chunk_id="hit", table_json=TABLE)]
    fallback = [item(chunk_id="other", table_json=TABLE)]
    rows = rp.lookup_table_value(candidates, "营业收入", fallback_pool=fallback)
    assert [r["chunk_id"] for r in rows] == ["hit"]


def test_fallback_pool_rows_are_deduplicated():
    table = [["营业收入", "100"], ["营业收入", "100"]]
    rows = rp.lookup_table_value(
        [], "营业收入", fallback_pool=[item(chunk_id="a", table_json=table)])
    assert len(rows) == 1, "同一 chunk 的重复行只应返回一次"


def _meta(stock, year, source, table=TABLE):
    return item(stock_code=stock, year=year, source_file=source,
                chunk_id=f"{stock}_{year}", table_json=table)


def test_infer_filter_from_company_name_and_year():
    meta = [_meta("600519", "2023", "600519_2023_贵州茅台_2023年年度报告.json"),
            _meta("000858", "2023", "000858_2023_五粮液_2023年年度报告.json")]
    filt = rp.infer_filter_from_question("贵州茅台2023年利润表营业收入", meta)
    assert filt == {"stock_code": "600519", "year": "2023"}


def test_infer_filter_from_stock_code():
    meta = [_meta("600519", "2023", "600519_2023_贵州茅台_2023年年度报告.json")]
    filt = rp.infer_filter_from_question("600519 2023 营业收入", meta)
    assert filt["stock_code"] == "600519" and filt["year"] == "2023"


def test_infer_filter_skips_ambiguous_years():
    """跨年对比题不锁年份，否则只能查到一半。"""
    meta = [_meta("600519", "2023", "600519_2023_贵州茅台_2023年年度报告.json")]
    filt = rp.infer_filter_from_question("贵州茅台2021到2023营收趋势", meta)
    assert filt == {"stock_code": "600519"}


def test_infer_filter_empty_when_nothing_known():
    assert rp.infer_filter_from_question("营业收入是多少", []) == {}


def test_infer_filter_skips_ambiguous_companies():
    """同时提到两家公司时不锁 stock_code，否则会拿 A 公司的数答 B 公司。"""
    meta = [_meta("600519", "2023", "600519_2023_贵州茅台_2023年年度报告.json"),
            _meta("000858", "2023", "000858_2023_五粮液_2023年年度报告.json")]
    filt = rp.infer_filter_from_question("贵州茅台和五粮液2023年营业收入", meta)
    assert "stock_code" not in filt and filt["year"] == "2023"


def _pipe(meta):
    pipe = object.__new__(rp.RAGPipeline)
    pipe.vec_store = type("V", (), {"meta_list": meta})()
    return pipe


def test_query_table_context_extracts_named_row():
    """自然语言问题里点名的指标，应补成一条可引用的上下文。"""
    table = [["项目", "2023年"], ["总资产", "1,000"], ["资产", "2,000"]]
    meta = [_meta("600519", "2023", "600519_2023_贵州茅台_2023年年度报告.json",
                  table=table)]
    items = _pipe(meta)._table_context_items("贵州茅台2023年的总资产是多少")
    assert len(items) == 1
    assert "总资产" in items[0]["content"]
    assert "1,000" in items[0]["content"]
    assert "2,000" not in items[0]["content"], "不应退到子串“资产”"


def test_query_table_context_requires_company():
    """没有公司线索时不取数，避免跨公司串数。"""
    meta = [_meta("600519", "2023", "600519_2023_贵州茅台_2023年年度报告.json")]
    assert _pipe(meta)._table_context_items("2023年的总资产是多少") == []


def test_query_table_context_empty_when_no_label_in_question():
    meta = [_meta("600519", "2023", "600519_2023_贵州茅台_2023年年度报告.json")]
    assert _pipe(meta)._table_context_items("贵州茅台2023年的经营情况如何") == []


def test_query_table_context_maps_metric_synonym():
    """2022 年报首列写“资产总计”，问题问“总资产”，应能对上。"""
    table = [["项目", "2022年"], ["资产总计", "152,714,727,880.22"]]
    meta = [_meta("000858", "2022", "000858_2022_五粮液_2022年年度报告.json",
                  table=table)]
    items = _pipe(meta)._table_context_items("五粮液2022年的总资产是多少？")
    assert len(items) == 1
    assert "资产总计" in items[0]["content"]
    assert "152,714,727,880.22" in items[0]["content"]


def test_query_table_context_skips_rows_without_level_value():
    """只有占比/增减列时不补位：空内容会挤掉原本有效的上下文。"""
    table = [["项目", "占营业收入比例"], ["研发费用", "5.2%"]]
    meta = [_meta("600519", "2023", "600519_2023_贵州茅台_2023年年度报告.json",
                  table=table)]
    assert _pipe(meta)._table_context_items("贵州茅台2023年研发费用占比是多少") == []


def test_chinese_ordinal_prefix_is_normalized():
    """利润表行常写成“一、营业收入”，问题只写“营业收入”也应能取到。"""
    table = [["项目", "2023年"], ["一、营业收入", "100"]]
    rows = rp.lookup_table_value([item(table_json=table)], "营业收入")
    assert len(rows) == 1
    assert rows[0]["row"][0] == "一、营业收入"
