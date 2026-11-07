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
