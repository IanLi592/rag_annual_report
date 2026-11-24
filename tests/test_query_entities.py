"""查询实体归一守卫。

数值题大量写成「000858 在 2021 年的营业收入是多少」，而 chunk 正文只有
「五粮液」没有股票代码。检索前把代码补成「公司简称+代码」，否则 Top-4
经常落到别的公司（实测 semantic Hit@4 0.500→1.000）。
"""
from __future__ import annotations

import rag_pipeline as rp


def meta(code: str, source: str) -> dict:
    return {"stock_code": code, "source_file": source}


META = [
    meta("000858", "000858_2021_五粮液_2021年年度报告.json"),
    meta("600519", "600519_2023_贵州茅台_2023年年度报告.json"),
]


def test_stock_code_is_expanded_with_company_name():
    q = rp.expand_query_entities("000858在2021年的营业收入是多少？", META)
    assert "五粮液" in q and "000858" in q


def test_company_name_already_present_is_not_duplicated():
    q = rp.expand_query_entities("五粮液000858在2021年的营业收入", META)
    assert q.count("五粮液") == 1


def test_unknown_code_is_left_untouched():
    q = rp.expand_query_entities("999999在2021年的营业收入", META)
    assert q == "999999在2021年的营业收入"


def test_multiple_codes_are_all_expanded():
    q = rp.expand_query_entities("000858和600519的营业收入对比", META)
    assert "五粮液" in q and "贵州茅台" in q


def test_no_meta_is_noop():
    q = rp.expand_query_entities("000858在2021年的营业收入", [])
    assert q == "000858在2021年的营业收入"


def test_company_name_map_skips_malformed_source_file():
    names = rp.company_name_map([
        {"stock_code": "000858", "source_file": "bad.json"},
        {"stock_code": "600519", "source_file": "600519_2023_贵州茅台_2023年年度报告.json"},
    ])
    assert names == {"600519": "贵州茅台"}
