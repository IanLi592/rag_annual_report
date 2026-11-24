"""Qwen-VL 分支的离线守卫（不联网、不渲染真实 PDF）。

守四件事：
  1. 模型输出千奇百怪（代码围栏、前后废话、分段 content），解析不能整页失败
  2. 表格必须保留二维结构（raw_table），不能拍平成一段文字
  3. 标题层级必须能从年报编号规律推出来
  4. 默认关闭、失败回退：VLM 坏了不能拖垮整份年报
"""
from __future__ import annotations

import pytest

import parse_pdf as pp
import parse_pdf_vlm as vlm


# ── VlmConfig ────────────────────────────────────────────────────────────────

def test_vlm_is_off_by_default(monkeypatch):
    monkeypatch.delenv("VLM_MODE", raising=False)
    cfg = vlm.VlmConfig()
    assert cfg.mode == "off"
    assert cfg.enabled() is False, "默认必须关闭，否则一跑 parse_pdf 就产生不可控 token 费用"


def test_explicit_pages_enable_even_when_mode_off():
    cfg = vlm.VlmConfig(mode="off", pages={3, 8})
    assert cfg.enabled() is True


@pytest.mark.parametrize("bad", ["on", "true", ""])
def test_invalid_mode_rejected(bad):
    with pytest.raises(ValueError):
        vlm.VlmConfig(mode=bad)


def test_negative_budget_rejected():
    with pytest.raises(ValueError):
        vlm.VlmConfig(max_pages=-1)


# ── 响应解析 ─────────────────────────────────────────────────────────────────

def test_extract_json_array_strips_code_fence():
    raw = "```json\n[{\"block_type\":\"text\",\"content\":\"你好\"}]\n```"
    data = vlm._extract_json_array(raw)
    assert data[0]["content"] == "你好"


def test_extract_json_array_ignores_surrounding_prose():
    raw = "解析结果如下：\n[{\"content\":\"x\"}]\n以上。"
    assert vlm._extract_json_array(raw)[0]["content"] == "x"


def test_extract_json_array_joins_multipart_content():
    raw = [{"text": '[{"content":'}, {"text": '"a"}]'}]
    assert vlm._extract_json_array(raw)[0]["content"] == "a"


def test_extract_json_array_rejects_garbage():
    with pytest.raises(ValueError):
        vlm._extract_json_array("模型拒绝输出")
    with pytest.raises(ValueError):
        vlm._extract_json_array("")


def test_normalize_table_accepts_header_rows_dict():
    table = vlm._normalize_table({"header": ["项目", "金额"], "rows": [["营收", 100]]})
    assert table == [["项目", "金额"], ["营收", "100"]]


def test_normalize_table_accepts_2d_list_and_drops_empty_rows():
    table = vlm._normalize_table([["a", "b"], [None, None], ["c", "d"]])
    assert table == [["a", "b"], ["c", "d"]]


def test_parse_response_keeps_table_structure():
    raw = ('[{"block_type":"table","content":"营业收入表",'
           '"table":{"header":["项目","2023"],"rows":[["营收","100亿"]]}}]')
    blocks = vlm.parse_vlm_response(raw, page_num=12, section_path=["第三章"])
    b = blocks[0]
    assert b["block_type"] == "table"
    assert b["raw_table"] == [["项目", "2023"], ["营收", "100亿"]], "表格被拍平了"
    assert b["content"] == "营业收入表"
    assert b["parser"] == "vlm"
    assert b["page_num"] == 12 and b["section_path"] == ["第三章"]


def test_parse_response_synthesizes_markdown_when_summary_missing():
    raw = '[{"block_type":"table","table":{"header":["a","b"],"rows":[["1","2"]]}}]'
    b = vlm.parse_vlm_response(raw, 1, [])[0]
    assert "| a | b |" in b["content"] and "| 1 | 2 |" in b["content"]


def test_parse_response_skips_invalid_entries_but_keeps_page():
    raw = ('["不是对象", {"block_type":"text","content":""},'
           '{"block_type":"table"}, {"block_type":"text","content":"保留"}]')
    blocks = vlm.parse_vlm_response(raw, 3, [])
    assert [b["content"] for b in blocks] == ["保留"]


def test_parse_response_normalizes_unknown_block_type():
    b = vlm.parse_vlm_response('[{"block_type":"footnote","content":"x"}]', 1, [])[0]
    assert b["block_type"] == "text"


@pytest.mark.parametrize("title,level", [
    ("第三章 管理层讨论与分析", 1),
    ("第二节 公司基本情况", 2),
    ("一、经营情况概述", 3),
    ("1. 主营业务分行业情况", 3),
    ("贵州茅台酒股份有限公司", 0),
])
def test_classify_heading_by_numbering(title, level):
    assert vlm._classify_heading(title) == level


def test_model_supplied_heading_level_wins():
    assert vlm._classify_heading("第三章", level=2) == 2


# ── 与 parse_pdf 的接线 ──────────────────────────────────────────────────────

def test_rule_parser_classify_heading_agrees():
    assert pp.classify_heading("第三章 管理层讨论") == 1
    assert pp.classify_heading("第二节 公司基本情况") == 2
    assert pp.classify_heading("一、经营情况") == 3


def test_section_stack_tracks_h1_h2_h3():
    p = pp.AnnualReportParser.__new__(pp.AnnualReportParser)   # 不碰 PDF
    p._section_stack, p._section_levels = [], []
    p._update_section("第三章 管理层讨论", 1)
    p._update_section("第二节 经营情况", 2)
    p._update_section("一、营业收入", 3)
    assert p._section_stack == ["第三章 管理层讨论", "第二节 经营情况", "一、营业收入"]
    assert p._section_levels == [1, 2, 3]
    # 新的 H2 要截断旧的 H2/H3
    p._update_section("第三节 财务情况", 2)
    assert p._section_stack == ["第三章 管理层讨论", "第三节 财务情况"]


def test_vlm_failure_falls_back_to_rule(monkeypatch, tmp_path):
    """没有 key 时 _try_vlm 必须返回 None，让 parse() 走规则解析，而不是抛异常。"""
    p = pp.AnnualReportParser.__new__(pp.AnnualReportParser)
    p.pdf_path = tmp_path / "600519_2023.pdf"
    p.vlm_config = vlm.VlmConfig(mode="all", cache_dir=tmp_path / "cache")
    p._vlm_pages_used = p._vlm_sample_seen = 0
    p._vlm_budget_warned = False
    p._section_stack = []

    monkeypatch.setattr(pp, "get_vlm_client", lambda: None)
    assert p._try_vlm(0, "正文" * 100, None, None) is None
    assert p._vlm_budget_warned is True, "缺 key 只应告警一次"


def test_vlm_budget_caps_paid_pages(monkeypatch, tmp_path):
    p = pp.AnnualReportParser.__new__(pp.AnnualReportParser)
    p.pdf_path = tmp_path / "600519_2023.pdf"
    p.vlm_config = vlm.VlmConfig(mode="all", max_pages=1, cache_dir=tmp_path / "cache")
    p._vlm_pages_used = 1
    p._vlm_sample_seen = 0
    p._vlm_budget_warned = False
    p._section_stack = []

    called = []
    monkeypatch.setattr(pp, "get_vlm_client", lambda: called.append(1))
    assert p._try_vlm(0, "正文" * 100, None, None) is None
    assert called == [], "超预算后不能再建客户端、不能再计费"


def test_vlm_mode_off_never_routes(monkeypatch, tmp_path):
    p = pp.AnnualReportParser.__new__(pp.AnnualReportParser)
    p.vlm_config = vlm.VlmConfig(mode="off")
    p._vlm_sample_seen = 0
    assert p._should_use_vlm(0, "x", None, None) is False
