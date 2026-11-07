"""章节栈守卫：目录/封面噪声不能进入 section_path，第X节要能当一级标题。

背景：实测修复前主索引 63.5% 的 chunk 章节根是“目录/公司名/封面色”，
structured 的 parents.json 有 57% 同样被污染。根因是 `_update_section()`
在“第X节”到来时保留栈顶，以及封面/目录标题被当成 H1/H3。
"""
from __future__ import annotations

import parse_pdf as pp


def parser():
    """只测章节栈，不读 PDF、不初始化客户端。"""
    p = object.__new__(pp.AnnualReportParser)
    p._section_stack = []
    p._section_levels = []
    return p


# ── 噪声标题识别 ──────────────────────────────────────────────────────────────

def test_toc_and_cover_titles_are_not_sections():
    assert pp.is_non_section_title("目录")
    assert pp.is_non_section_title("目 录")
    assert pp.is_non_section_title("二〇二二年四月十六日")
    assert pp.is_non_section_title("贵州茅台酒股份有限公司")
    assert pp.is_non_section_title("第一节 重要提示和释义 ................................ 3")


def test_real_headings_are_not_filtered():
    # 合法章节名里也可能带“目录”二字，不能一刀切
    assert not pp.is_non_section_title("第一节 重要提示、目录和释义")
    assert not pp.is_non_section_title("合并利润表")
    assert not pp.is_non_section_title("财务报表附注")
    assert not pp.is_non_section_title("一、营业收入")


def test_toc_page_detection():
    assert pp.is_toc_page(["第一节 重要提示 ...... 3", "第二节 公司简介 ...... 5"])
    assert not pp.is_toc_page(["第一节 重要提示 ...... 3", "正文段落"])


# ── 章节栈层级 ────────────────────────────────────────────────────────────────

def test_section_is_h1_when_no_chapter_exists():
    """很多年报直接用“第X节”做一级，没有“第X章”。"""
    p = parser()
    p._update_section("第三节 管理层讨论与分析")
    assert p._section_stack == ["第三节 管理层讨论与分析"]
    assert p._section_levels == [1]


def test_section_is_h2_under_chapter():
    p = parser()
    p._update_section("第一章 总则")
    p._update_section("第三节 管理层讨论与分析")
    assert p._section_stack == ["第一章 总则", "第三节 管理层讨论与分析"]
    assert p._section_levels == [1, 2]


def test_sibling_sections_replace_instead_of_nesting():
    p = parser()
    p._update_section("第三节 管理层讨论与分析")
    p._update_section("第四节 公司治理")
    assert p._section_stack == ["第四节 公司治理"]


def test_noise_does_not_change_stack():
    p = parser()
    p._update_section("第三节 管理层讨论与分析")
    before = list(p._section_stack)
    for noise in ("目录", "二〇二二年四月十六日", "贵州茅台酒股份有限公司",
                  "第一节 重要提示和释义 ................................ 3"):
        p._update_section(noise)
    assert p._section_stack == before


def test_noise_title_has_level_zero():
    """目录/封面标题不推进章节栈，也不该被记成 H1/H2/H3。"""
    p = parser()
    assert p._section_level("目录") == 0
    assert p._section_level("二〇二二年四月十六日") == 0
    assert p._section_level("贵州茅台酒股份有限公司") == 0


def test_h3_stays_under_section():
    p = parser()
    p._update_section("第三节 管理层讨论与分析")
    p._update_section("一、经营情况讨论与分析")
    assert p._section_stack == ["第三节 管理层讨论与分析", "一、经营情况讨论与分析"]
    assert p._section_levels == [1, 3]
