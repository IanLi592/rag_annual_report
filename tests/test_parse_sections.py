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


# ── 两类漏网的噪声：页脚尾缀 + 带空格的落款日期 ──────────────────────────────
# 11-10 全量重解析后实测：旧过滤规则对它们双漏（is_noise_line 与
# is_non_section_title 同时返回 False），残留形态与数量：
#   · `宜宾五粮液股份有限公司2021 年度报告全文` 进栈 493 块，
#     形成 231 个 chunk 的 `第十节 财务报告 > 一、审计报告 > …全文`
#   · `2022 年4 月29 日` / `2021 年12 月31 日` 等带空格落款进栈 33 块
#     （旧 _DATE_TITLE_RE 不允许数字与“年月日”之间的空格）

def test_footer_title_with_suffix_is_not_a_section():
    assert pp.is_non_section_title("宜宾五粮液股份有限公司2021 年度报告全文")
    assert pp.is_non_section_title("宁德时代新能源科技股份有限公司2023 年年度报告全文")


def test_spaced_signing_date_is_not_a_section():
    assert pp.is_non_section_title("2022 年4 月29 日")
    assert pp.is_non_section_title("2021 年12 月31 日")
    assert pp.is_non_section_title("2023 年4 月29 日")


def push(p, title, fontsize=10.5, is_bold=True):
    """复刻 `_parse_page_rule` 的真实调用式（parse_pdf.py:449-451）。

    页脚/落款在 PDF 里是加粗的，`classify_heading(..., is_bold=True)` 会先给出 3，
    再由 `_section_level` 决定是否否决。只调 `_section_level(title)` 会拿到默认
    level=0 而“自然通过”，那是假绿，钉不住这条链路。
    """
    level = p._section_level(title, pp.classify_heading(title, fontsize, is_bold))
    p._update_section(title, level)
    return level


def test_bold_footer_and_spaced_date_do_not_change_stack():
    p = parser()
    assert push(p, "第十节 财务报告", is_bold=False) == 1
    assert push(p, "一、审计报告") == 3
    before = list(p._section_stack)
    assert push(p, "宜宾五粮液股份有限公司2021 年度报告全文") == 0
    assert push(p, "2022 年4 月29 日") == 0
    assert p._section_stack == before == ["第十节 财务报告", "一、审计报告"]


# 反向对照：合法的加粗标题不能被误杀，否则“修污染”就变成了“丢章节”。
def test_bold_real_headings_still_enter_stack():
    p = parser()
    assert push(p, "第三节 管理层讨论与分析") == 1
    assert push(p, "一、 报告期内公司所处的行业情况") == 3
    assert p._section_stack == ["第三节 管理层讨论与分析", "一、 报告期内公司所处的行业情况"]


def test_footer_and_spaced_date_are_noise_lines():
    assert pp.is_noise_line("宜宾五粮液股份有限公司2021 年度报告全文")
    assert pp.is_non_section_title("2021 年12 月31 日")


# 日期过滤必须保持整行锚定：语料里 `于2021年12月31日` `截至…止12个月` 这类
# 正文片段共 17 块，锚定一丢就会被整句误杀（正文丢失比章节脏更严重）。
def test_prose_date_fragments_are_not_filtered():
    assert not pp.is_non_section_title("于2021年12月31日")
    assert not pp.is_non_section_title("截至2021年12月31日止12个月")
    assert not pp.is_non_section_title("长25.3%。截至2022年12月31日，")
    assert not pp.is_non_section_title("第十节 财务报告")
    assert not pp.is_non_section_title("一、审计报告")


def test_h3_stays_under_section():
    p = parser()
    p._update_section("第三节 管理层讨论与分析")
    p._update_section("一、经营情况讨论与分析")
    assert p._section_stack == ["第三节 管理层讨论与分析", "一、经营情况讨论与分析"]
    assert p._section_levels == [1, 3]
