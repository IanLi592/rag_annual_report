"""页眉/页脚噪声过滤守卫。

页码一旦被当成标题，就会进入 `section_path` 并跟着 chunk 走，最终出现在引用
来源里（实测 structured 索引出现 `... > 财务报告 > 62 / 143`）。这里钉住
常见页脚形式的过滤，避免它污染章节树。
"""
from __future__ import annotations

import parse_pdf as pp


def test_standalone_page_number_is_noise():
    assert pp.is_noise_line("38") is True


def test_slash_page_footer_is_noise():
    assert pp.is_noise_line("62 / 143") is True
    assert pp.is_noise_line("62/143") is True


def test_dash_page_footer_is_noise():
    assert pp.is_noise_line("— 38 —") is True


def test_annual_report_header_is_noise():
    assert pp.is_noise_line("贵州茅台2023年年度报告") is True


def test_real_section_heading_is_not_noise():
    assert pp.is_noise_line("第三章 管理层讨论与分析") is False
    assert pp.is_noise_line("一、营业收入") is False
    assert pp.is_noise_line("2. 茅台金石（贵州）产业发展基金") is False


# ── 页脚尾缀：公司名 + 年份 + 年度报告全文 ────────────────────────────────────
# 实测（11-10 全量重解析产物 39,499 块）这类行共 894 条，形态只有 `全文` 一种尾缀：
#   宁德时代新能源科技股份有限公司2023 年年度报告全文  230
#   宜宾五粮液股份有限公司2021 年度报告全文            155
# 旧 NOISE_PATTERNS 的第一条要求以“年度报告”结尾，尾缀“全文”一来就漏，
# 于是它作为正文落进 chunk：语义索引里 383/7931 个 chunk 正文含该行、
# 其中 314 个是 chunk 首行（embedding 与 BM25 的输入就是 content）。

def test_footer_with_quanwen_suffix_is_noise():
    assert pp.is_noise_line("宜宾五粮液股份有限公司2021 年度报告全文") is True
    assert pp.is_noise_line("宁德时代新能源科技股份有限公司2022 年年度报告全文") is True


# 尾缀集合必须钉死为“全文”：语料里另有以 正文/原件/摘要 收尾的真实句子，
# 放宽就等于删正文。
def test_other_trailing_words_are_not_footer():
    assert pp.is_noise_line("载有本公司负责人签字的年度报告正文") is False
    assert pp.is_noise_line("一、载有公司法定代表人签字的2021 年年度报告原件") is False
    assert pp.is_noise_line("贵州茅台2023年年度报告摘要") is False


def test_prose_mentioning_annual_report_is_not_noise():
    assert pp.is_noise_line("四、公司年度报告。") is False
    assert pp.is_noise_line("审查关联交易相关年度报告；") is False
    assert pp.is_noise_line("九、年度报告披露后面临退市情况") is False
    assert pp.is_noise_line("| 本报告 | 指 | 2021年年度报告 |") is False
