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
