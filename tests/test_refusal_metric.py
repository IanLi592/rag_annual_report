"""拒答判定守卫：区分"真拒答"与"检索为空"。

这条口径直接决定简历里的拒答率。旧实现只看关键词，n_ctx=0 的检索失败
也被算成模型拒答，等于把检索质量问题记在了生成头上。
"""
from __future__ import annotations

import pytest

from refusal import classify_refusal, is_refusal


@pytest.mark.parametrize("answer", [
    "根据提供的资料，无法回答该问题。",
    "参考资料中没有披露2027年的业绩指引。",
    "现有资料不足以确定该数据。",
    "该问题超出年报资料范围。",
])
def test_explicit_refusal_with_context(answer):
    assert classify_refusal(answer, n_ctx=3) == "refused_with_context"
    assert is_refusal(answer, n_ctx=3) is True


def test_empty_context_is_not_model_refusal():
    """检索为空时，哪怕答案是"未找到相关内容"，也不能算模型拒答。"""
    assert classify_refusal("未找到相关内容。", n_ctx=0) == "empty_context"
    assert is_refusal("未找到相关内容。", n_ctx=0) is False


def test_empty_context_wins_over_refusal_pattern():
    assert classify_refusal("无法回答", n_ctx=0) == "empty_context"


def test_normal_answer_is_not_refusal():
    answer = "2023年营业收入为1505.60亿元，主要来自茅台酒销售[1]。"
    assert classify_refusal(answer, n_ctx=4) == "answered"


def test_bare_negation_is_not_a_refusal():
    """"没有"单独出现不算拒答——正常答案里也有（如"没有重大变化"）。"""
    assert classify_refusal("报告期内没有重大资产重组。", n_ctx=2) == "answered"


def test_none_answer_does_not_crash():
    assert classify_refusal(None, n_ctx=1) == "answered"
    assert classify_refusal(None, n_ctx=0) == "empty_context"
