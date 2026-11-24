"""拒答判定：把“模型按约束拒答”和“检索为空/正常回答”分开。

旧实现只在一段文本里找“无法回答”等关键词，导致两个问题：
  1. `n_ctx=0` 时系统返回“未找到相关内容”，也被算成拒答 —— 其实是检索失败；
  2. “没有”这类词在正常回答里也会出现，造成误报。

这里用更严格的句式匹配，并要求调用方传入上下文条数；分类结果只有三种：
  - refused_with_context：有上下文，但模型明确说资料不足/无法回答
  - empty_context：压根没检索到上下文（检索/过滤失败，不是模型拒答）
  - answered：模型给出了回答（可能对，也可能错，需要人工/LLM 判）
"""
from __future__ import annotations

import re

_REFUSAL_PATTERNS = [
    re.compile(r"(无法|不能|未能|没有办法).{0,10}(回答|提供|确定|获取|判断|给出|计算|找到)"),
    re.compile(r"(未|没有).{0,8}(披露|提及|包含|涉及|找到|提供).{0,12}(相关|该|此|具体|明确)?"),
    re.compile(r"根据(提供的)?(参考)?资料.{0,30}(无法|未|没有|不能|不足)"),
    re.compile(r"(资料|信息|内容).{0,12}(不足|中没有|未提及|未披露|不包含|不涉及)"),
    re.compile(r"(超出|不在).{0,10}(范围|资料|年报|知识库)"),
]


def classify_refusal(answer: str, n_ctx: int) -> str:
    """返回 refused_with_context / empty_context / answered。"""
    if n_ctx <= 0:
        return "empty_context"
    text = answer or ""
    if any(p.search(text) for p in _REFUSAL_PATTERNS):
        return "refused_with_context"
    return "answered"


def is_refusal(answer: str, n_ctx: int) -> bool:
    return classify_refusal(answer, n_ctx) == "refused_with_context"
