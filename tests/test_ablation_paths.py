"""消融脚本的索引选择守卫。

背景（实测抓到）：旧版 load_index 对任何缺失策略都回退到 semantic 主索引，
于是消融表里会出现标着 "fixed/hierarchical" 但实际跑 semantic 数据的行 ——
那种假结论不会报错，只会安静地污染对比结论。
"""
from __future__ import annotations

import logging

import compare_strategies as cs
import pytest


@pytest.fixture(autouse=True)
def quiet():
    logging.disable(logging.CRITICAL)
    yield
    logging.disable(logging.NOTSET)


def test_semantic_falls_back_to_main_index():
    """semantic 就是主索引构建用的策略，回退是诚实的。"""
    assert cs.STRATEGY_WHEN_FLAT == "semantic"
    index, meta = cs.load_index("semantic")
    assert index.ntotal == len(meta)


@pytest.mark.parametrize("strategy", ["fixed", "hierarchical"])
def test_other_strategies_refuse_to_impersonate_semantic(strategy):
    with pytest.raises(FileNotFoundError) as e:
        cs.load_index(strategy)
    msg = str(e.value)
    assert strategy in msg
    assert "semantic" in msg, "报错必须说清为什么不能回退，否则使用者只会去乱改路径"
