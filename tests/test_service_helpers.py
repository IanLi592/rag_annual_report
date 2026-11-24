"""serve.py 三个工具函数的回归守卫（P1-1）。

期望值全部来自本会话对现有实现的实测快照，不是照着代码猜的：
  _build_source({'stock_code':'600519','year':'2021'})          -> '600519 2021年报'
  _build_source({...,'section':'A > B > C','page_num':12})       -> '600519 2021年报 · B > C · 第12页'
  _build_source({...,'page_num':-1}) / _build_source({...,'page_num':0}) -> 都省掉页码
  _preview('a'*150,150) -> 150 字符无省略号；('a'*151,150) -> 151 字符含省略号
"""
from __future__ import annotations

import serve


# ── _build_source ────────────────────────────────────────────────────────────

def test_build_source_minimal_form():
    assert serve._build_source({"stock_code": "600519", "year": "2021"}) == "600519 2021年报"


def test_build_source_keeps_only_last_two_section_levels():
    """三级以上章节只留最后两级，前端才不会溢出（serve.py:141）。"""
    item = {"stock_code": "600519", "year": "2021", "section": "A > B > C", "page_num": 12}
    assert serve._build_source(item) == "600519 2021年报 · B > C · 第12页"


def test_build_source_two_level_section_is_not_truncated():
    item = {"stock_code": "6", "year": "1", "section": "X > Y", "page_num": 3}
    assert serve._build_source(item) == "6 1年报 · X > Y · 第3页"


def test_build_source_omits_sentinel_page_num():
    """page_num 缺失/-1 时不显示页码。源码用 `if page and page != -1`，
    所以 0 也会被当假值省掉 —— 这是既有行为，钉住它而不是偷偷改。"""
    base = {"stock_code": "6", "year": "1"}
    assert serve._build_source(dict(base)) == "6 1年报"
    assert serve._build_source(dict(base, page_num=-1)) == "6 1年报"
    assert serve._build_source(dict(base, page_num=0)) == "6 1年报"


# ── _preview ─────────────────────────────────────────────────────────────────

def test_preview_exactly_at_limit_has_no_ellipsis():
    out = serve._preview("a" * 150, 150)
    assert out == "a" * 150 and len(out) == 150


def test_preview_over_limit_truncates_and_marks():
    out = serve._preview("a" * 151, 150)
    assert out.endswith("…") and len(out) == 151  # 150 正文 + 1 省略号


def test_preview_strips_before_measuring():
    assert serve._preview("  hi  ") == "hi"
    assert serve._preview("   ") == ""


# ── _env_float ───────────────────────────────────────────────────────────────

def test_env_float_default_when_unset(monkeypatch):
    monkeypatch.delenv("RERANK_MIN_SCORE", raising=False)
    assert serve._env_float("RERANK_MIN_SCORE", None) is None
    assert serve._env_float("RERANK_MIN_SCORE", 0.75) == 0.75


def test_env_float_blank_is_treated_as_unset(monkeypatch):
    """docker-compose 里 `RERANK_MIN_SCORE=` 会传空串，不能当成 float("") 崩掉。"""
    monkeypatch.setenv("RERANK_MIN_SCORE", "  ")
    assert serve._env_float("RERANK_MIN_SCORE", None) is None


def test_env_float_parses_override(monkeypatch):
    monkeypatch.setenv("RERANK_MIN_SCORE", "0.82")
    assert serve._env_float("RERANK_MIN_SCORE", None) == 0.82


# ── _filter_meta ─────────────────────────────────────────────────────────────

def test_filter_meta_returns_none_when_no_filters():
    """必须是 None 而不是 {} —— VectorStore.search 用真值判断决定是否过滤，
    返回 {} 会让"无过滤"变成"过滤到空集"。"""
    assert serve._filter_meta(serve.QueryRequest(question="q")) is None


def test_filter_meta_picks_each_field_independently():
    assert serve._filter_meta(serve.QueryRequest(question="q", stock_code="600519")) == {"stock_code": "600519"}
    assert serve._filter_meta(serve.QueryRequest(question="q", year="2021")) == {"year": "2021"}
    both = serve._filter_meta(serve.QueryRequest(question="q", stock_code="600519", year="2021"))
    assert both == {"stock_code": "600519", "year": "2021"}
