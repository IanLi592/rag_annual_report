"""领域词表机制的守卫（P0-2）。

注意分工：这里只守"机制正确接线"，**不守"指标一定变好"** ——
后者是实测问题，实测结论见 docs/resume_gap_plan.md §7：17 题上方向不一致，
差值都只有 1 题的量级，不足以支撑任何百分比主张。
"""
from __future__ import annotations

import ast
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

import jieba_domain as jd


@pytest.fixture(autouse=True)
def reset(monkeypatch):
    monkeypatch.delenv("JIEBA_DOMAIN_DICT", raising=False)
    monkeypatch.delenv("DOMAIN_DICT_PATH", raising=False)
    jd.reset_for_test()
    yield
    jd.reset_for_test()


def test_default_dict_exists_and_is_substantial():
    words = jd.dict_path().read_text(encoding="utf-8").splitlines()
    terms = [w.strip() for w in words if w.strip() and not w.strip().startswith("#")]
    assert jd.dict_path().exists()
    assert len(terms) >= 100, f"词表太小，覆盖不住财报术语: {len(terms)}"
    assert len(terms) == len(set(terms)), "词表有重复词条"
    assert all(len(t) >= 2 for t in terms), "单词条目会污染分词"
    # 简历点名的两个术语必须在表里
    for must in ("扣非净利润", "商誉减值"):
        assert must in terms


def test_loader_is_idempotent_and_reports_counts():
    import jieba
    a = jd.load_domain_dict(jieba)
    b = jd.load_domain_dict(jieba)
    assert a["loaded"] == b["loaded"] and a["loaded"] >= 100
    assert jd.is_loaded() is True


def test_switch_off_skips_loading(monkeypatch):
    monkeypatch.setenv("JIEBA_DOMAIN_DICT", "0")
    import jieba
    st = jd.load_domain_dict(jieba)
    assert st["enabled"] is False and st["loaded"] == 0


def test_path_override_is_honoured(monkeypatch, tmp_path):
    d = tmp_path / "mini.txt"
    d.write_text("测试领域词\n另一个词\n", encoding="utf-8")
    monkeypatch.setenv("DOMAIN_DICT_PATH", str(d))
    import jieba
    st = jd.load_domain_dict(jieba)
    assert st["loaded"] == 2 and st["path"] == str(d)


def test_missing_or_empty_dict_fails_loudly(monkeypatch, tmp_path):
    import jieba
    monkeypatch.setenv("DOMAIN_DICT_PATH", str(tmp_path / "nope.txt"))
    with pytest.raises(FileNotFoundError):
        jd.load_domain_dict(jieba)
    empty = tmp_path / "empty.txt"
    empty.write_text("# 只有注释\n\n", encoding="utf-8")
    monkeypatch.setenv("DOMAIN_DICT_PATH", str(empty))
    with pytest.raises(ValueError):
        jd.load_domain_dict(jieba)


def test_long_term_actually_fixes_chopped_terms(tmp_path):
    """**必须跨进程比** —— jieba 的前缀词典是进程级全局的，同进程里先加载再想看
    "before" 是看不到的（我第一版就栽在这，测试假失败）。
    这里用两个子进程分别跑关表/开表，比同一句话的切分结果。"""
    probe = tmp_path / "probe.py"
    probe.write_text(
        "import sys, json\n"
        "sys.path.insert(0, sys.argv[1])\n"
        "import jieba\n"
        "from jieba_domain import load_domain_dict\n"
        "load_domain_dict(jieba)\n"
        "print(json.dumps([t for t in jieba.cut('扣非净利润和商誉减值和经营活动产生的现金流量净额') if t.strip()]))\n",
        encoding="utf-8",
    )
    src_dir = str(Path(jd.__file__).parent)
    outs = {}
    for flag in ("0", "1"):
        r = subprocess.run([sys.executable, str(probe), src_dir],
                           capture_output=True, text=True, encoding="utf-8",
                           env=dict(os.environ, JIEBA_DOMAIN_DICT=flag))
        assert r.returncode == 0, r.stderr[-300:]
        outs[flag] = json.loads(r.stdout.strip().splitlines()[-1])

    assert "扣非净利润" not in outs["0"], \
        f"对照组里术语居然是整词，测试前提失效: {outs['0']}"
    assert "扣非净利润" in outs["1"], f"开表后仍被切碎: {outs['1']}"
    assert "经营活动产生的现金流量净额" in outs["1"], f"长术语没被修复: {outs['1']}"
    assert len(outs["1"]) < len(outs["0"]), f"切分粒度没变粗: off={outs['0']} on={outs['1']}"


# ── 接线守卫：能力存在但没被调用，等于没做（BM25 过滤那次就是这么翻车的）──

def _find_func(tree, func: str, cls: str | None = None):
    """按（类名, 函数名）精确定位。文件里 __init__ 有好几个，只按函数名抓会抓错。"""
    import ast
    scopes = [tree]
    if cls:
        node = next((n for n in ast.walk(tree)
                     if isinstance(n, ast.ClassDef) and n.name == cls), None)
        assert node, f"找不到类 {cls}"
        scopes = [node]
    for s in scopes:
        for n in ast.walk(s):
            if isinstance(n, ast.FunctionDef) and n.name == func:
                return n
    raise AssertionError(f"找不到函数 {cls or '<module>'}.{func}")


def _loads_dict_before_cutting(fn) -> tuple[bool, str]:
    """load_domain_dict 的调用必须排在第一次 jieba.cut 之前。"""
    import ast
    order = []
    for node in ast.walk(fn):
        if isinstance(node, ast.Call):
            name = node.func.attr if isinstance(node.func, ast.Attribute) else getattr(node.func, "id", "")
            if name in ("load_domain_dict", "cut"):
                order.append((getattr(node, "lineno", 0), name))
    if "load_domain_dict" not in [n for _, n in order]:
        return False, "压根没调用 load_domain_dict"
    first_load = min(l for l, n in order if n == "load_domain_dict")
    cuts = [l for l, n in order if n == "cut"]
    if cuts and min(cuts) < first_load:
        return False, f"cut(第{min(cuts)}行) 早于 load_domain_dict(第{first_load}行)，词典不会生效"
    return True, "ok"


def test_online_bm25_path_loads_dict_before_tokenizing():
    src = Path(jd.__file__).parent / "rag_pipeline.py"
    import ast
    tree = ast.parse(src.read_text(encoding="utf-8"))
    ok, why = _loads_dict_before_cutting(_find_func(tree, "__init__", cls="BM25Store"))
    assert ok, f"BM25Store.__init__: {why}"


def test_ablation_bm25_path_loads_dict_before_tokenizing():
    """消融链路与线上必须同源，否则测出来的收益是另一套分词口径下的。"""
    src = Path(jd.__file__).parents[1] / "evaluation" / "compare_strategies.py"
    import ast
    tree = ast.parse(src.read_text(encoding="utf-8"))
    ok, why = _loads_dict_before_cutting(_find_func(tree, "retrieve_bm25"))
    assert ok, f"retrieve_bm25: {why}"
