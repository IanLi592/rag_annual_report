"""策略 A/B/C 分块的回归守卫（docs/cicd_requirements.md P1-1）。

全部零密钥、零网络、零磁盘产物 —— 喂内存里构造的 blocks 即可。

注意：chunk_documents.py:37 在 **import 期** 就执行
`CHUNKS_DIR.mkdir(parents=True, exist_ok=True)`，所以 import 会建目录。
这是既有行为，本文件不改它，只记录。
"""
from __future__ import annotations

import pytest

import chunk_documents as cd


def blk(btype: str, content: str, page: int = 1, path=None, ocr: bool = False) -> dict:
    """构造 parse_pdf 产出的 block 形状（字段名以 chunk_semantic 实际读取的为准）。"""
    return {
        "content": content,
        "block_type": btype,
        "page_num": page,
        "section_path": path if path is not None else ["第一节", "重要提示"],
        "is_ocr": ocr,
    }


# ── 策略 A：固定大小 ──────────────────────────────────────────────────────────

def test_chunk_fixed_step_is_size_minus_overlap():
    """size=30/overlap=10 的起点是 0,20,40,60,80 → 5 片（实测基线）。"""
    chunks = list(cd.chunk_fixed("x" * 100, chunk_size=30, overlap=10))
    assert len(chunks) == 5
    assert [c[:1] for c in chunks] == ["x"] * 5
    assert all(len(c) <= 30 for c in chunks)
    assert chunks[1] == "x" * 30  # 中间片是满长度


def test_chunk_fixed_no_overlap_is_disjoint():
    """overlap=0 时两片必须互不重叠且拼回原文（实测 100 字 → 2 片）。"""
    text = "y" * 100
    chunks = list(cd.chunk_fixed(text, chunk_size=50, overlap=0))
    assert "".join(chunks) == text


def test_chunk_fixed_overlap_repeats_content():
    """相邻块必须真的有 overlap 个字符重叠，否则 overlap 参数形同虚设。"""
    text = "".join(chr(ord("a") + (i % 26)) for i in range(120))
    chunks = list(cd.chunk_fixed(text, chunk_size=50, overlap=10))
    assert chunks[0][-10:] == chunks[1][:10]


def test_chunk_fixed_empty_text_yields_nothing():
    assert list(cd.chunk_fixed("")) == []


# ── 策略 B：语义分块 ──────────────────────────────────────────────────────────

def test_semantic_keeps_table_as_standalone_chunk():
    """表格不与正文合并（chunk_documents.py:104 的设计意图：防 embedding 变差）。"""
    blocks = [blk("text", "a" * 120), blk("table", "T" * 50), blk("text", "b" * 120)]
    out = list(cd.chunk_semantic(blocks))
    assert len(out) == 3  # 实测基线
    assert [c["metadata"]["block_types"] for c in out] == [["text"], ["table"], ["text"]]
    assert [len(c["content"]) for c in out] == [120, 50, 120]
    assert out[1]["content"] == "T" * 50  # 表格内容原样，未被拼接


def test_semantic_drops_text_below_min_chunk_size():
    """正文不足 min_chunk_size(默认100) 被丢弃；表格块不受此限（直接 yield）。"""
    out = list(cd.chunk_semantic([blk("text", "tiny"), blk("table", "T" * 50)]))
    assert [(c["metadata"]["block_types"], len(c["content"])) for c in out] == [(["table"], 50)]


def test_semantic_title_forces_flush_and_keeps_last_two_section_levels():
    """标题强制切分；section 用 ' > ' 拼接 section_path。"""
    blocks = [blk("text", "a" * 120, path=["P1"]), blk("title", "标题", path=["P2"]), blk("text", "b" * 120, path=["P2"])]
    out = list(cd.chunk_semantic(blocks))
    assert len(out) >= 2
    assert out[0]["metadata"]["section"] == "P1"
    # block_types 来自 set（源码 list({...})），跨进程顺序不稳定 → 按集合比
    assert set(out[1]["metadata"]["block_types"]) == {"title", "text"}
    assert out[1]["metadata"]["section"] == "P2"


def test_semantic_is_ocr_propagates_as_any():
    """块内只要有一块是 OCR，整块 is_ocr 必须为 True。"""
    blocks = [blk("text", "a" * 60, ocr=False), blk("text", "b" * 60, ocr=True)]
    out = list(cd.chunk_semantic(blocks))
    assert len(out) == 1
    assert out[0]["metadata"]["is_ocr"] is True


def test_semantic_table_chunk_keeps_raw_table_json():
    """默认 semantic 索引也要保留二维表，否则 /table/lookup 必须先切 structured。"""
    table = blk("table", "| 项目 | 2023年 |", page=7)
    table["raw_table"] = [["项目", "2023年"], ["营业收入", "100"]]
    out = list(cd.chunk_semantic([table]))
    assert out[0]["metadata"]["table_json"] == [["项目", "2023年"], ["营业收入", "100"]]
    assert out[0]["metadata"]["parent_id"] == ""


# ── 策略 C：层级分块 ──────────────────────────────────────────────────────────

def test_hierarchical_children_carry_parent_id_and_parent_content():
    """小to大检索的命门：子块必须带 parent_id 和可直接给 LLM 的 parent_content。"""
    out = list(cd.chunk_hierarchical([blk("text", "z" * 1000)]))
    assert out, "1000 字应至少产出 1 个子块"
    ids = {c["metadata"]["parent_id"] for c in out}
    assert len(ids) == 1
    for c in out:
        assert len(c["content"]) <= 400  # child_size 默认 400
        assert c["metadata"]["parent_content"] == "z" * 1000


def test_hierarchical_marks_page_num_as_unknown():
    """源码注释：层级分块页码对应关系复杂，刻意置 -1。钉住它，别被"顺手优化"改掉。"""
    out = list(cd.chunk_hierarchical([blk("text", "z" * 1000, page=7)]))
    assert all(c["metadata"]["page_num"] == -1 for c in out)


# ── chunk_id ─────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("idx,expect", [(0, "600519_2021_00000"), (7, "600519_2021_00007"), (99999, "600519_2021_99999")])
def test_build_chunk_id_zero_padded_to_five(idx: int, expect: str):
    """chunk_id 是索引对齐的主键，补零位数变了就会和已有 faiss_meta 对不上。"""
    assert cd.build_chunk_id("600519", "2021", idx) == expect
