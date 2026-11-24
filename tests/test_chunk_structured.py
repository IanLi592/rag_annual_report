"""structured 分块（表格摘要注入 + 章节树 Parent-Child）的守卫。

守的是结构不变量，不是指标 —— 指标在 scripts/ci/measure_index_ab.py 里量。
这里每一条都对应一个真实会静默出错的地方：
  - 子块指向不存在的父块 → 检索时父块拼装静默退回子块，"小to大"名存实亡
  - 表格摘要为空串 → 该表在向量路彻底消失
  - 表格内容又流进 text 子块 → 4,491 重复块那个老毛病复发
"""
from __future__ import annotations

import chunk_documents as cd
import pytest


def blk(btype, content, page=1, path=None, raw=None, ocr=False):
    return {"block_type": btype, "content": content, "page_num": page,
            "section_path": path if path is not None else ["公司", "第十节 财务报告"],
            "is_ocr": ocr, "raw_table": raw}


TABLE = [["项目", "2023年", "2022年"],
         ["营业收入", "147,693,604,994.14", "124,099,843,771.99"],
         ["", "", ""],
         ["净利润", "74,734,071,550.75", "62,717,467,870.12"]]


# ── 表格清洗与摘要 ────────────────────────────────────────────────────────────

def test_clean_table_drops_empty_rows_and_cols():
    raw = [["a", "", "b"], ["", "", ""], ["1", "", "2"]]
    assert cd.clean_table(raw) == [["a", "b"], ["1", "2"]]


def test_clean_table_handles_empty_and_ragged():
    assert cd.clean_table([]) == []
    assert cd.clean_table([["", ""], ["", ""]]) == []
    out = cd.clean_table([["a"], ["1", "2"]])     # 行宽不齐要补齐再裁
    assert all(len(r) == len(out[0]) for r in out)


def test_summary_pairs_header_with_values():
    s = cd.summarize_table(TABLE, "第十节 财务报告")
    assert s.startswith("【表格摘要】第十节 财务报告")
    # 第一列是行标签，不该被当成普通列配对出"项目: 营业收入"这种无信息量对子
    assert "项目: 营业收入" not in s
    assert "行「营业收入」：2023年: 147,693,604,994.14，2022年: 124,099,843,771.99" in s
    assert "行「净利润」：2023年: 74,734,071,550.75" in s


def test_summary_includes_company_and_year():
    s = cd.summarize_table(TABLE, "第十节 财务报告", company="贵州茅台", year="2023")
    assert s.startswith("【表格摘要】贵州茅台2023年报 · 第十节 财务报告")


def test_structured_children_pass_company_into_summary():
    children, _ = cd.chunk_structured(
        [blk("table", "| 项目 |", raw=TABLE)], company="贵州茅台", year="2023")
    assert any("贵州茅台2023年报" in c["content"] for c in children)


def test_summary_of_empty_table_is_empty():
    assert cd.summarize_table([], "x") == ""
    assert cd.summarize_table([["", ""], ["", ""]], "x") == ""


def test_summary_row_cap():
    big = [["k", "v"]] + [[f"r{i}", f"v{i}"] for i in range(50)]
    s = cd.summarize_table(big, "s", max_rows=10)
    assert "行「r9」" in s and "行「r10」" not in s


# ── 章节树 Parent-Child ──────────────────────────────────────────────────────

def make_blocks():
    return [
        blk("title", "第十节 财务报告", 5),
        blk("text", "公司本年度经营情况稳定，" + "收入增长。" * 20, 5),
        blk("table", "| 项目 | 2023年 |\n| --- | --- |\n| 营业收入 | 147,693,604,994.14 |", 6, raw=TABLE),
        blk("text", "审计意见为标准无保留意见。" * 12, 7),
        blk("title", "第十一节 备查文件", 9),
        blk("text", "备查文件目录如下。" * 10, 9),
    ]


def test_children_reference_existing_parents():
    children, parents = cd.chunk_structured(make_blocks())
    pids = {p["parent_id"] for p in parents}
    assert children, "没有子块"
    for c in children:
        assert c["metadata"]["parent_id"] in pids, "子块指向不存在的父块"


def test_child_content_is_substring_of_its_parent():
    """父块是子块内容的拼接 —— 这条不变量一旦破了，"父块拼装"就是假的。"""
    children, parents = cd.chunk_structured(make_blocks())
    by_id = {p["parent_id"]: p["content"] for p in parents}
    for c in children:
        body = c["content"]
        # 子块可能带标题前缀，父块里存的是去前缀的正文；两种都接受
        core = body.split("\n", 1)[1] if body.count("\n") and not body.startswith("【表格摘要】") else body
        assert core in by_id[c["metadata"]["parent_id"]] or body in by_id[c["metadata"]["parent_id"]], \
            f"子块内容不在其父块里: {body[:40]!r}"


def test_table_child_carries_summary_and_raw():
    """子块 = 摘要 + 原始 markdown。

    早先版本只放摘要，测试也断言"不该有管道符" —— 实测证明那是错的：
    旧实现里表格文字在 table/text 两处重复，那份重复在充当检索面的数据增强，
    去掉后 BM25 Hit 从 0.400 掉到 0.304。块内带两份才既干净又不丢检索面。
    """
    children, _ = cd.chunk_structured(make_blocks())
    tabs = [c for c in children if c["metadata"]["block_types"] == ["table"]]
    assert tabs, "表格没有产出子块"
    for t in tabs:
        assert t["content"].startswith("【表格摘要】"), t["content"][:30]
        assert "|" in t["content"], "原始表格文本丢了，检索面会塌"
        assert t["metadata"]["table_json"], "结构化表格没进 metadata"
        assert t["metadata"]["table_json"][1][0] == "营业收入"


def test_parents_carry_section_and_level():
    _, parents = cd.chunk_structured(make_blocks())
    for p in parents:
        assert p["section"], "父块没有章节背景"
        assert p["level"] in (1, 2, 3)
        assert p["page_start"] >= 1


def test_oversized_single_block_is_split():
    """单个 6000 字段落不能被原样当成一个子块（父块上限会随之失效）。
    这条是反向验证抓出来的：最初实现里它原样通过，父块 6000 字。"""
    blocks = [blk("title", "第一节", 1), blk("text", "字。" * 3000, 1)]
    children, parents = cd.chunk_structured(blocks, max_child=800, parent_cap=2000)
    assert all(len(c["content"]) <= 800 for c in children), \
        [len(c["content"]) for c in children]
    assert len(children) >= 7


def test_parent_size_capped():
    """父块按**子块边界**切，不是按字节切 —— 所以真实上界是 cap + 一个子块。
    钉住这个上界即可；若哪天改成 cap+1 字节硬切，会把子块内容拦腰截断。"""
    long_text = "字。" * 3000
    blocks = [blk("title", "第一节", 1), blk("text", long_text, 1),
              blk("text", long_text, 2), blk("text", long_text, 3)]
    _, parents = cd.chunk_structured(blocks, parent_cap=2000)
    assert len(parents) >= 2, "超长章节没被切成多个父块"
    assert all(len(p["content"]) <= 2000 + 800 for p in parents)


def test_empty_input_yields_nothing():
    assert cd.chunk_structured([]) == ([], [])


def test_title_only_section_is_dropped_not_crashed():
    children, parents = cd.chunk_structured([blk("title", "只有标题", 1)])
    assert children == [] and parents == []
