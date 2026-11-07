"""标题层级从解析 → 分块 → 索引的传递守卫。

heading_level 是 structured 章节树的分级依据。它一旦在某一层丢掉，
chunk_structured 会退回"路径深度"推断，表面上仍能跑，但 H1/H2/H3 全错，
父块切分和"章节背景"都会随之偏移 —— 没有报错，只能靠断言守住。
"""
from __future__ import annotations

import chunk_documents as cd


def blk(btype, content, page=1, path=None, level=0, raw=None):
    return {"block_type": btype, "content": content, "page_num": page,
            "section_path": path if path is not None else ["第三章 管理层讨论"],
            "is_ocr": False, "heading_level": level, "raw_table": raw}


def test_semantic_keeps_heading_level_on_text_chunk():
    blocks = [blk("title", "第三章 管理层讨论", level=1),
              blk("text", "正文" * 80)]
    chunks = list(cd.chunk_semantic(blocks))
    assert chunks[0]["metadata"]["heading_level"] == 1


def test_semantic_keeps_heading_level_on_table_chunk():
    blocks = [blk("table", "| 项目 | 金额 |\n| --- | --- |\n| 营收 | 100 |",
                  level=3)]
    chunks = list(cd.chunk_semantic(blocks))
    assert chunks[0]["metadata"]["heading_level"] == 3


def test_structured_prefers_explicit_level_over_path_depth():
    """路径有 3 级，但解析标的是 H1 —— 必须听解析的。"""
    blocks = [blk("title", "第十章 财务报告", level=1,
                  path=["甲公司2023年报", "第十章 财务报告", "一、合并报表"]),
              blk("text", "正文" * 80, level=0,
                  path=["甲公司2023年报", "第十章 财务报告", "一、合并报表"])]
    children, parents = cd.chunk_structured(blocks)
    assert children and parents
    assert parents[0]["level"] == 1
    assert parents[0]["heading_level"] == 1
    assert all(c["metadata"]["heading_level"] == 1 for c in children)


def test_structured_falls_back_to_path_depth_for_old_parsed_json():
    """旧 parsed JSON 没有 heading_level 字段，不能因此算出 0 级。"""
    old = {"block_type": "text", "content": "正文" * 80, "page_num": 1,
           "section_path": ["第三章", "一、经营情况"], "is_ocr": False}
    _, parents = cd.chunk_structured([old])
    assert parents[0]["level"] == 2


def test_structured_child_and_parent_levels_agree():
    blocks = [blk("title", "第二节 公司治理", level=2),
              blk("text", "正文" * 200, level=2)]
    children, parents = cd.chunk_structured(blocks)
    by_parent = {p["parent_id"]: p for p in parents}
    for c in children:
        assert c["metadata"]["heading_level"] == by_parent[c["metadata"]["parent_id"]]["level"]
