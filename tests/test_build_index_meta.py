"""索引侧元数据守卫：chunk 里有的字段，索引里不能悄悄丢。

`table_json` / `heading_level` / `parent_id` 是 structured 与层级分块的接口。
它们只存在于 chunk metadata 里，一旦 make_meta_list 漏拷，检索侧拿不到，
表现是"表格结构还在，但精确取数永远走不到"—— 没有报错，只能靠断言发现。
"""
from __future__ import annotations

import build_index as bi


def chunk(cid, meta=None, content="正文"):
    return {"chunk_id": cid, "content": content, "metadata": meta or {}}


def test_make_meta_list_copies_all_fields():
    cs = [chunk("c1", {
        "stock_code": "600519", "year": "2023", "page_num": 38,
        "section": "第三章 > 一、营业收入", "block_types": ["table"],
        "is_ocr": True, "strategy": "structured", "source_file": "600519_2023.pdf",
        "heading_level": 3,
        "table_json": [["项目", "金额"], ["营收", "100"]],
        "parent_content": "父块全文",
        "parent_id": "p1",
    })]
    m = bi.make_meta_list(cs)[0]
    assert m["chunk_id"] == "c1" and m["content"] == "正文"
    assert m["heading_level"] == 3
    assert m["table_json"] == [["项目", "金额"], ["营收", "100"]]
    assert m["parent_content"] == "父块全文" and m["parent_id"] == "p1"
    assert m["strategy"] == "structured" and m["is_ocr"] is True


def test_make_meta_list_defaults_without_new_fields():
    """semantic 旧索引没有这些字段，必须给出可判定的默认值而不是 KeyError。"""
    m = bi.make_meta_list([chunk("c1")])[0]
    assert m["heading_level"] == 0
    assert m["table_json"] is None
    assert m["parent_content"] == "" and m["parent_id"] == ""


def test_make_meta_list_preserves_order_and_length():
    cs = [chunk(str(i)) for i in range(5)]
    ms = bi.make_meta_list(cs)
    assert [m["chunk_id"] for m in ms] == ["0", "1", "2", "3", "4"]


def test_make_meta_list_does_not_alias_table_json():
    """table_json 是原表格的引用；索引写入方之后不应意外改动它。"""
    table = [["a", "b"]]
    m = bi.make_meta_list([chunk("c", {"table_json": table})])[0]
    assert m["table_json"] is table   # 只读语义；这里钉住"别深拷贝到丢结构"
