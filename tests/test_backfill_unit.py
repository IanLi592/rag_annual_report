"""表格计量单位标注守卫（2026-11-21 修复缺陷 2）。

缺陷形状（实测 v1 Q12 的补位上下文，两家单位不同却并排裸呈）：

    【表格精确取数】002415 2023年报 … 研发费用：本年发生额: 11,392,948,404.47
    【表格精确取数】300750 2023年报 … 研发费用：2023年: 1,835,610.84

前者是元、后者是万元，模型直接比大小必然得出相反结论。

单位有两层来源，都不成立就**不标注**（保持修复前的逐字输出）：
  1. 行标签 / 表头自带的「（元）（万元）」后缀 —— 就在索引里，最可靠且离线可得；
  2. data/parsed 里的页级「单位：X」声明，且只在页窗内采信（实测 002415 2023
     最后一次声明在第 136 页，而研发费用附注表在第 164 页：无限 carry-over 会把
     元表标成万元，**标错比不标更糟**）。
同一上下文里部分行查得到、部分查不到时，查不到的标「单位未标注」，让量纲差异可见。

守卫全部离线：页级来源用 tmp_path 里的合成解析文件，不读真实的 data/parsed。
"""
from __future__ import annotations

import json

import rag_pipeline as rp

# 夹具一律用 data/parsed 里**不存在**的年份（本机只有 000858 的 2021/2022/2023）：
# 这些守卫只能看 tmp_path 里的合成解析文件，不能被机器上的解析产物影响
WT_FILE = "000858_2019_五粮液_2019年年度报告.json"
WT2_FILE = "000858_2020_五粮液_2020年年度报告.json"
MT_FILE = "600519_2019_贵州茅台_2019年年度报告.json"
# 解析文件压根不存在的场合（守卫要证明"查不到就一个字都不加"）
GHOST_FILE = "000858_2018_五粮液_2018年年度报告.json"

# 混合单位场景用「同公司跨年度」造，而不是跨公司：跨公司机制另有守卫
# （tests/test_cross_doc.py），这里不重复依赖它，才能让两套突变各自只打红一条。
MIX_Q = "五粮液2019到2020年的研发费用分别是多少？"


def tbl(stock, year, page, source, label="研发费用", value="1,000.00", header=None):
    return {
        "chunk_id": f"{stock}_{year}_{page}",
        "stock_code": stock,
        "year": year,
        "page_num": page,
        "section": "第十节 财务报告 > 3、合并利润表",
        "content": "主要会计数据和财务指标",
        "source_file": source,
        "table_json": [header or ["项目", f"{year}年度"], [label, value]],
    }


def pipe_of(meta, unit_dir=None, with_dir=True):
    pipe = object.__new__(rp.RAGPipeline)
    pipe.vec_store = type("V", (), {"meta_list": meta})()
    if with_dir:
        pipe.unit_dir = unit_dir      # 真实 RAGPipeline 在 __init__ 里赋值
    return pipe


def write_parsed(dir_, name, blocks):
    (dir_ / name).write_text(
        json.dumps({"meta": {}, "source": "t", "blocks": blocks}, ensure_ascii=False),
        encoding="utf-8")


def blob_of(pipe, q):
    items = pipe._table_context_items(q)
    assert len(items) == 1, f"多条补位上下文 = 槽位失控：{len(items)}"
    return items[0]["content"]


# ── 层 2：页级声明 ────────────────────────────────────────────────────────────

def test_page_declaration_is_only_trusted_within_the_window():
    decls = [(20, "万元"), (40, "元")]
    assert rp.unit_at(decls, 20) == "万元", "同页声明必须生效（单位行就贴在表格上方）"
    assert rp.unit_at(decls, 22) == "万元", "窗口内继续生效"
    assert rp.unit_at(decls, 30) == "", "窗外声明属于别的表 —— 宁可未知，也不标错单位"
    assert rp.unit_at(decls, 40) == "元"
    assert rp.unit_at(decls, 19) == "", "首次声明之前没有单位可用"
    assert rp.unit_at(decls, 22, window=0) == "", "window=0 只认同页声明"


def test_multiple_declarations_on_one_page_take_the_last():
    blocks = [{"content": "单位：万元\n…\n单位：元", "page_num": 30}]
    assert rp.parse_unit_declarations(blocks) == [(30, "元")]


def test_unit_word_alternatives_are_recognised():
    blocks = [{"content": "单位:百万元", "page_num": 8},
              {"content": "编制单位：某某公司\n单位：亿元", "page_num": 9},
              {"content": "单位：千元", "page_num": 10}]
    assert rp.parse_unit_declarations(blocks) == [(8, "百万元"), (9, "亿元"), (10, "千元")]


def test_broken_or_absent_parsed_file_degrades_to_unknown(tmp_path):
    (tmp_path / "bad.json").write_text("{这不是合法 JSON", encoding="utf-8")
    assert rp.load_unit_declarations("bad.json", tmp_path) == []
    assert rp.load_unit_declarations("missing.json", tmp_path) == []
    assert rp.load_unit_declarations("", tmp_path) == []


# ── 层 1：行标签 / 表头后缀 ───────────────────────────────────────────────────

def test_row_label_suffix_works_without_any_parsed_file():
    """摘要表把单位写在行标签里（「…净利润（元）」），索引内即可判定。"""
    meta = [tbl("600519", "2019", 61, MT_FILE,
                label="归属于上市公司股东的净利润（元）", value="16,800,411,032.05")]
    pipe = pipe_of(meta, unit_dir=None, with_dir=False)   # 连解析目录都不给
    blob = blob_of(pipe, "贵州茅台2019年的归属于上市公司股东的净利润是多少？")
    assert "（单位：元）" in blob, blob


def test_label_suffix_wins_over_page_declaration(tmp_path):
    """行标签说元、页声明说万元 —— 行标签更贴这行数字，必须赢。"""
    write_parsed(tmp_path, WT_FILE, [{"content": "单位：万元", "page_num": 61}])
    meta = [tbl("000858", "2019", 61, WT_FILE, label="总资产（元）", value="1,000.00")]
    blob = blob_of(pipe_of(meta, tmp_path), "五粮液2019年的总资产是多少？")
    assert "（单位：元）" in blob and "（单位：万元）" not in blob, blob


def test_label_suffix_survives_pdf_line_break():
    """实测 300750 2023 的行标签被断成「营业收入（万\\n元）」，不能因此漏标。"""
    meta = [tbl("000858", "2018", 61, GHOST_FILE, label="营业收入（万\n元）",
                value="40,091,704.49")]
    blob = blob_of(pipe_of(meta, with_dir=False), "五粮液2018年的营业收入是多少？")
    assert "（单位：万元）" in blob, blob


# ── 端到端标注规则 ────────────────────────────────────────────────────────────

def test_mixed_item_marks_known_unit_and_flags_unknown(tmp_path):
    write_parsed(tmp_path, WT_FILE, [{"content": "单位：万元", "page_num": 60},
                                     {"content": "| 项目 |", "page_num": 61}])
    meta = [tbl("000858", "2019", 61, WT_FILE, value="266,906,613.97"),
            tbl("000858", "2020", 62, WT2_FILE, value="6,271,756,574.61")]
    blob = blob_of(pipe_of(meta, tmp_path), MIX_Q)
    assert blob.count("（单位：万元）") == 1, blob
    assert blob.count(rp.UNIT_UNKNOWN_MARK) == 1, "量纲不一致时必须让'这行单位未知'可见"


def test_all_units_unknown_keeps_pre_fix_output_verbatim(tmp_path):
    """一道单位都查不到时一个字都不加 —— 保住既有逐字输出与上下文长度。"""
    meta = [tbl("000858", "2019", 61, WT_FILE), tbl("000858", "2020", 62, WT2_FILE)]
    assert "单位" not in blob_of(pipe_of(meta, tmp_path), MIX_Q)


def test_ghost_parsed_dir_pipe_adds_nothing():
    """没有 unit_dir 的离线守卫实例不得依赖 data/parsed 的真实内容。"""
    meta = [tbl("000858", "2018", 61, GHOST_FILE)]
    blob = blob_of(pipe_of(meta, with_dir=False), "五粮液2018年的研发费用是多少？")
    assert "单位" not in blob, blob


# ── 注入点本身 ────────────────────────────────────────────────────────────────

def test_real_pipeline_wires_unit_dir_in_init():
    """真实构造的 pipeline 必须挂上单位来源；离线守卫实例靠"没有这个属性"保持 hermetic。"""
    import inspect
    assert "self.unit_dir" in inspect.getsource(rp.RAGPipeline.__init__)
