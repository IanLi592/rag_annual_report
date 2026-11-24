"""同页阅读顺序 + 章节栈层级 的离线守卫（不读真实 PDF、不联网、零成本）。

守两件真实缺陷（都用五粮液 2021 第 5 页的实测坐标复刻）：

  A. 表格盖章时机。修复前 `_parse_page_rule` 第 2 步把所有表格立刻 append，
     第 3 步才逐行扫文字并推进章节栈 —— 同一页内「表格永远排在标题之前」，
     表格盖的是「进入本页之前」的旧栈。实测该页 4 张表（股票代码/董秘联系人/
     披露地点/注册变更）全被标成「第一节 重要提示和释义 > 释 义」，而它们真正
     的标题就在同一页、就在表格上方；全库 4,491 个表格块有 3,059 个（68.1%）
     暴露于该缺陷。

  B. H3 截断长度。修复前硬切 `stack[:2]`，隐含假设栈一定是 [H1, H2, H3]；
     语料普遍是「第X节」当 H1、「一、xxx」直接就是最低层（修复前全库仅 3 个
     chunk 带 H2 层），于是每个新 H3 都把上一个同级标题永久留在栈中间，
     修复前实测 2,957 块路径里同时挂着两个不同的「一、」样式。

假页面只用真实解析器会碰到的那几种接口：find_tables / extract / bbox /
get_text("dict")。表格数据是二维数组，和线上「双轨」的 raw_table 同构。
"""
from __future__ import annotations

import parse_pdf as pp


# ── 假页面 ────────────────────────────────────────────────────────────────────

class FakeTable:
    def __init__(self, bbox, rows):
        self.bbox = bbox
        self._rows = rows

    def extract(self):
        return self._rows


class FakePlumbPage:
    """pdfplumber.Page 的最小面：find_tables() 与 extract_tables()。"""

    def __init__(self, tables):
        # tables: [(bbox, rows)]，按调用方给的顺序返回（可以故意乱序）
        self._tables = [FakeTable(bbox, rows) for bbox, rows in tables]

    def find_tables(self):
        return list(self._tables)

    def extract_tables(self):
        # 真实 pdfplumber 0.11.10 里两者等价（extract_tables 内部就是
        # [t.extract(**text_settings) for t in find_tables()]），
        # 保留这个入口，改回旧调用式的突变体也能跑起来。
        return [t.extract() for t in self._tables]


class FakeFitzPage:
    """PyMuPDF.Page 的最小面：get_text("text"|"dict") / get_images。"""

    def __init__(self, lines):
        # lines: [(y, text, fontsize, is_bold)]
        self._lines = lines

    def get_text(self, mode="text", flags=0):
        if mode == "text":
            return "\n".join(t for _y, t, _fs, _b in self._lines)
        return {
            "blocks": [{
                "type": 0,
                "lines": [{
                    "bbox": (60.0, y, 300.0, y + 10.0),
                    "spans": [{"text": t, "size": fs,
                               "font": "Bold" if bold else "Serif"}],
                } for (y, t, fs, bold) in self._lines],
            }] if self._lines else []
        }

    def get_images(self, full=True):
        return []

    @property
    def rect(self):
        return (0, 0, 595, 842)


def parse_page(lines, tables, stack=None, levels=None, page_num=4):
    """跑一页规则解析，返回 blocks（不落盘、不读 PDF）。"""
    p = object.__new__(pp.AnnualReportParser)
    p.blocks = []
    p._section_stack  = list(stack or [])
    p._section_levels = list(levels or [])
    raw_text = "正文占位。" * 30          # >50 字，确保不会误判成扫描页去调 OCR
    p._parse_page_rule(page_num, FakeFitzPage(lines), FakePlumbPage(tables),
                       raw_text=raw_text)
    return p.blocks


def types_of(blocks):
    return [b.block_type for b in blocks]


T1_ROWS = [["股票简称", "五粮液", "股票代码", "000858"],
           ["股票上市交易所", "深圳证券交易所", "变更前的股票简称", "无"]]
T2_ROWS = [["", "董事会秘书", "证券事务代表"],
           ["姓名", "占位甲", "占位乙、占位丙"]]   # 纯 fixture，不对应任何真实高管


# ── A. 同页阅读顺序 ───────────────────────────────────────────────────────────

def build_wuliangye_page5():
    """五粮液 2021 第 5 页：标题/表格交替，y 坐标取自真 PDF 实测。"""
    lines = [
        (97.67,  "第二节 公司简介和主要财务指标", 15.96, False),
        (142.83, "一、公司信息", 12.00, True),
        (450.32, "二、联系人和联系方式", 12.00, True),
        (600.00, "公司坚持聚焦主业，持续提升产品品质。", 10.50, False),
    ]
    tables = [
        ((60.0, 162.04, 500.0, 443.21), T1_ROWS),
        ((60.0, 469.65, 500.0, 590.00), T2_ROWS),
    ]
    return lines, tables


def test_page5_blocks_follow_reading_order():
    """表格不能整页抢跑到标题前面。"""
    lines, tables = build_wuliangye_page5()
    blocks = parse_page(lines, tables,
                        stack=["第一节 重要提示和释义", "释  义"], levels=[1, 3])
    assert types_of(blocks) == ["title", "title", "table", "title", "table", "text"]


def test_table_is_stamped_with_the_title_above_it():
    """核心断言：每张表拿它上方那个标题，而不是上一页留下的旧栈。"""
    lines, tables = build_wuliangye_page5()
    blocks = parse_page(lines, tables,
                        stack=["第一节 重要提示和释义", "释  义"], levels=[1, 3])
    t1, t2 = [b for b in blocks if b.block_type == "table"]
    assert t1.section_path == ["第二节 公司简介和主要财务指标", "一、公司信息"], t1.section_path
    assert t2.section_path == ["第二节 公司简介和主要财务指标", "二、联系人和联系方式"], t2.section_path


def test_old_stack_never_leaks_into_this_pages_tables():
    """旧栈（上一页的「释 义」）一个字都不该再出现在本页表格里。"""
    lines, tables = build_wuliangye_page5()
    blocks = parse_page(lines, tables,
                        stack=["第一节 重要提示和释义", "释  义"], levels=[1, 3])
    for t in [b for b in blocks if b.block_type == "table"]:
        assert "释  义" not in t.section_path
        assert "重要提示" not in t.section_path[0]


def test_table_above_every_title_keeps_previous_stack():
    """反向对照：位于本页所有标题之上的表格，必须继续用进入本页之前的栈。

    这条钉住「flush 时机」——如果 flush 被挪到 _update_section 之后，
    表格会被它下方的标题倒着污染，这里就会红。
    """
    lines = [(400.00, "三、会计数据和财务指标摘要", 12.00, True)]
    tables = [((60.0, 80.0, 500.0, 200.0), T1_ROWS)]
    blocks = parse_page(lines, tables,
                        stack=["第十节 财务报告", "一、审计报告"], levels=[1, 3])
    assert types_of(blocks) == ["table", "title"]
    assert blocks[0].section_path == ["第十节 财务报告", "一、审计报告"]


def test_tables_queue_in_y_order_even_if_finder_returns_them_scrambled():
    """pdfplumber 不保证 find_tables 按 y 返回，入库前必须自己排序。"""
    lines = [(700.00, "上表所列数据经审计确认无误。", 10.50, False)]
    tables = [
        ((60.0, 400.0, 500.0, 500.0), T2_ROWS),      # 故意先给下面的表
        ((60.0, 100.0, 500.0, 300.0), T1_ROWS),
    ]
    blocks = parse_page(lines, tables, stack=["第二节 公司简介"], levels=[1])
    assert types_of(blocks) == ["table", "table", "text"]
    assert blocks[0].raw_table == T1_ROWS, "两张表换了位置（没按 y 排序）"
    assert blocks[1].raw_table == T2_ROWS
    assert blocks[0].content.startswith("| 股票简称")


def test_pure_table_page_still_emits_its_table():
    """整页只有表格（无一行正文）：页尾收尾 flush 不能少，否则表格整张丢失。"""
    tables = [((60.0, 100.0, 500.0, 300.0), T1_ROWS)]
    blocks = parse_page([], tables, stack=["第二节 公司简介"], levels=[1])
    assert types_of(blocks) == ["table"]
    assert blocks[0].section_path == ["第二节 公司简介"]


def test_table_block_keeps_both_copies_of_the_table():
    """双轨不能被顺序改造弄丢：content 仍是 Markdown，raw_table 仍是二维数组。"""
    lines, tables = build_wuliangye_page5()
    blocks = parse_page(lines, tables, stack=["第一节"], levels=[1])
    t1 = [b for b in blocks if b.block_type == "table"][0]
    assert t1.raw_table == T1_ROWS
    assert t1.content == pp.table_to_markdown(T1_ROWS)
    for row in T1_ROWS:                              # 两份记录必须描述同一张表
        for cell in row:
            assert cell in t1.content, "Markdown 那份丢了单元格"
    assert t1.page_num == 5                                  # page_num+1 的口径不变


# ── B. 章节栈的 H3 截断 ───────────────────────────────────────────────────────

def stack_of(title_levels):
    """按顺序把标题推进栈，返回 (stack, levels)。"""
    p = object.__new__(pp.AnnualReportParser)
    p._section_stack, p._section_levels = [], []
    for title in title_levels:
        p._update_section(title, pp.classify_heading(title))
    return p._section_stack, p._section_levels


def test_h3_siblings_do_not_nest_under_each_other():
    """没有真实 H2 时，新 H3 要顶掉旧 H3，而不是挂在它下面。"""
    stack, levels = stack_of(["第二节 公司简介和主要财务指标",
                              "一、公司信息", "二、联系人和联系方式",
                              "三、信息披露及备置地点", "四、注册变更情况"])
    assert stack == ["第二节 公司简介和主要财务指标", "四、注册变更情况"], stack
    assert levels == [1, 3], levels


def test_h3_still_nests_under_a_real_h2():
    """有真实 H2 时，H3 必须留在 H2 之下 —— 不能把层级压平。"""
    stack, levels = stack_of(["第五章 财务报告", "第二节 财务报表",
                              "一、合并资产负债表", "二、合并利润表"])
    assert stack == ["第五章 财务报告", "第二节 财务报表", "二、合并利润表"], stack
    assert levels == [1, 2, 3], levels


def test_h3_after_h2_does_not_keep_stale_h3():
    """[H1,H2,H3] 再来一个 H3：切在 H2 之后，旧的 H3 必须掉。"""
    p = object.__new__(pp.AnnualReportParser)
    p._section_stack  = ["第五章 财务报告", "第二节 财务报表", "一、合并资产负债表"]
    p._section_levels = [1, 2, 3]
    p._update_section("二、合并利润表", 3)
    assert p._section_stack == ["第五章 财务报告", "第二节 财务报表", "二、合并利润表"]
    assert p._section_levels == [1, 2, 3]


def test_stale_sibling_never_reaches_section_path():
    """把 B 的缺陷接到 A 的消费端：路径里不能同时出现两个「一、」式同级标题。"""
    lines, tables = build_wuliangye_page5()
    blocks = parse_page(lines, tables, stack=["第一节 重要提示和释义"], levels=[1])
    for b in blocks:
        leaf = [s for s in b.section_path if s.startswith("一、")]
        assert len(leaf) <= 1, f"{b.block_type} 路径挂了过期同级：{b.section_path}"
