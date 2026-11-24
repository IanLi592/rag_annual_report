"""补位锚点守卫：指标行标签与数值列的最小资格（2026-11-21 修复缺陷 1）。

实测失效模式（v1 Q11「比较贵州茅台和五粮液2022年的销售毛利率，哪家**公司**更高」）：
两家年报块里都没有"销售毛利率"这一行，旧代码退到 2 字碎片标签"公司"，把第五节
环保表的「主要: 水污染物」当成财务指标取进了上下文。

两道闸各自独立成立，所以守卫按机制分层，一个突变只该打红一条：
  1. 标签至少 3 个汉字（`test_2char_metric_label_is_not_an_anchor`）
  2. 数值列至少一格像金额（`test_text_valued_row_is_not_backfilled`）
  3~5. 「像金额」的三条分支各钉一种真实写法：无分组小数 / 长整数 / 百分号
  6.   真实场景的形状回归钉（两道闸任意一道在位即可挡住，不单独作为突变锚点）
"""
from __future__ import annotations

import rag_pipeline as rp

SEC = "第二节 公司简介和主要财务指标 > 六、主要会计数据和财务指标"


def meta(label: str, value: str, stock: str = "000858", year: str = "2022",
         section: str = SEC) -> list[dict]:
    return [{
        "chunk_id": f"{stock}_{year}_t0",
        "stock_code": stock,
        "year": year,
        "page_num": 11,
        "section": section,
        "content": "主要会计数据和财务指标",
        "source_file": f"{stock}_{year}_五粮液_{year}年年度报告.json",
        "table_json": [["项目", f"{year}年"], [label, value]],
    }]


def pipe_of(m):
    pipe = object.__new__(rp.RAGPipeline)
    pipe.vec_store = type("V", (), {"meta_list": m})()
    return pipe


def backfill(m, question: str) -> str:
    items = pipe_of(m)._table_context_items(question)
    return items[0]["content"] if items else ""


# ── 1：2 字标签没资格当补位锚 ─────────────────────────────────────────────────

def test_2char_metric_label_is_not_an_anchor():
    """"商誉"是 2 字指标名，但 2 字标签里混杂大量碎片（公司/资产/总额）。
    取"宁可退回正常检索，也不拿碎片赌一行"——这条钉住这个取舍。"""
    out = backfill(meta("商誉", "1,234,567.00"), "五粮液2022年的商誉是多少？")
    assert out == "", f"2 字标签被当成指标锚：{out!r}"


# ── 2：数值列不像金额的行不补位 ───────────────────────────────────────────────

def test_text_valued_row_is_not_backfilled():
    """标签 3 字、合法，但整行都是文字说明 —— 这才是 Q11 那次的取错行形状。"""
    m = meta("毛利率", "主要: 水污染物")
    out = backfill(m, "五粮液2022年的毛利率是多少？")
    assert out == "", f"纯文字行被当金额取出：{out!r}"
    assert "水污染物" not in out


# ── 3~5：「像金额」的三条分支各钉一种写法 ─────────────────────────────────────

def test_plain_decimal_amount_is_backfilled():
    """万元单位表常见 3220.55 这种不带千分位的小数，不能被金额判据误杀。"""
    out = backfill(meta("研发投入", "3220.55"), "五粮液2022年的研发投入是多少？")
    assert "3220.55" in out, out


def test_long_integer_amount_is_backfilled():
    """元单位表偶尔写 8252000000 这种长整数（无千分位），同样是合法金额。"""
    out = backfill(meta("研发费用", "8252000000"), "五粮液2022年的研发费用是多少？")
    assert "8252000000" in out, out


def test_percent_amount_is_backfilled():
    """毛利率类问题的答案本身就是百分数，必须放行。"""
    out = backfill(meta("毛利率", "91.60%"), "五粮液2022年的毛利率是多少？")
    assert "91.60%" in out, out


# ── 6：真实场景形状回归（双闸联合，不单独作突变锚）────────────────────────────

def test_q11_env_pollution_row_is_not_backfilled():
    """复刻 Q11：两家都无毛利率行，五粮液块里有环保"公司"行。"""
    m = meta("公司", "主要: 水污染物；主要污染物及特征污染物",
             section="第五节 环境和社会责任 > 一、重大环保问题")
    q = "比较贵州茅台和五粮液2022年的销售毛利率，哪家公司更高？分别是多少？"
    out = backfill(m, q)
    assert "水污染物" not in out, "又把环保表当财务指标取出来了"


def test_year_like_cell_is_not_amount():
    """谓词级钉一次：4 位整数是年份/序号，不是金额（长整数分支要求 ≥5 位）。"""
    assert not rp.looks_like_amount("2022")
    assert not rp.looks_like_amount("主要: 水污染物")
    assert rp.looks_like_amount("272,699,660,092.25")
