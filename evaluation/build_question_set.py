#!/usr/bin/env python
"""从年报摘要数据派生评测集 → evaluation/questions_v2.json

## 为什么这么设计

正文正则乱抓数字会出错，三条实测反例：
  - 五粮液 2023 正文里的 `净利润140.45亿元` 不是它的归母净利（302.11 亿）
  - 宁德时代 `营业收入 100.00%` 是"纳入评价范围单位占比"，不是营收
  - 海康威视 `母公司实现净利润10,552,055,515.88元` 是母公司口径，不是合并口径

而「主要会计数据」这张表在 15 份年报里有**三种完全不同的排版**（实测）：
  - 贵州茅台：markdown 表格，表头和数据在同一个 table 块
  - 五粮液：纯文本块，标签一行、数值逐行跟在后面
  - 中国平安：拆成十几个小标题/文本块，单位是**人民币百万元**，
    且标签是「归属于母公司股东的净利润」而非「归属上市公司…」

所以抽取器必须格式无关：只认「白名单标签 → 紧随其后的若干数字」这个模式。

## 三道锁（不满足就丢弃候选，绝不将就用）

1. **区域锁**：只认摘要区（锚点页起 3 页内，或前 15 页 / 锚点块后 40 块）
2. **引用锁**：证据必须是**原文逐字子串**。跨块时拆成 label_quote + value_quote 两段，
   两段各自逐字。理由：如果引用是"把 token 用换行重新拼出来的"，校验器就只是在跟
   生成器自己比，等于没校验。
3. **一致锁**：值 V 被接受，需满足 B1 或 B2
   - B1 跨报告/跨文档：同一值出现在 ≥2 份不同年报（如 2022 年报的本年列 == 2023 年报的上年列）
   - B2 同文档多处：该值在同一份年报的 ≥2 个不同块里出现（摘要表 + 正文/利润表）

## 单位处理

**题目与答案一律不带单位**，只引用原文里出现的数字字符串本身。
理由：平安用百万元、其余用元，任何换算或单位标注都可能引入错误；
而"原文里那个数字字符串"是可机器复核的。原文单位记录在 unit_in_source 字段供人核对。

用法：python evaluation/build_question_set.py [--out evaluation/questions_v2.json]
退出码：0 达标 / 1 事实太少或类别不达标
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

EVAL_DIR = Path(__file__).resolve().parent
REPO = EVAL_DIR.parent
PARSED = REPO / "data" / "parsed"

# 长标签在前，避免"净利润"抢先匹配掉长标签；同时覆盖两种披露口径
# （一般公司写"上市…股东"，中国平安写"母公司股东"）
METRICS = [
    "归属于上市公司股东的扣除非经常性损益的净利润",
    "归属于母公司股东的扣除非经常性损益后的净利润",
    "扣除非经常性损益后的加权平均净资产收益率",
    "扣除非经常性损益后的基本每股收益",
    "归属于上市公司股东的净利润",
    "归属于母公司股东的净利润",
    "归属于上市公司股东的净资产",
    "归属于母公司股东权益",
    "经营活动产生的现金流量净额",
    "加权平均净资产收益率",
    "基本每股收益",
    "稀释每股收益",
    "研发投入",
    "研发费用",
    "营业收入",
    "总资产",
    "总负债",
    "资产总额",
    "负债总额",
    "股东权益",
]
UNIT_TOKENS = ("人民币百万元", "人民币万元", "百万元", "万元", "元")
SUMMARY_ANCHOR = "主要会计数据"
# 至少 5 位数字，滤掉年份/百分比/序号噪声
NUM_RE = re.compile(r"-?[0-9][0-9,]{4,}(?:\.[0-9]+)?")
PCT_RE = re.compile(r"^-?[0-9]{1,3}(?:\.[0-9]+)?%?$")


def norm(s: str) -> str:
    """去掉空白与全角空格 —— PDF 换行会把标签切碎成「归属于上市公 司股东的净利 润」。"""
    return re.sub(r"[\s　]+", "", s or "")


def split_tokens(text: str) -> list[str]:
    out: list[str] = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith("|"):
            out.extend(c.strip() for c in line.strip("|").split("|") if c.strip())
        else:
            out.append(line)
    return out


def match_metric(nt: str):
    return next((m for m in METRICS if nt == m or (len(nt) >= 6 and nt.startswith(m))), None)


def collect_nums(toks: list[str], start: int, window: int) -> tuple[list[str], int]:
    """取标签之后、下一个标签之前的数字序列。返回 (数字列表, 最后一个数字的 token 下标)。"""
    nums: list[str] = []
    last = start
    for j in range(start + 1, min(start + 1 + window, len(toks))):
        c = norm(toks[j])
        if match_metric(c):
            break
        if PCT_RE.match(c) and len(c) <= 8:
            continue
        m = NUM_RE.match(c)
        if m:
            nums.append(m.group())
            last = j
        if len(nums) >= 6:
            break
    return nums, last


def verbatim(content: str, label_token: str, value: str):
    """返回同时包含标签与数值的原文逐字子串；两者不连续则返回 None。"""
    li = content.find(label_token)
    if li < 0:
        return None
    vi = content.find(value, li)
    if vi < 0:
        return None
    return content[li:vi + len(value)]


def extract_block(file: str, page: int, block_idx: int, content: str, unit: str) -> list[dict]:
    out = []
    toks = split_tokens(content)
    for i, t in enumerate(toks):
        metric = match_metric(norm(t))
        if not metric:
            continue
        nums, _ = collect_nums(toks, i, 9)
        if not nums:
            continue
        q = verbatim(content, t, nums[0])
        if q is None:
            continue
        out.append({"file": file, "page": page, "block": block_idx, "metric": metric,
                    "numbers": nums, "quote": q, "unit_in_source": unit})
    return out


def extract_region(file: str, blocks: list, unit: str) -> list[dict]:
    """合并摘要区成一条 token 流再抽；跨块时拆成两段逐字引用。"""
    toks: list[str] = []
    meta: list[tuple[int, int, str]] = []
    for i, b in blocks:
        for t in split_tokens(b["content"]):
            toks.append(t)
            meta.append((b["page_num"], i, b["content"]))
    out = []
    for i, t in enumerate(toks):
        metric = match_metric(norm(t))
        if not metric:
            continue
        nums, last = collect_nums(toks, i, 12)
        if not nums:
            continue
        page, blk, content = meta[i]
        q = verbatim(content, t, nums[0])
        if q is not None:
            out.append({"file": file, "page": page, "block": blk, "metric": metric,
                        "numbers": nums, "quote": q, "unit_in_source": unit})
            continue
        # 跨块：标签与数值各自逐字引用
        vpage, vblk, vcontent = meta[last]
        v = nums[0]
        if v not in vcontent or t not in content:
            continue
        out.append({"file": file, "page": page, "block": blk, "metric": metric,
                    "numbers": nums, "quote": t,
                    "value_quote": v, "value_file": file, "value_page": vpage,
                    "unit_in_source": unit})
    return out


def detect_unit(blocks: list) -> str:
    """摘要区里出现的第一个单位 token。只记录不换算。"""
    for _, b in blocks[:60]:
        for u in UNIT_TOKENS:
            if u in b["content"]:
                return u
    return ""


def load_all() -> tuple[dict, dict]:
    docs: dict = {}
    value_index: dict = defaultdict(set)        # value -> {(文件名, 块号)}
    for p in sorted(PARSED.glob("*.json")):
        d = json.loads(p.read_text(encoding="utf-8"))
        m = d["meta"]
        key = (m["stock_code"], m["year"])
        cand: list[dict] = []
        anchor_page = anchor_idx = None
        for i, b in enumerate(d["blocks"]):
            for v in NUM_RE.findall(b["content"]):
                value_index[v].add((p.name, i))
            if SUMMARY_ANCHOR in norm(b["content"]) and anchor_page is None:
                anchor_page, anchor_idx = b["page_num"], i
            in_summary = (anchor_page is not None and b["page_num"] <= anchor_page + 3) \
                or b["page_num"] <= 15
            if in_summary:
                cand.extend(extract_block(p.name, b["page_num"], i, b["content"], ""))
        if anchor_idx is not None:
            region = list(enumerate(d["blocks"]))[anchor_idx:anchor_idx + 40]
            unit = detect_unit(region)
            for c in extract_region(p.name, region, unit):
                cand.append(c)
        seen: dict = {}
        for c in cand:
            k = (c["metric"], c["numbers"][0])
            if k not in seen:
                seen[k] = c
        docs[key] = {"meta": m, "cands": list(seen.values())}
    return docs, value_index


def build_facts(docs: dict, value_index: dict) -> dict:
    """列语义：标签后第 1 个数字 = 本年列；第 2/3 个 = 上年（调整前/后）；再往后 = 前年。"""
    current: dict = defaultdict(list)
    prior: dict = defaultdict(set)
    for (code, ryear), doc in docs.items():
        for c in doc["cands"]:
            y = int(ryear)
            metric, nums = c["metric"], c["numbers"]
            current[(code, str(y), metric)].append(c)
            for offset, idx in ((1, 1), (1, 2), (2, 3), (2, 4), (2, 5), (1, 3), (2, 6)):
                if idx < len(nums):
                    prior[(code, str(y - offset), metric)].add(nums[idx])

    facts, stats = {}, Counter()
    for key, cands in current.items():
        counts = Counter(c["numbers"][0] for c in cands)
        value, _n = counts.most_common(1)[0]
        b1 = value in prior.get(key, set()) or \
            len({f for (f, _b) in value_index.get(value, set())}) >= 2
        same_file = {(f, blk) for (f, blk) in value_index.get(value, set())
                     if f == cands[0]["file"]}
        b2 = len(same_file) >= 2
        if b1 or b2:
            facts[key] = {"value": value,
                          "sources": [c for c in cands if c["numbers"][0] == value][:2],
                          "why": "B1跨报告/文档" if b1 else "B2同文档多处"}
            stats["接受_" + ("B1" if b1 else "B2")] += 1
        else:
            stats["拒绝_无佐证"] += 1
    build_facts.last_stats = stats
    return facts


def ev(cand: dict) -> list[dict]:
    """把候选的引用转成 evidence 列表（跨块的两段都收）。"""
    out = [{"file": f"data/parsed/{cand['file']}", "page": cand["page"],
            "quote": cand["quote"], "contains_value": cand["numbers"][0] in cand["quote"]}]
    if cand.get("value_quote"):
        out.append({"file": f"data/parsed/{cand['value_file']}", "page": cand["value_page"],
                    "quote": cand["value_quote"], "contains_value": True})
    return out


def make_questions(facts: dict, names: dict | None = None):
    names = names or {}

    def nm(code: str) -> str:
        """题干用公司简称，不用股票代码：真实用户不会问“000858在2021年…”。"""
        return names.get(code, code)

    ids = iter(range(1, 10_000))
    by_cm = defaultdict(dict)
    for (code, year, metric), f in facts.items():
        by_cm[(code, metric)][year] = f

    numeric, multihop, halluc = [], [], []

    for (code, year, metric), f in sorted(facts.items()):
        if len(numeric) >= 36:
            break
        s = f["sources"][0]
        numeric.append({
            "id": next(ids), "category": "numeric", "type": "precise_number",
            "difficulty": "easy",
            "question": f"{nm(code)}在{year}年的{metric}是多少？",
            "target_docs": [f"{code}_{year}"],
            "ground_truth": f"{year}年{metric}为 {f['value']}。",
            "evidence": ev(s), "unit_in_source": s.get("unit_in_source", ""),
            "note": f["why"],
        })

    # 跨年趋势
    for (code, metric), per in sorted(by_cm.items()):
        years = sorted(per)
        if len(years) < 3 or len(multihop) >= 22:
            continue
        v0 = float(per[years[0]]["value"].replace(",", ""))
        v2 = float(per[years[-1]]["value"].replace(",", ""))
        trend = "上升" if v2 > v0 else ("下降" if v2 < v0 else "持平")
        multihop.append({
            "id": next(ids), "category": "multihop", "type": "time_trend",
            "difficulty": "hard",
            "question": f"{nm(code)}的{metric}在{years[0]}到{years[-1]}年整体是上升还是下降？各年分别是多少？",
            "target_docs": [f"{code}_{y}" for y in years],
            "ground_truth": f"整体{trend}。" + "，".join(
                f"{y}年 {per[y]['value']}" for y in years) + "。",
            "evidence": [e for y in years for e in ev(per[y]["sources"][0])][:6],
        })

    # 跨公司对比（同年同指标）
    pair_seen = set()
    for (year, metric), per in sorted(_group(facts).items()):
        codes = sorted(per)
        for i, a in enumerate(codes):
            for b in codes[i + 1:]:
                if len(multihop) >= 34 or (a, b, metric) in pair_seen:
                    continue
                try:
                    va = float(per[a]["value"].replace(",", ""))
                    vb = float(per[b]["value"].replace(",", ""))
                except ValueError:
                    continue
                if va == vb:
                    continue
                pair_seen.add((a, b, metric))
                # 答案一律用原文数字字符串，保证与 evidence 逐字一致
                if va > vb:
                    hi_code, hi_v, lo_code, lo_v = a, per[a]["value"], b, per[b]["value"]
                else:
                    hi_code, hi_v, lo_code, lo_v = b, per[b]["value"], a, per[a]["value"]
                multihop.append({
                    "id": next(ids), "category": "multihop", "type": "cross_doc_compare",
                    "difficulty": "hard",
                    "question": f"{year}年，{nm(a)}和{nm(b)}的{metric}相比哪家更高？分别是多少？",
                    "target_docs": [f"{a}_{year}", f"{b}_{year}"],
                    "ground_truth": f"{hi_code} 更高：{hi_code} 为 {hi_v}，{lo_code} 为 {lo_v}。",
                    "evidence": ev(per[a]["sources"][0]) + ev(per[b]["sources"][0]),
                })

    # 幻觉对抗：张冠李戴 + 年份错位
    items = sorted(facts.items())
    all_codes = sorted({k[0] for k in facts})
    for n, ((code, year, metric), f) in enumerate(items):
        if len(halluc) >= 20:
            break
        others = [c for c in all_codes if c != code]
        if not others:
            continue
        fake = others[n % len(others)]
        halluc.append({
            "id": next(ids), "category": "hallucination", "type": "precise_number",
            "difficulty": "hard",
            "question": f"{nm(fake)}在{year}年的{metric}是不是 {f['value']} ？",
            "target_docs": [f"{fake}_{year}"],
            "ground_truth": (f"不是。{f['value']} 是 {code} 在 {year} 年的{metric}，"
                             f"不属于 {fake}；应回答 {fake} 自己的数值。"),
            "evidence": ev(f["sources"][0]),
            "note": f"错误前提：把 {code} 的数值安到 {fake}",
        })
    for (code, year, metric), f in items[:10]:
        wy = str(int(year) + 1)
        halluc.append({
            "id": next(ids), "category": "hallucination", "type": "time_trend",
            "difficulty": "medium",
            "question": f"{nm(code)}{wy}年的{metric}是 {f['value']} 吗？",
            "target_docs": [f"{code}_{year}"],
            "ground_truth": (f"{f['value']} 对应 {year} 年，不是 {wy} 年。"
                             f"若资料中没有 {wy} 年数据，应说明无法确认。"),
            "evidence": ev(f["sources"][0]),
            "note": f"错误前提：年份 {year} 被说成 {wy}",
        })
    return numeric, multihop, halluc


def _group(facts):
    g = defaultdict(dict)
    for (code, year, metric), f in facts.items():
        g[(year, metric)][code] = f
    return g


REFUSAL = [
    ("目前{c}的股价是多少？今天适合买入吗？", "年报不含实时股价与投资建议"),
    ("{c}的股价目标价和分析师评级是什么？", "年报不含第三方目标价与评级"),
    ("{c}明年预计营收能增长多少？给个业绩指引。", "年报不提供未来业绩指引"),
    ("{c}最近一个月的日K线走势如何？", "年报不含逐日行情数据"),
    ("{c}董事长个人身家有多少亿？", "年报不披露高管个人财富"),
    ("{c}今年三季度的净利润是多少？", "本语料只含年报（全年），无季度拆分"),
    ("{c}在行业里市占率排第几？", "年报一般不给第三方市占率排名"),
    ("{c}员工平均月薪是多少？", "年报披露薪酬总额与董监高报酬，不含全员平均月薪"),
    ("{c}的短视频账号粉丝数是多少？", "与年报无关"),
    ("{c}明年每股分红多少元？", "未来分红方案在年报中尚未确定"),
]
COMPANIES = {"600519": "贵州茅台", "000858": "五粮液", "300750": "宁德时代",
             "002415": "海康威视", "601318": "中国平安"}


def make_refusal(limit: int = 25):
    """设上限的理由：模板全组合会得到 50 题（占 32%），而 RAGAS 把这批排除在打分外，
    那会让"总题数"虚高。宁可题少而口径诚实。"""
    qs = []
    for code, name in COMPANIES.items():
        for tmpl, basis in REFUSAL[:5]:
            qs.append({
                "id": 0, "category": "refusal", "type": "should_refuse",
                "difficulty": "medium",
                "question": tmpl.format(c=name),
                "target_docs": [],
                "ground_truth": f"期望模型拒答或说明资料不足：{basis}。不应编造具体数字。",
                "evidence": [], "note": basis,
            })
    return qs[:limit]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(EVAL_DIR / "questions_v2.json"))
    a = ap.parse_args()

    docs, value_index = load_all()
    facts = build_facts(docs, value_index)
    print(f"载入 {len(docs)} 份年报；候选 "
          f"{sum(len(d['cands']) for d in docs.values())} 条；验证统计 {dict(B_stats())}")
    print(f"交叉验证通过的事实: {len(facts)} 条")
    print("按公司:", dict(Counter(k[0] for k in facts)))
    if len(facts) < 60:
        print("[FAIL] 可用事实太少（<60），够不出 100+ 题")
        return 1

    numeric, multihop, halluc = make_questions(
        facts, {k[0]: d["meta"].get("company_name", k[0]) for k, d in docs.items()})
    refusal = make_refusal()
    allq = numeric + multihop + halluc + refusal
    for i, q in enumerate(allq, 1):
        q["id"] = i

    dist = Counter(q["category"] for q in allq)
    covered = sorted({td.split("_")[0] for q in allq for td in q.get("target_docs", [])})
    missing = sorted(set(COMPANIES) - set(covered))
    out = {
        "version": "2.0",
        "description": "A 股年报 RAG 评测集 v2：摘要区数据经区域锁/引用锁/一致锁三重验证派生",
        "note": ("题目与答案一律不带单位（平安用百万元、其余用元，换算会引入错误）；"
                 "evidence.quote 均为原文逐字子串，由 verify_question_set.py 独立复核。"),
        "coverage": {
            "companies_with_derived_questions": covered,
            "companies_without": missing,
            "why": (f"{missing} 的摘要表在本解析产物里排版碎片化/存在多处同名标签"
                     "（实测平安 page≤15 内有多处「营业收入」，指向分部数据而非合并营收），"
                     "抽取候选未通过一致性锁，因此**宁可不给题也不给可能是错的 ground truth**。"
                     "这几家公司只由 refusal 类覆盖。" if missing else "全部公司均有派生题"),
        },
        "built_by": "evaluation/build_question_set.py",
        "questions": allq,
    }
    Path(a.out).write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print("四类分布:", dict(dist), "| 总题数:", len(allq))
    need = {"numeric": 30, "multihop": 30, "hallucination": 25, "refusal": 20}
    bad = [k for k, v in need.items() if dist.get(k, 0) < v]
    if bad:
        print(f"[FAIL] 类别不达标: {bad}")
        return 1
    print(f"已写出 {a.out}")
    return 0


def B_stats():
    return getattr(build_facts, "last_stats", {})


if __name__ == "__main__":
    sys.exit(main())
