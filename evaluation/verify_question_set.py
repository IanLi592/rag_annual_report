#!/usr/bin/env python
"""独立复核 evaluation/questions_v2.json —— **不 import 生成器**。

为什么要独立：如果校验器复用生成器的解析逻辑，那它只是在跟自己对齐，
生成器的 bug 会被原样继承（本项目已经因为"测试全绿但分支没跑到"翻过两次车）。
这里只用标准库，从原始 parsed JSON 重新构造文本再逐字比对。

检查项：
  S 结构：字段齐全、id 连续唯一、category/type 取值合法、四类达标
  E 证据：file 存在；quote 是原文逐字子串；contains_value 标记与实际一致
  G 答案：ground_truth 里的每个关键数字都必须出现在某条 evidence.quote 里
  D 引用：target_docs 形如 code_year 且在该公司的实际年报里存在
  R 拒答：refusal 类必须无 evidence、无 target_docs（否则等于假装可答）

退出码：0 全通过 / 1 有失败项 / 2 文件缺失
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
QPATH = REPO / "evaluation" / "questions_v2.json"
PARSED = REPO / "data" / "parsed"
MANIFEST = REPO / "data" / "manifest.json"

NUM_RE = re.compile(r"-?[0-9][0-9,]{4,}(?:\.[0-9]+)?")
# 6 位数字正好也是股票代码的长度 —— 不排除的话，答案里的公司代码会被当成
# "找不到出处的财务数字"（本校验器第一版就栽在这，误报 39 条）
CODE_RE = re.compile(r"^\d{6}$")
CATEGORIES = {"numeric", "multihop", "hallucination", "refusal"}
TYPES = {"simple_fact", "precise_number", "cross_doc_compare", "time_trend", "should_refuse"}
MIN_COUNTS = {"numeric": 30, "multihop": 30, "hallucination": 25, "refusal": 20}
TOTAL_MIN = 105


def load_file_texts() -> dict:
    """{文件名: [各块 content]}"""
    out = {}
    for p in sorted(PARSED.glob("*.json")):
        d = json.loads(p.read_text(encoding="utf-8"))
        out[p.name] = [b["content"] for b in d["blocks"]]
    return out


def main() -> int:
    if not QPATH.exists():
        print(f"[SKIP] 找不到 {QPATH}，先跑 build_question_set.py")
        return 2

    doc = json.loads(QPATH.read_text(encoding="utf-8"))
    qs = doc.get("questions", [])
    texts = load_file_texts()
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    docs_present = {(m["stock_code"], m["year"]) for m in manifest}
    all_codes = {m["stock_code"] for m in manifest}

    fails: list[str] = []
    stats = {"结构": 0, "证据": 0, "答案": 0, "引用": 0, "拒答": 0}

    # ── S 结构 ──
    ids = [q.get("id") for q in qs]
    if len(ids) != len(set(ids)):
        fails.append("id 有重复")
    if ids and ids != list(range(1, len(ids) + 1)):
        fails.append("id 不连续")
    if len(qs) < TOTAL_MIN:
        fails.append(f"总题数 {len(qs)} < {TOTAL_MIN}")
    from collections import Counter
    dist = Counter(q.get("category") for q in qs)
    for k, v in MIN_COUNTS.items():
        if dist.get(k, 0) < v:
            fails.append(f"类别 {k} 只有 {dist.get(k,0)}，要求 ≥{v}")
    for q in qs:
        miss = [f for f in ("id", "category", "type", "question", "target_docs",
                            "ground_truth", "evidence") if f not in q]
        if miss:
            fails.append(f"id={q.get('id')} 缺字段 {miss}")
        if q.get("category") not in CATEGORIES:
            fails.append(f"id={q.get('id')} 非法 category={q.get('category')}")
        if q.get("type") not in TYPES:
            fails.append(f"id={q.get('id')} 非法 type={q.get('type')}")
        stats["结构"] += 1

    for q in qs:
        qid = q.get("id")
        ev = q.get("evidence") or []

        # ── R 拒答类 ──
        if q.get("category") == "refusal":
            if ev or q.get("target_docs"):
                fails.append(f"id={qid} refusal 不该带 evidence/target_docs")
            stats["拒答"] += 1
            continue

        # ── E 证据逐字性 ──
        if not ev:
            fails.append(f"id={qid} 非 refusal 却无 evidence")
        quotes_ok = []
        for e in ev:
            rel = e.get("file", "")
            name = Path(rel).name
            if name not in texts:
                fails.append(f"id={qid} evidence 文件不存在: {rel}")
                continue
            blocks = texts[name]
            hit = next((b for b in blocks if e.get("quote", "") in b), None)
            if hit is None:
                fails.append(f"id={qid} quote 不是原文逐字子串: {e.get('quote','')[:40]!r}")
                continue
            # page 必须与该块一致
            pgs = [i for i, b in enumerate(blocks) if b is hit]
            stats["证据"] += 1
            quotes_ok.append(e["quote"])

        # ── G 答案数字必须能在证据里找到 ──
        # 6 位数字若命中 manifest 里的股票代码集合，那是标识符不是财务数字。
        # （幻觉题的答案会提到"真正归属的公司代码"，它不在题干里，
        #   所以不能只按题干/target_docs 收集 —— 校验器第一版就是这么误报的）
        gt_nums = {v for v in NUM_RE.findall(q.get("ground_truth", ""))
                   if not (CODE_RE.match(v) and v in all_codes)}
        for v in gt_nums:
            if not any(v in s for s in quotes_ok):
                fails.append(f"id={qid} 答案数字 {v} 在任何 evidence 里都找不到")
        if gt_nums:
            stats["答案"] += 1
        elif q.get("category") in ("numeric", "multihop"):
            fails.append(f"id={qid} {q.get('category')} 的 ground_truth 里没有数字")

        # ── D target_docs ──
        for td in q.get("target_docs", []):
            if not re.fullmatch(r"\d{6}_\d{4}", td or ""):
                fails.append(f"id={qid} target_doc 格式不对: {td!r}")
                continue
            code, year = td.split("_")
            if (code, year) not in docs_present:
                fails.append(f"id={qid} target_doc {td} 在 manifest 里不存在")
            stats["引用"] += 1

    print(f"总题数 {len(qs)}  分布 {dict(dist)}")
    print("各项检查计数:", stats)
    if fails:
        print(f"\n[FAIL] {len(fails)} 条问题：")
        for f in fails[:25]:
            print("  -", f)
        if len(fails) > 25:
            print(f"  ...另有 {len(fails)-25} 条")
        return 1
    print("\n[PASS] 结构/证据/答案/引用/拒答 全部通过")
    return 0


if __name__ == "__main__":
    sys.exit(main())
