#!/usr/bin/env python
"""从最新一次评估产物里做逐题归因，并顺手校验落盘对齐。

两件事：
  1. 归因 —— 只看均值答不了"Context Precision 0.27 是被哪类题拖的"。这里按题型给均值，
     并列出最差的几道题，直接指向要修的检索场景。
  2. 自检 —— 用逐题分数重算均值，和顶层 scores 里存的均值对一遍。
     两者不一致说明 evaluate.py 里"按行序 zip 回填"错位了（那是真会静默污染结论的 bug）。

用法：python scripts/ci/analyze_scores.py [--top 8] [--file 指定json]
退出码：0 一致且有分数 / 1 均值对不上或全无助益数据 / 2 找不到可用产物
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
RESULTS = REPO / "evaluation" / "results"
METRICS = ("faithfulness", "answer_relevancy", "context_precision", "context_recall")
TOL = 1e-6


def load_newest(explicit: str | None) -> tuple[str, dict]:
    if explicit:
        p = Path(explicit)
        if not p.exists():
            raise FileNotFoundError(explicit)
        return str(p), json.loads(p.read_text(encoding="utf-8"))

    # 自动发现：只认"带逐题 ragas 分数"的产物。§13 之前的旧文件没有这些分数，
    # 拿它们出报告等于用空数据编结论 —— 宁可直接 SKIP。
    found: list[tuple[str, dict]] = []
    for p in sorted(RESULTS.glob("native_*.json")):
        try:
            d = json.loads(p.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as e:
            print(f"[CI][WARN] 跳过读不了的产物 {p.name}: {type(e).__name__}")
            continue
        if any((r.get("ragas") or {}) for r in d.get("raw_results", [])):
            found.append((str(p), d))
    if not found:
        raise FileNotFoundError(
            f"{RESULTS} 下没有带逐题 ragas 分数的产物（§13 之前跑的都缺这列）")
    path, doc = max(found, key=lambda t: os.path.getmtime(t[0]))
    return path, doc


def scored_rows(doc: dict) -> list[dict]:
    return [r for r in doc.get("raw_results", []) if r.get("ragas")]


def means_from_rows(rows: list[dict]) -> dict[str, float]:
    out = {}
    for m in METRICS:
        vals = [float(r["ragas"][m]) for r in rows if r["ragas"].get(m) is not None]
        out[m] = statistics.fmean(vals) if vals else float("nan")
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--top", type=int, default=8)
    ap.add_argument("--file", default=None)
    a = ap.parse_args()

    try:
        path, doc = load_newest(a.file)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"[CI][SKIP] {e}")
        return 2

    rows = scored_rows(doc)
    skipped = len(doc.get("raw_results", [])) - len(rows)
    if not rows:
        # 没有任何逐题分数：这是"没数据"，不是"数据坏了"，判 SKIP 才不误导。
        # 另外 by_type 为空时下面 max() 会直接 ValueError，这里顺带堵住。
        print(f"[CI][SKIP] {os.path.basename(path)} 里没有逐题 ragas 分数"
              f"（§13 之前的产物就是这种）—— 无法归因")
        return 2
    print(f"产物: {os.path.basename(path)}")
    print(f"逐题有分数 {len(rows)} 题，无分数（should_refuse 等）{skipped} 题\n")

    by_type: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_type[r.get("question_type", "unknown")].append(r)

    w = max(len(t) for t in by_type)
    print(f"{'题型':<{w}}{'n':>3} " + " ".join(f"{m[:14]:>15}" for m in METRICS))
    for t, rs in sorted(by_type.items(), key=lambda kv: len(kv[1]), reverse=True):
        mm = means_from_rows(rs)
        print(f"{t:<{w}}{len(rs):>3} " + " ".join(f"{mm[m]:>15.3f}" for m in METRICS))

    worst = sorted(rows, key=lambda r: (r["ragas"].get("context_precision") is None,
                                        r["ragas"].get("context_precision", 1.0)))[: a.top]
    print(f"\ncontext_precision 最低的 {len(worst)} 题：")
    for r in worst:
        cp = r["ragas"].get("context_precision")
        print(f"  id={r.get('question_id'):>3} cp={cp if cp is None else round(float(cp), 3)} "
              f"ctx={len(r.get('contexts', []))} {r['question'][:34]}")

    # ── 自检：逐题重算 vs 顶层存的均值 ────────────────────────────────────
    stored = {k: v for k, v in doc.get("scores", {}).items() if k in METRICS}
    recomputed = means_from_rows(rows)
    bad = []
    print("\n落盘对齐自检（逐题重算 vs 顶层均值）：")
    for m in METRICS:
        if m not in stored:
            continue
        diff = abs(stored[m] - recomputed[m])
        flag = "一致" if diff <= TOL else f"不一致 差{diff:.4f}"
        print(f"  {m:<20} 顶层={stored[m]:.4f} 重算={recomputed[m]:.4f} -> {flag}")
        if diff > TOL:
            bad.append(m)

    if bad:
        print(f"\n[CI][FAIL] {bad} 的均值与逐题分数对不上 —— zip 回填疑似错位，分数不可信")
        return 1
    print("\n[CI][PASS] 逐题分数与顶层均值自洽")
    return 0


if __name__ == "__main__":
    sys.exit(main())
