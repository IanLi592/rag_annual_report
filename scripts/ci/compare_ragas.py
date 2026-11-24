#!/usr/bin/env python
"""比较两份 RAGAS 评估产物，输出指标 diff 与逐题回退。

为什么单独抽出来：`evaluate.py` 只负责跑分，CI 需要的是“这次 vs 基线”。
没有对比就没有回归门；只看单次均值无法判断 Chunk/Prompt 改动是升是降。

口径：
  - 优先用顶层 `scores`；缺失时用 `raw_results[*].ragas` 重算均值
  - 逐题对比只在两边的 question_id 交集上做
  - 默认容忍度 0.03（RAGAS 在 6~20 题上的抖动很大，小于 3pp 不当回归）
  - `--strict` 时才以回退退出码 1；否则只报告，不拦人

用法：
  python scripts/ci/compare_ragas.py --baseline evaluation/results/ragas_baseline.json \
      --latest evaluation/results/native_20261107_190000.json --strict
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

METRICS = ("faithfulness", "answer_relevancy", "context_precision", "context_recall")
LABELS = {
    "faithfulness":       "Faithfulness",
    "answer_relevancy":   "Answer Relevancy",
    "context_precision":  "Context Precision",
    "context_recall":     "Context Recall",
}


def _use_utf8_streams() -> None:
    """把 stdout/stderr 钉成 UTF-8。

    实测踩坑（2026-11-16）：Windows 控制台默认 GBK，而报告里有 `⚠`（回退标记）。
    于是一旦真的出现回退，`print(render(report))` 就抛 UnicodeEncodeError ——
    评估早就跑完了，门却永远打不出对比结果，退出码还被 traceback 顶成 1，
    看起来像「回归门失败」，实际是编码崩的。越该看的那次越看不到。
    """
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass          # 被换成非 TextIOWrapper（例如捕获管道）时保持原样


def load_doc(path: str | Path) -> dict:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"找不到评估产物: {p}")
    doc = json.loads(p.read_text(encoding="utf-8"))
    if not isinstance(doc, dict):
        raise ValueError(f"{p} 不是合法的评估产物（顶层不是对象）")
    return doc


def per_question(doc: dict) -> dict[str, dict[str, float | None]]:
    rows = doc.get("raw_results") or []
    out: dict[str, dict[str, float | None]] = {}
    for row in rows:
        qid = row.get("question_id")
        if qid is None:
            continue
        rag = row.get("ragas") or {}
        out[str(qid)] = {
            m: (float(rag[m]) if isinstance(rag.get(m), (int, float)) else None)
            for m in METRICS
        }
    return out


def metric_mean(doc: dict, metric: str) -> float | None:
    """顶层 scores 优先；没有就用逐题分数重算（与 analyze_scores 同口径）。"""
    scores = doc.get("scores") or {}
    v = scores.get(metric)
    if isinstance(v, (int, float)):
        return float(v)
    vals = [
        float(r[metric])
        for r in per_question(doc).values()
        if r.get(metric) is not None
    ]
    return statistics.mean(vals) if vals else None


def compare_docs(baseline: dict, latest: dict, tolerance: float = 0.03) -> dict:
    bq, lq = per_question(baseline), per_question(latest)
    common = sorted(set(bq) & set(lq), key=lambda x: (len(x), x))

    metrics = {}
    for m in METRICS:
        b, l = metric_mean(baseline, m), metric_mean(latest, m)
        delta = None if (b is None or l is None) else l - b
        metrics[m] = {
            "baseline": b,
            "latest": l,
            "delta": delta,
            "regressed": bool(delta is not None and delta < -tolerance),
        }

    regressions = []
    for qid in common:
        for m in METRICS:
            b, l = bq[qid].get(m), lq[qid].get(m)
            if b is None or l is None:
                continue
            if l - b < -tolerance:
                regressions.append({
                    "question_id": qid, "metric": m,
                    "baseline": b, "latest": l, "delta": l - b,
                })
    regressions.sort(key=lambda r: r["delta"])
    return {
        "tolerance": tolerance,
        "questions": {
            "baseline": len(bq),
            "latest": len(lq),
            "common": len(common),
            "only_baseline": sorted(set(bq) - set(lq)),
            "only_latest": sorted(set(lq) - set(bq)),
        },
        "metrics": metrics,
        "regressions": regressions,
    }


def render(report: dict) -> str:
    q = report["questions"]
    lines = [
        f"题数：baseline={q['baseline']} latest={q['latest']} 交集={q['common']} "
        f"（容忍度 {report['tolerance']:.2f}）",
        "",
        "| 指标 | baseline | latest | Δ |",
        "|---|---:|---:|---:|",
    ]
    for m in METRICS:
        r = report["metrics"][m]
        b = "—" if r["baseline"] is None else f"{r['baseline']:.4f}"
        l = "—" if r["latest"] is None else f"{r['latest']:.4f}"
        d = "—" if r["delta"] is None else f"{r['delta']:+.4f}"
        flag = " ⚠" if r["regressed"] else ""
        lines.append(f"| {LABELS[m]} | {b} | {l} | {d}{flag} |")

    if report["regressions"]:
        lines += ["", "逐题回退（top 10）：", "", "| 题号 | 指标 | baseline | latest | Δ |", "|---:|---|---:|---:|---:|"]
        for r in report["regressions"][:10]:
            lines.append(
                f"| {r['question_id']} | {LABELS[r['metric']]} | "
                f"{r['baseline']:.4f} | {r['latest']:.4f} | {r['delta']:+.4f} |")
    else:
        lines += ["", "逐题：无超过容忍度的回退。"]
    return "\n".join(lines)


def main() -> int:
    _use_utf8_streams()
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", required=True)
    ap.add_argument("--latest", required=True)
    ap.add_argument("--tolerance", type=float, default=0.03)
    ap.add_argument("--json-out", default=None)
    ap.add_argument("--strict", action="store_true",
                    help="任一指标回退超过容忍度时退出码 1；默认只报告")
    a = ap.parse_args()

    try:
        baseline, latest = load_doc(a.baseline), load_doc(a.latest)
    except (FileNotFoundError, ValueError) as e:
        print(f"[CI][SKIP] {e}", file=sys.stderr)
        return 2

    report = compare_docs(baseline, latest, a.tolerance)
    print(render(report))
    if a.json_out:
        Path(a.json_out).write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\n[CI] diff 已写入 {a.json_out}")

    if a.strict and any(r["regressed"] for r in report["metrics"].values()):
        print("[CI][FAIL] 有指标回退超过容忍度", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
