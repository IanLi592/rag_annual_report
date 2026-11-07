#!/usr/bin/env python
"""拒答专项测量：把“模型拒答”“空检索”“正常回答”分开统计。

为什么单独一个脚本：
  - RAGAS 会排除 refusal 题（ground_truth 是描述性的），所以拒答质量没有指标；
  - 旧 evaluate.py 只按关键词算 refuse_rate，会把 n_ctx=0 的检索失败算成拒答；
  - 简历里的“拒答正确率”必须能回答“25 题里几题真拒答、几题是空检索”。

用法：
  python scripts/ci/measure_refusal.py                 # 默认同时测线上配置和评估配置
  python scripts/ci/measure_refusal.py --config served # 只测线上（重排+阈值）
  python scripts/ci/measure_refusal.py --config eval   # 只测评估（无重排）

退出码：0 完成 / 2 缺 key 或题集。
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

CI_DIR = Path(__file__).resolve().parent
REPO = CI_DIR.parents[1]
for p in (str(REPO / "src"), str(REPO / "evaluation")):
    if p not in sys.path:
        sys.path.insert(0, p)

from refusal import classify_refusal  # noqa: E402


def load_refusal_questions() -> list[dict]:
    path = REPO / "evaluation" / "questions_v2.json"
    if not path.exists():
        raise FileNotFoundError(f"题集不存在: {path}")
    qs = json.loads(path.read_text(encoding="utf-8"))["questions"]
    return [q for q in qs if q.get("category") == "refusal"]


def run_config(pipe, rows: list[dict], use_rerank: bool, floor: float | None) -> dict:
    pipe.use_rerank = use_rerank
    pipe.rerank_min_score = floor
    out = []
    for q in rows:
        r = pipe.query(q["question"], verbose=False)
        n_ctx = len(r.get("retrieved") or [])
        category = classify_refusal(r.get("answer", ""), n_ctx)
        out.append({
            "id": q["id"],
            "question": q["question"],
            "n_ctx": n_ctx,
            "category": category,
            "answer": r.get("answer", ""),
        })

    n = len(out)
    refused = sum(1 for x in out if x["category"] == "refused_with_context")
    empty = sum(1 for x in out if x["category"] == "empty_context")
    answered = sum(1 for x in out if x["category"] == "answered")
    judgeable = n - empty
    return {
        "use_rerank": use_rerank,
        "floor": floor,
        "n": n,
        "refused_with_context": refused,
        "empty_context": empty,
        "answered": answered,
        "refusal_rate_all": round(refused / n, 4) if n else 0.0,
        "refusal_rate_judgeable": round(refused / judgeable, 4) if judgeable else 0.0,
        "rows": out,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", choices=["served", "eval", "both"], default="both")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    if not os.getenv("DASHSCOPE_API_KEY"):
        print("[CI][SKIP] 需要 DASHSCOPE_API_KEY")
        return 2
    try:
        qs = load_refusal_questions()
    except FileNotFoundError as e:
        print(f"[CI][SKIP] {e}")
        return 2

    import rag_pipeline as rp
    pipe = rp.RAGPipeline(use_bm25=True, use_rerank=True, rerank_min_score=None)
    served_floor = float(os.getenv("RERANK_MIN_SCORE")) if os.getenv("RERANK_MIN_SCORE") else None

    configs = []
    if a.config in ("served", "both"):
        configs.append(("served", True, served_floor))
    if a.config in ("eval", "both"):
        configs.append(("eval", False, None))

    results = []
    for name, use_rerank, floor in configs:
        print(f"\n── {name}: use_rerank={use_rerank} floor={floor} ──")
        res = run_config(pipe, qs, use_rerank, floor)
        res["config"] = name
        results.append(res)
        print(
            f"  {res['n']} 题：真拒答 {res['refused_with_context']} / "
            f"空检索 {res['empty_context']} / 正常回答 {res['answered']}；"
            f"可判定题拒答率 {res['refusal_rate_judgeable']:.1%}")

    payload = {"questions": len(qs), "configs": results}
    out = Path(a.out) if a.out else REPO / "evaluation" / "results" / "refusal_measure_latest.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[CI][PASS] 已保存 → {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
