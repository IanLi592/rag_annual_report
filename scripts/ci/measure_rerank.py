#!/usr/bin/env python
"""Rerank 漏斗的实测：排序质量 + 上下文削减比例。

背景（实测抓到的事实，不是猜测）：
  旧 rerank() 只排序+截断，**永远返回 top_k 条**，所以"重排过滤长尾低分文档
  使上下文 Token 削减 X%"这个说法在旧代码下恒为 0%。现已加 min_score 过滤能力，
  这个脚本负责把它的真实效果量出来。

口径说明（面试被问要能答）：
  - 削减比例用**字符数**度量，不是 token 数。中文场景字符≈token 的上界，
    真实 token 削减要用模型 tokenizer 另测；我不会把字符比例说成 token 比例。
  - 命中判定沿用 compare_strategies：doc_key = stock_code_year 与 target_docs 前缀匹配。
  - 基线是"RRF 后取 top4"，与线上 use_rerank=False 的行为一致。
  - 阈值必须来自分数分布，不能为了凑某个比例反推。

用法：
  python scripts/ci/measure_rerank.py --sample 25            # 先看分布
  python scripts/ci/measure_rerank.py --sample 25 --floors 0.0,0.2,0.5
退出码：0 完成 / 2 前置缺失
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

CI_DIR = Path(__file__).resolve().parent
REPO = CI_DIR.parents[1]
for p in (str(REPO / "src"), str(REPO / "evaluation")):
    if p not in sys.path:
        sys.path.insert(0, p)


def load_questions(sample: int) -> list[dict]:
    """优先用 v2（125 题）里有 target_docs 的派生题；没有就退回 v1。"""
    for name in ("questions_v2.json", "questions.json"):
        path = REPO / "evaluation" / name
        if not path.exists():
            continue
        qs = json.loads(path.read_text(encoding="utf-8"))["questions"]
        qs = [q for q in qs if q.get("target_docs")]
        if qs:
            print(f"题集 {name}：{len(qs)} 题有 target_docs，取前 {min(sample, len(qs))} 题")
            return qs[:sample]
    raise FileNotFoundError("没有可用题集，先跑 build_question_set.py")


def first_hit(items: list[dict], target_docs: list[str]):
    for i, it in enumerate(items, 1):
        key = f"{it.get('stock_code', '')}_{it.get('year', '')}"
        if any(key.startswith(td) for td in target_docs):
            return i
    return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", type=int, default=25)
    ap.add_argument("--floors", default="", help="逗号分隔的分数下限，如 0.0,0.2,0.5")
    ap.add_argument("--top-k", type=int, default=4)
    ap.add_argument("--rerank-query", choices=["expanded", "original"], default="expanded",
                    help="重排用实体归一后的问题还是原问题")
    a = ap.parse_args()

    try:
        qs = load_questions(a.sample)
    except FileNotFoundError as e:
        print(f"[CI][SKIP] {e}")
        return 2

    import rag_pipeline as rp
    if not os.getenv("DASHSCOPE_API_KEY"):
        print("[CI][SKIP] 向量召回需要 DASHSCOPE_API_KEY（.env 里填）")
        return 2

    client = rp.get_client()
    vec = rp.VectorStore(client)
    bm25 = rp.BM25Store()
    t0 = time.perf_counter()
    reranker = rp.get_reranker()
    load_s = time.perf_counter() - t0
    if reranker is None:
        print("[CI][SKIP] Reranker 加载失败，测不了")
        return 2
    print(f"Reranker 加载 {load_s:.1f}s（单例复用，后续每请求不再付这个时间）\n")

    rows = []
    all_scores: list[float] = []
    for q in qs:
        qtext = rp.expand_query_entities(q["question"], vec.meta_list)
        rq = qtext if a.rerank_query == "expanded" else q["question"]
        v = vec.search(qtext, rp.TOP_K_RETRIEVE)
        b = bm25.search(qtext, rp.TOP_K_RETRIEVE)
        cands = rp.reciprocal_rank_fusion(v, b)
        pairs = [(rq, c["content"]) for c in cands]
        scores = reranker.predict(pairs)
        for c, s in zip(cands, scores):
            c["rerank_score"] = float(s)
            all_scores.append(float(s))
        ranked = sorted(cands, key=lambda x: -x["rerank_score"])
        rows.append({
            "id": q["id"], "td": q["target_docs"],
            "baseline": cands[:a.top_k],          # RRF 原序取 top4
            "rerank": ranked[:a.top_k],           # 只重排
            "n_cand": len(cands),
            "scores": [c["rerank_score"] for c in ranked],
        })

    def summarize(items_key: str, floor=None):
        hits, mrrs, chars, kept = [], [], [], []
        for r in rows:
            sel = r[items_key]
            if floor is not None:
                sel = [c for c in sel if c["rerank_score"] >= floor][:a.top_k]
            pos = first_hit(sel, r["td"])
            hits.append(1 if pos else 0)
            mrrs.append(1 / pos if pos else 0.0)
            chars.append(sum(len(c["content"]) for c in sel))
            kept.append(len(sel))
        n = len(rows)
        return {"hit": sum(hits) / n, "mrr": sum(mrrs) / n,
                "chars": sum(chars) / n, "kept": sum(kept) / n}

    base = summarize("baseline")
    rr = summarize("rerank")
    print(f"{'口径':<22}{'Hit@%d' % a.top_k:>9}{'MRR@%d' % a.top_k:>10}{'平均上下文字符':>16}{'平均条数':>10}")
    print(f"{'基线 RRF top%d' % a.top_k:<22}{base['hit']:>9.3f}{base['mrr']:>10.3f}"
          f"{base['chars']:>16.0f}{base['kept']:>10.1f}")
    print(f"{'重排 top%d（不过滤）' % a.top_k:<20}{rr['hit']:>9.3f}{rr['mrr']:>10.3f}"
          f"{rr['chars']:>16.0f}{rr['kept']:>10.1f}")
    print(f"{'重排带来的变化':<22}{rr['hit']-base['hit']:>+9.3f}{rr['mrr']-base['mrr']:>+10.3f}"
          f"{(rr['chars']-base['chars'])/max(base['chars'],1)*100:>+15.1f}%")

    # 分数分布 —— 阈值必须从这里选，不能反过来凑比例
    s = sorted(all_scores)
    def pct(p):
        return s[min(int(len(s) * p), len(s) - 1)]
    print(f"\nrerank 分数分布（n={len(s)}）：min {s[0]:.3f} / p10 {pct(.10):.3f} / "
          f"p25 {pct(.25):.3f} / 中位 {pct(.5):.3f} / p75 {pct(.75):.3f} / max {s[-1]:.3f}")

    if a.floors:
        print(f"\n{'下限':<12}{'Hit':>8}{'MRR':>8}{'平均字符':>12}{'削减':>10}{'平均条数':>10}")
        for f in [float(x) for x in a.floors.split(",")]:
            m = summarize("rerank", floor=f)
            cut = (m["chars"] - base["chars"]) / max(base["chars"], 1) * 100
            print(f"{f:<12.2f}{m['hit']:>8.3f}{m['mrr']:>8.3f}{m['chars']:>12.0f}"
                  f"{cut:>+9.1f}%{m['kept']:>10.2f}")

    print("\n[CI][PASS] 测量完成。请把上面的数字原样引用，不要换算成 token 比例。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
