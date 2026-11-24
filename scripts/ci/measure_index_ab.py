#!/usr/bin/env python
"""两套索引的混合检索 A/B：semantic（线上）vs structured（新分块）。

量三件事，缺一不可：
  1. Hit@4 / MRR@4 —— 检索质量
  2. **拼装父块后的上下文字符数** —— 这才是送进 LLM 的真实体量。
     structured 的"子块召回、父块拼装"会让上下文变长而不是变短，
     所以"削减上下文"这类说法必须拿这个数说话，不能拿子块长度说话。
  3. 带/不带精排的差值（--rerank），与 measure_rerank.py 口径一致。

用法：
  python scripts/ci/measure_index_ab.py --sample 40
  python scripts/ci/measure_index_ab.py --sample 40 --rerank
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
sys.path.insert(0, str(REPO / "src"))

SETS = {
    "semantic":   REPO / "vectorstore",
    "structured": REPO / "vectorstore" / "structured",
}


def load_set(name: str):
    """BM25 一律走生产构建路径（BM25Store.__init__），经临时 META_PATH 指向目标 meta。

    不许手工拼 BM25Okapi：本轮排查证明临时拼装与生产路径会给出不同聚合结果
    （0.288 vs 0.380，来源未定位），所有结论必须只出自生产路径。
    """
    import faiss
    import rag_pipeline as rp
    d = SETS[name]
    idx = faiss.read_index(str(d / "faiss_index.bin"))
    meta = json.load(open(d / "faiss_meta.json", encoding="utf-8"))
    parents = {}
    pf = d / "parents.json"
    if pf.exists():
        parents = {p["parent_id"]: p["content"] for p in json.load(open(pf, encoding="utf-8"))}

    tmp = REPO / "evaluation" / "results" / f"_measure_meta_{name}.json"
    tmp.write_text(json.dumps(meta, ensure_ascii=False), encoding="utf-8")
    rp.META_PATH = tmp
    bm = rp.BM25Store()
    return idx, meta, parents, bm


def first_hit(items, target_docs):
    for i, it in enumerate(items, 1):
        k = f"{it.get('stock_code','')}_{it.get('year','')}"
        if any(k.startswith(td) for td in target_docs):
            return i
    return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", type=int, default=40)
    ap.add_argument("--rerank", action="store_true")
    ap.add_argument("--floor", type=float, default=0.7)
    ap.add_argument("--top-k", type=int, default=4)
    a = ap.parse_args()

    if not os.getenv("DASHSCOPE_API_KEY"):
        print("[CI][SKIP] 向量召回需要 DASHSCOPE_API_KEY")
        return 2

    import rag_pipeline as rp
    client = rp.get_client()
    qs = [q for q in json.load(open(REPO / "evaluation" / "questions_v2.json",
                                    encoding="utf-8"))["questions"] if q.get("target_docs")]
    qs = qs[:a.sample]
    print(f"题数 {len(qs)}  top_k={a.top_k}  rerank={'on floor=' + str(a.floor) if a.rerank else 'off'}\n")

    results = {}
    for name in SETS:
        idx, meta, parents, bm = load_set(name)
        vec = rp.VectorStore(client)
        vec.index, vec.meta_list = idx, meta
        t0 = time.perf_counter()
        build_s = time.perf_counter() - t0

        hits, mrrs, chars, docs_n = [], [], [], []
        for q in qs:
            qtext = rp.expand_query_entities(q["question"], meta)
            v = vec.search(qtext, rp.TOP_K_RETRIEVE)
            b = bm.search(qtext, rp.TOP_K_RETRIEVE)
            cands = rp.reciprocal_rank_fusion(v, b)
            if a.rerank:
                final = rp.rerank(qtext, cands, a.top_k, a.floor)
            else:
                final = cands[:a.top_k]
            ctx, _ = rp.build_context(final, parents)
            pos = first_hit(final, q["target_docs"])
            hits.append(1 if pos else 0)
            mrrs.append(1 / pos if pos else 0.0)
            chars.append(len(ctx))
            docs_n.append(len(final))
        results[name] = {
            "hit": sum(hits) / len(hits), "mrr": sum(mrrs) / len(mrrs),
            "chars": sum(chars) / len(chars), "docs": sum(docs_n) / len(docs_n),
            "bm25_build_s": round(build_s, 1),
        }
        print(f"{name:11} meta={len(meta):>6} 父块={len(parents):>5} BM25建库{build_s:5.1f}s")

    s, t = results["semantic"], results["structured"]
    print(f"\n{'口径':<12}{'Hit@%d' % a.top_k:>9}{'MRR@%d' % a.top_k:>9}{'上下文字符':>12}{'平均条数':>9}")
    for n, r in (("semantic", s), ("structured", t)):
        print(f"{n:<12}{r['hit']:>9.3f}{r['mrr']:>9.3f}{r['chars']:>12.0f}{r['docs']:>9.2f}")
    print(f"{'structured−semantic':<12}{t['hit']-s['hit']:>+9.3f}{t['mrr']-s['mrr']:>+9.3f}"
          f"{(t['chars']-s['chars'])/max(s['chars'],1)*100:>+11.1f}%")
    print("\n[CI][PASS] A/B 完成")
    return 0


if __name__ == "__main__":
    sys.exit(main())
