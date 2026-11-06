#!/usr/bin/env python
"""领域词表的 A/B 实测：同一份 BM25 语料，加载 vs 不加载词表，看 Hit@4 / MRR@4 差多少。

为什么值得单独测：jieba 把「归属于上市公司股东的净利润」这类长术语切碎后，
BM25 的 term 权重被稀释、精确匹配失效 —— 这是财报关键词检索最常见的翻车点。
简历里"防止切词粉碎"这句必须有个数撑着。

口径说明（面试被问要能答）：
  - 只测 BM25 关键词召回，不测向量召回 —— 词表影响不到 embedding（模型内部分词）。
  - 不加元数据过滤，与 compare_strategies 的消融口径一致（过滤会人为抬高命中率）。
  - top_k=4 与线上送给 LLM 的上下文数量一致；召回时多取一倍（top_k*2）再截断。
  - BM25 索引每个配置只建一次（10–12 s），比 compare_strategies 的 per-call 重建快得多。

用法：python scripts/ci/measure_domain_dict.py [--top-k 4]
退出码：0 完成 / 2 前置缺失（无索引或无题）
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

CI_DIR = Path(__file__).resolve().parent
REPO = CI_DIR.parents[1]
EXIT_PASS, EXIT_PREREQ = 0, 2


def evaluate_child(top_k: int, out_path: str, mode: str = "bm25") -> int:
    """在子进程里跑一个配置（词表开/关由环境变量决定），结果写 JSON。

    mode=bm25   只测关键词召回（零 API）
    mode=hybrid 测 RRF 融合后的净效果 —— 这才是简历那句话对应的口径，但要花 embedding 调用
    """
    sys.path.insert(0, str(REPO / "src"))
    sys.path.insert(0, str(REPO / "evaluation"))

    import jieba_domain
    stat = jieba_domain.load_domain_dict()

    import rag_pipeline as rp
    t0 = time.perf_counter()
    store = rp.BM25Store()
    build_s = time.perf_counter() - t0

    vec = None
    if mode == "hybrid":
        vec = rp.VectorStore(rp.get_client())

    qs = json.load(open(REPO / "evaluation" / "questions.json", encoding="utf-8"))["questions"]
    qs = [q for q in qs if q.get("target_docs")]
    if not qs:
        print("[CI][SKIP] questions.json 里没有带 target_docs 的题")
        return EXIT_PREREQ

    hits, mrrs, missed = [], [], []
    for q in qs:
        if mode == "hybrid":
            v = vec.search(q["question"], top_k * 2)
            b = store.search(q["question"], top_k * 2)
            retrieved = rp.reciprocal_rank_fusion(v, b)[:top_k]
        else:
            retrieved = store.search(q["question"], top_k * 2)[:top_k]
        # 与 compare_strategies.compute_metrics 同样的判定：doc_key 前缀匹配
        first = None
        for i, item in enumerate(retrieved, 1):
            doc_key = f"{item.get('stock_code', '')}_{item.get('year', '')}"
            if any(doc_key.startswith(td) for td in q["target_docs"]):
                first = i
                break
        hits.append(1 if first else 0)
        mrrs.append(1 / first if first else 0.0)
        if not first:
            missed.append({"id": q["id"], "question": q["question"][:36],
                           "top1": f"{retrieved[0].get('stock_code')}_{retrieved[0].get('year')}"
                                   if retrieved else None})

    res = {
        "mode": mode,
        "dict_enabled": stat["enabled"] and stat["loaded"] > 0,
        "words_loaded": stat["loaded"],
        "n_questions": len(qs),
        "top_k": top_k,
        "build_s": round(build_s, 2),
        "hit_rate": sum(hits) / len(hits),
        "mrr": sum(mrrs) / len(mrrs),
        "missed": missed,
    }
    Path(out_path).write_text(json.dumps(res, ensure_ascii=False), encoding="utf-8")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--top-k", type=int, default=4)
    ap.add_argument("--mode", default="bm25", choices=["bm25", "hybrid"])
    ap.add_argument("--child", action="store_true")
    ap.add_argument("--emit", default="")
    a = ap.parse_args()

    if a.child:
        return evaluate_child(a.top_k, a.emit, a.mode)

    py = os.getenv("CI_PYTHON") or sys.executable
    tmp = REPO / "evaluation" / "results"
    tmp.mkdir(parents=True, exist_ok=True)

    runs = {}
    for label, flag in (("with_dict", "1"), ("no_dict", "0")):
        out = tmp / f"domain_ab_{a.mode}_{label}.json"
        env = dict(os.environ, JIEBA_DOMAIN_DICT=flag)
        r = subprocess.run([py, str(__file__), "--child", "--emit", str(out),
                            "--top-k", str(a.top_k), "--mode", a.mode], env=env,
                           capture_output=True, text=True, encoding="utf-8", errors="replace")
        if r.returncode != 0 or not out.exists():
            print(f"[CI][FAIL] {label} 这一跑没出结果，退出码 {r.returncode}")
            print((r.stderr or r.stdout)[-400:])
            return EXIT_PREREQ
        runs[label] = json.loads(out.read_text(encoding="utf-8"))

    w, n = runs["with_dict"], runs["no_dict"]
    print(f"口径: {a.mode}  题数 {w['n_questions']}  top_k={w['top_k']}  "
          f"BM25 建库 开表 {w['build_s']}s / 关表 {n['build_s']}s")
    print(f"词表实际加载词数: {w['words_loaded']}\n")
    print(f"{'指标':<12}{'加载词表':>12}{'不加载':>12}{'差值':>12}")
    for key, name in (("hit_rate", "Hit@%d" % w["top_k"]), ("mrr", "MRR@%d" % w["top_k"])):
        d = w[key] - n[key]
        print(f"{name:<12}{w[key]:>12.4f}{n[key]:>12.4f}{d:>+12.4f}")

    fixed = [m for m in n["missed"] if m["id"] not in {x["id"] for x in w["missed"]}]
    broke = [m for m in w["missed"] if m["id"] not in {x["id"] for x in n["missed"]}]
    print(f"\n开表后被修复的题（关表 miss、开表 hit）: {len(fixed)}")
    for m in fixed:
        print(f"  id={m['id']:>3} {m['question']}  关表时 top1={m['top1']}")
    print(f"开表后反而变差的题: {len(broke)}")
    for m in broke:
        print(f"  id={m['id']:>3} {m['question']}")

    delta = w["mrr"] - n["mrr"]
    print("\n[CI][PASS] A/B 完成" if delta >= 0 else "[CI][WARN] 词表让 MRR 下降了，需要复核词表质量")
    return EXIT_PASS


if __name__ == "__main__":
    sys.exit(main())
