#!/usr/bin/env python
"""分块方案对比的**唯一**测量入口（BM25 单路，零 API）。

为什么存在：本轮排查中发现同一份数据、同一构建路径，在不同临时脚本里给出过
0.380 与 0.288 两个 Hit@4（差异来源至今未定位）。教训是**临时测量脚本不可信**。
本工具用两条硬门槛把这类问题挡在结论外面：

  1. 只走生产代码路径构建 BM25（`BM25Store.__init__`，经临时 META_PATH 指向目标语料），
     不允许任何手工拼装的 BM25Okapi。
  2. **每个配置独立跑两遍（两个子进程），数字不一致就退出码 3 并拒绝出结论。**
     不可复现的数字不是数字。

同时打印语料内容 sha，保证"比的是哪份数据"可追溯。

用法：python scripts/ci/measure_chunksets.py [--repeat 2]
退出码：0 可复现 / 2 前置缺失 / 3 不可复现（拒绝出结论）
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

CI_DIR = Path(__file__).resolve().parent
REPO = CI_DIR.parents[1]

SETS = {
    "semantic_old":  REPO / "data" / "chunks" / "all_semantic_INDEXMATCH.json.bak",
    "semantic_new":  REPO / "data" / "chunks" / "all_semantic.json",
    "structured":    REPO / "data" / "chunks" / "all_structured.json",
}


def _one_pass(set_name: str, emit: str) -> int:
    """单子进程：构建 store 并聚合，结果写 JSON。"""
    sys.path.insert(0, str(REPO / "src"))
    import rag_pipeline as rp

    src = SETS[set_name]
    if not src.exists():
        print(f"[CI][SKIP] 缺语料 {src.name}")
        return 2
    cs = json.loads(src.read_text(encoding="utf-8"))
    meta = [{
        "chunk_id": c["chunk_id"], "content": c["content"],
        "stock_code": c["metadata"]["stock_code"], "year": c["metadata"]["year"],
        "page_num": c["metadata"]["page_num"], "section": c["metadata"]["section"],
        "block_types": c["metadata"]["block_types"], "is_ocr": c["metadata"]["is_ocr"],
        "strategy": c["metadata"]["strategy"], "source_file": c["metadata"]["source_file"],
        "parent_content": "", "parent_id": c["metadata"].get("parent_id", ""),
    } for c in cs]
    sha = hashlib.sha256("".join(m["content"] for m in meta).encode()).hexdigest()[:10]

    tmp = REPO / "evaluation" / "results" / "_measure_meta.json"
    tmp.write_text(json.dumps(meta, ensure_ascii=False), encoding="utf-8")
    rp.META_PATH = tmp
    store = rp.BM25Store()                       # 只走生产构建路径

    qs = [q for q in json.loads((REPO / "evaluation" / "questions_v2.json")
                                .read_text(encoding="utf-8"))["questions"]
          if q.get("target_docs")]
    hits, mrrs = [], []
    for q in qs:
        sel = store.search(q["question"], 8)[:4]
        pos = None
        for i, it in enumerate(sel, 1):
            k = f"{it['stock_code']}_{it['year']}"
            if any(k.startswith(td) for td in q["target_docs"]):
                pos = i
                break
        hits.append(1 if pos else 0)
        mrrs.append(1 / pos if pos else 0.0)

    Path(emit).write_text(json.dumps({
        "set": set_name, "n_chunks": len(meta), "sha": sha, "n_q": len(qs),
        "hit": round(sum(hits) / len(hits), 4), "mrr": round(sum(mrrs) / len(mrrs), 4),
    }, ensure_ascii=False), encoding="utf-8")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repeat", type=int, default=2)
    ap.add_argument("--child", default="")
    ap.add_argument("--emit", default="")
    a = ap.parse_args()
    if a.child:
        return _one_pass(a.child, a.emit)

    py = os.getenv("CI_PYTHON") or sys.executable
    tmp = REPO / "evaluation" / "results"
    tmp.mkdir(parents=True, exist_ok=True)

    table = {}
    for name in SETS:
        runs = []
        for r in range(a.repeat):
            out = tmp / f"_measure_{name}_{r}.json"
            res = subprocess.run([py, str(__file__), "--child", name, "--emit", str(out)],
                                 capture_output=True, text=True, encoding="utf-8", errors="replace")
            if res.returncode != 0 or not out.exists():
                print(f"[CI][FAIL] {name} 第 {r+1} 遍没跑出来：{(res.stderr or res.stdout)[-200:]}")
                return 2
            runs.append(json.loads(out.read_text(encoding="utf-8")))
        if len({(x["hit"], x["mrr"], x["sha"]) for x in runs}) != 1:
            print(f"[CI][FAIL] {name} 不可复现：{[(x['hit'], x['mrr']) for x in runs]} —— 拒绝出结论")
            return 3
        table[name] = runs[0]

    print(f"{'方案':<14}{'chunks':>8}{'sha':>12}{'Hit@4':>9}{'MRR@4':>9}")
    for name, r in table.items():
        print(f"{name:<14}{r['n_chunks']:>8}{r['sha']:>12}{r['hit']:>9.3f}{r['mrr']:>9.3f}")
    base = table.get("semantic_old")
    if base:
        for name, r in table.items():
            if name != "semantic_old":
                print(f"  {name} − semantic_old: Hit {r['hit']-base['hit']:+.3f}  MRR {r['mrr']-base['mrr']:+.3f}")
    print("\n[CI][PASS] 各配置重复测量一致，结论可复现")
    return 0


if __name__ == "__main__":
    sys.exit(main())
