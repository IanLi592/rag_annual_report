#!/usr/bin/env python
"""表格摘要注入的 A/B：同一批 structured 子块，开/关摘要，只比 BM25 单路。

用途：简历“表格摘要注入提升核心财务指标检索”必须由**摘要开/关**对照支持，
      不能拿 semantic vs structured 的差异冒充（两者块边界都不同）。
      本工具复用生产 `chunk_structured` 与生产 `BM25Store`，零 API、零索引。

用法：python scripts/ci/measure_table_summary.py [--sample 100]
退出码：0 完成 / 2 缺解析产物
"""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

CI_DIR = Path(__file__).resolve().parent
REPO = CI_DIR.parents[1]
sys.path.insert(0, str(REPO / "src"))


def build_meta(parsed_files, with_summary: bool):
    import chunk_documents as cd

    meta = []
    for p in parsed_files:
        data = json.loads(p.read_text(encoding="utf-8"))
        md = data.get("meta", {})
        children, _ = cd.chunk_structured(
            data.get("blocks", []),
            company=md.get("company_name", ""),
            year=str(md.get("year", "")),
            table_summary=with_summary,
        )
        for i, c in enumerate(children):
            meta.append({
                "chunk_id": f"{md.get('stock_code','')}_{md.get('year','')}_{i:05d}",
                "content": c["content"],
                "stock_code": md.get("stock_code", ""),
                "year": str(md.get("year", "")),
                "page_num": c["metadata"].get("page_num", -1),
                "section": c["metadata"].get("section", ""),
                "block_types": c["metadata"].get("block_types", []),
                "is_ocr": c["metadata"].get("is_ocr", False),
                "strategy": "structured",
                "source_file": p.name,
                "parent_id": c["metadata"].get("parent_id", ""),
                "table_json": c["metadata"].get("table_json"),
            })
    return meta


def evaluate(meta, questions, label: str):
    import rag_pipeline as rp

    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False,
                                     encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False)
        tmp = f.name
    rp.META_PATH = Path(tmp)
    store = rp.BM25Store()

    hits, mrrs = [], []
    for q in questions:
        qtext = rp.expand_query_entities(q["question"], meta)
        sel = store.search(qtext, 8)[:4]
        pos = None
        for i, it in enumerate(sel, 1):
            doc = f"{it.get('stock_code','')}_{it.get('year','')}"
            if any(doc.startswith(td) for td in q["target_docs"]):
                pos = i
                break
        hits.append(1 if pos else 0)
        mrrs.append(1 / pos if pos else 0.0)

    Path(tmp).unlink(missing_ok=True)
    return {
        "label": label, "n_chunks": len(meta), "n_q": len(questions),
        "hit": round(sum(hits) / len(hits), 4),
        "mrr": round(sum(mrrs) / len(mrrs), 4),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", type=int, default=100)
    a = ap.parse_args()

    parsed = sorted((REPO / "data" / "parsed").glob("*.json"))
    if not parsed:
        print("[CI][SKIP] 缺 data/parsed/*.json")
        return 2
    qs = [q for q in json.loads((REPO / "evaluation" / "questions_v2.json")
                               .read_text(encoding="utf-8"))["questions"]
          if q.get("target_docs")][:a.sample]
    if not qs:
        print("[CI][SKIP] 没有带 target_docs 的题")
        return 2

    rows = []
    for with_summary in (False, True):
        meta = build_meta(parsed, with_summary)
        rows.append(evaluate(meta, qs, "summary_on" if with_summary else "summary_off"))

    print(f"\n表格摘要 A/B（BM25 单路，n={len(qs)}，chunks={rows[0]['n_chunks']}）")
    print(f"{'口径':<14}{'Hit@4':>9}{'MRR@4':>9}")
    for r in rows:
        print(f"{r['label']:<14}{r['hit']:>9.4f}{r['mrr']:>9.4f}")
    off, on = rows
    print(f"{'Δ on-off':<14}{on['hit']-off['hit']:>+9.4f}{on['mrr']-off['mrr']:>+9.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
