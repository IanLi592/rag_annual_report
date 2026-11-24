#!/usr/bin/env python
"""重建索引时复用旧向量，只对内容变化的 chunk 重新 embedding。

为什么需要：解析/分块的小修（如过滤页脚页码、补 heading_level）通常只改动
几十到几百个 chunk，但 `build_index.py` 会全量重算 —— 一次 20+ 分钟、约 0.35 元。
本脚本把旧索引按 `content` 精确匹配到新 chunk，命中的直接复用旧向量，
只对新增/变更内容调用 DashScope。

安全边界：
  - 只做**精确内容匹配**，不做相似匹配；内容变一个字就重新 embedding。
  - 旧索引与新索引的向量维度必须一致（都在 EMBED_DIM=1024）。
  - 输出写到独立目录，不覆盖线上索引；验证后由调用方自行切换。

用法：
  python scripts/ci/rebuild_index_reuse.py \
      --chunks data/chunks/all_semantic.json \
      --old-dir vectorstore --out-dir vectorstore/semantic_clean
退出码：0 成功 / 2 前置缺失
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

CI_DIR = Path(__file__).resolve().parent
REPO = CI_DIR.parents[1]
sys.path.insert(0, str(REPO / "src"))

import numpy as np  # noqa: E402

import build_index as bi  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunks", required=True)
    ap.add_argument("--old-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    a = ap.parse_args()

    chunks_path = Path(a.chunks)
    old_dir = Path(a.old_dir)
    out_dir = Path(a.out_dir)
    for p in (chunks_path, old_dir / "faiss_index.bin", old_dir / "faiss_meta.json"):
        if not p.exists():
            print(f"[SKIP] 缺 {p}")
            return 2
    if not os.getenv("DASHSCOPE_API_KEY"):
        print("[SKIP] 需要 DASHSCOPE_API_KEY（只有新增内容才会调用）")
        return 2

    import faiss

    chunks = json.loads(chunks_path.read_text(encoding="utf-8"))
    old_index = faiss.read_index(str(old_dir / "faiss_index.bin"))
    old_meta = json.loads((old_dir / "faiss_meta.json").read_text(encoding="utf-8"))
    if old_index.d != bi.EMBED_DIM:
        print(f"[SKIP] 旧索引维度 {old_index.d} != {bi.EMBED_DIM}，必须全量重建")
        return 2
    if old_index.ntotal != len(old_meta):
        print(f"[SKIP] 旧索引自洽失败：ntotal={old_index.ntotal} meta={len(old_meta)}")
        return 2

    # content -> 旧向量（重复内容取第一次出现，向量相同）
    old_vecs: dict[str, np.ndarray] = {}
    for i, m in enumerate(old_meta):
        old_vecs.setdefault(m["content"], old_index.reconstruct(i))

    missing: dict[str, None] = {}
    for c in chunks:
        if c["content"] not in old_vecs:
            missing.setdefault(c["content"], None)

    print(f"新 chunks={len(chunks)}  复用={len(chunks)-len(missing)}  待 embedding={len(missing)}")
    fresh: dict[str, np.ndarray] = {}
    if missing:
        client = bi.get_client()
        vecs = bi.embed_texts(client, list(missing.keys()), show_progress=True)
        fresh = dict(zip(missing.keys(), vecs))

    vectors = []
    for c in chunks:
        v = old_vecs.get(c["content"])
        if v is None:
            v = fresh[c["content"]]
        vectors.append(v)
    ref = np.stack(vectors).astype("float32")

    out_dir.mkdir(parents=True, exist_ok=True)
    index = faiss.IndexFlatIP(bi.EMBED_DIM)
    index.add(ref)
    faiss.write_index(index, str(out_dir / "faiss_index.bin"))
    (out_dir / "faiss_meta.json").write_text(
        json.dumps(bi.make_meta_list(chunks), ensure_ascii=False, indent=2),
        encoding="utf-8")
    print(f"索引已写入 {out_dir}（{index.ntotal} 条）")

    strategy = chunks[0]["metadata"].get("strategy", "") if chunks else ""
    if strategy == "structured":
        src = bi.CHUNKS_DIR / "all_structured_parents.json"
        if src.exists():
            parents = json.loads(src.read_text(encoding="utf-8"))
            (out_dir / "parents.json").write_text(
                json.dumps(parents, ensure_ascii=False, indent=2), encoding="utf-8")
            print(f"父块已写入 {out_dir / 'parents.json'}（{len(parents)} 个）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
