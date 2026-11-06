#!/usr/bin/env python
"""P0-2 索引一致性门。读本地现成产物，零密钥、零网络。

这一档云端做不了（runner 没有 56 MB 索引 + 147 MB 语料），是本地 CI 的独有价值：
拦的是"重建索引时静默少了一半 chunk / 维度写错 / 元信息字段改名"这类真实事故。

规约（docs/cicd_requirements.md §2 P0-2）：**禁止裸 .get() 取值就断言**。
本会话实测踩过：faiss_meta 里没有 company_name / block_type，.get() 返回 None
且不报错，断言静默通过。所以这里先做字段存在性校验，缺失即 FAIL。

用法：check_index.py [--fast] [--allow-drift] [--probe "查询串"]
退出码：0 通过 / 1 断言失败 / 2 前置缺失
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

CI_DIR = Path(__file__).resolve().parent
REPO = CI_DIR.parents[1]
EXIT_PASS, EXIT_ASSERT, EXIT_PREREQ = 0, 1, 2

PROBE_QUERY_DEFAULT = "贵州茅台 2021 年营业收入"


class Gate:
    def __init__(self) -> None:
        self.records: list[dict] = []

    def add(self, name: str, expect, actual, ok: bool, note: str = "") -> bool:
        self.records.append(
            {"name": name, "expect": str(expect), "actual": str(actual), "pass": bool(ok), "note": note}
        )
        tag = "PASS" if ok else "FAIL"
        line = f"[CI][{tag}] {name}: 期望={expect} 实际={actual}"
        print(line + (f"  ({note})" if note else ""))
        return bool(ok)


def load_baseline() -> dict:
    p = CI_DIR / "baseline.json"
    if not p.exists():
        print(f"[CI][FAIL] 缺基线文件: {p}")
        sys.exit(EXIT_PREREQ)
    return json.loads(p.read_text(encoding="utf-8"))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fast", action="store_true", help="只做元数据比对，不建 BM25（约 0.15s vs 12s）")
    ap.add_argument("--allow-drift", action="store_true", help="条数/维度与基线不符时降级为警告（自洽性仍强制）")
    ap.add_argument("--probe", default=PROBE_QUERY_DEFAULT)
    ap.add_argument("--emit-json", default="", help="把断言明细写成 JSON（供 record.py 用）")
    a = ap.parse_args()

    b = load_baseline()
    g = Gate()
    t_all = time.perf_counter()

    idx_path = REPO / b["artifacts"]["faiss_index"]
    meta_path = REPO / b["artifacts"]["faiss_meta"]

    # ── 前置检查：产物在不在 ────────────────────────────────────────────────
    missing = [str(p.relative_to(REPO)) for p in (idx_path, meta_path) if not p.exists()]
    if missing:
        print(f"[CI][SKIP] 缺产物: {', '.join(missing)}")
        print("  这不算失败也不算通过（§3.2）。重建： python src/build_index.py（需 DASHSCOPE_API_KEY，约 5-10 分钟）")
        return EXIT_PREREQ

    import faiss  # noqa: E402

    t0 = time.perf_counter()
    index = faiss.read_index(str(idx_path))
    t_faiss = time.perf_counter() - t0

    t0 = time.perf_counter()
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    t_meta = time.perf_counter() - t0

    ntotal, dim, nmeta = int(index.ntotal), int(index.d), len(meta)

    ok = True
    ok &= g.add("索引条数自洽(faiss==meta)", nmeta, ntotal, ntotal == nmeta)
    ok &= g.add(
        "条数与基线一致",
        b["chunks_total"],
        ntotal,
        ntotal == b["chunks_total"],
        "--allow-drift 生效" if (a.allow_drift and ntotal != b["chunks_total"]) else "",
    )
    ok &= g.add("向量维度与基线一致", b["embedding_dim"], dim, dim == b["embedding_dim"])

    # ── 字段存在性（先于任何取值断言）──────────────────────────────────────
    required = list(b["required_meta_fields"])
    key_sets = [set(m.keys()) for m in meta[:1]]
    absent = [f for f in required if f not in key_sets[0]]
    ok &= g.add("meta 首条含全部必需字段", required, f"缺失={absent}", not absent)

    if not absent:
        no_id = sum(1 for m in meta if not m.get("chunk_id"))
        ok &= g.add("chunk_id 非空", 0, no_id, no_id == 0)
        empty_body = sum(1 for m in meta if not (m.get("content") or "").strip())
        ok &= g.add("content 非空", 0, empty_body, empty_body == 0)

    if a.allow_drift:
        # 漂移模式下，条数/维度不符只留痕不改结论
        for r in g.records:
            if "与基线一致" in r["name"] and not r["pass"]:
                r["note"] = "allow-drift: 降级为警告"

    # ── 评测集完整性 ────────────────────────────────────────────────────────
    qpath = REPO / "evaluation" / "questions.json"
    if qpath.exists():
        qs = json.loads(qpath.read_text(encoding="utf-8"))["questions"]
        ok &= g.add("评测题数", b["questions_total"], len(qs), len(qs) == b["questions_total"])
        dist: dict[str, int] = {}
        for q in qs:
            dist[q["type"]] = dist.get(q["type"], 0) + 1
        ok &= g.add("题型分布", b["question_types"], dist, dist == b["question_types"])

    # ── BM25 真实检索（--fast 跳过）────────────────────────────────────────
    bm25_s = 0.0
    if not a.fast:
        sys.path.insert(0, str(REPO / "src"))
        import rag_pipeline as rp  # noqa: E402

        t0 = time.perf_counter()
        store = rp.BM25Store()
        bm25_s = time.perf_counter() - t0
        t1 = time.perf_counter()
        hits = store.search(a.probe, 10)
        search_s = time.perf_counter() - t1
        ok &= g.add("BM25 探针命中数>0", ">0", len(hits), len(hits) > 0, f"probe={a.probe!r}")
        if hits:
            # 显式校验字段存在，再取值 —— 不接受 None 蒙混过关
            ok &= g.add("Top1 带 bm25_score 字段", True, "bm25_score" in hits[0], "bm25_score" in hits[0])
            if "bm25_score" in hits[0]:
                ok &= g.add("Top1 bm25_score 为正", ">0", round(float(hits[0]["bm25_score"]), 4), float(hits[0]["bm25_score"]) > 0)
        print(f"[CI] 计时: faiss {t_faiss:.2f}s | meta {t_meta:.2f}s | BM25 建库 {bm25_s:.2f}s | 检索 {search_s:.2f}s")

        # ── 过滤语义守卫（真实语料级）────────────────────────────────────
        # 背景：BM25Store.search 原先不收 filter_meta，导致 year 过滤只约束一半候选。
        # 单测用的是 6 条合成语料，这里在 10352 条真实块上再验一次。
        fprobe = "贵州茅台 营业收入 亿元"
        unf_ids = {h["chunk_id"] for h in store.search(fprobe, 10)}
        unf_years = {h["year"] for h in store.search(fprobe, 10)}
        f_hits = store.search(fprobe, 10, {"year": "2021"})
        f_years = {h.get("year") for h in f_hits}
        ok &= g.add("真实语料上按 year=2021 过滤后只剩 2021", {"2021"}, f_years,
                    bool(f_hits) and f_years == {"2021"})
        # 对照：不加过滤时应混入多个年份，否则上面那条是空断言
        ok &= g.add("对照-无过滤确实跨年(证明过滤起了作用)", ">1 个年份", len(unf_years),
                    len(unf_years) > 1 and unf_ids != {h["chunk_id"] for h in f_hits})
    else:
        print(f"[CI] 计时: faiss {t_faiss:.2f}s | meta {t_meta:.2f}s | --fast 跳过 BM25 建库")

    dur = int(round(time.perf_counter() - t_all))
    if a.emit_json:
        Path(a.emit_json).write_text(
            json.dumps({"records": g.records, "duration_s": dur, "bm25_build_s": round(bm25_s, 2)}, ensure_ascii=False),
            encoding="utf-8",
        )

    n_fail = sum(1 for r in g.records if not r["pass"])
    if not ok:
        print(f"[CI][FAIL] 索引门未通过：{n_fail}/{len(g.records)} 条断言红（{dur}s）")
        print("  若你确实改了语料/重建了索引，用 --allow-drift 复核，或更新 scripts/ci/baseline.json")
        return EXIT_ASSERT
    print(f"[CI][PASS] 索引门通过：{len(g.records)} 条断言全绿（{dur}s）")
    return EXIT_PASS


if __name__ == "__main__":
    sys.exit(main())
