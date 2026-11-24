#!/usr/bin/env python
"""把一次 CI 阶段的结果并进 evaluation/results/ci-latest.json（P1-4）。

用法：
  python record.py --tier static --status PASS --duration 12 --note "8 文件"
  python record.py --tier index --status FAIL --duration 3 --assertions '[{"name":...}]'

只依赖标准库。写入失败不影响检查结论（CI 记录不是被守护的对象本身）。
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
RESULTS = REPO / "evaluation" / "results"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tier", required=True)
    ap.add_argument("--status", required=True, choices=["PASS", "FAIL", "SKIP"])
    ap.add_argument("--duration", type=int, default=0)
    ap.add_argument("--exit-code", type=int, default=0)
    ap.add_argument("--note", default="")
    ap.add_argument("--assertions", default="[]", help="JSON 数组字符串")
    a = ap.parse_args()

    try:
        assertions = json.loads(a.assertions)
    except json.JSONDecodeError as e:
        print(f"[CI][WARN] --assertions 不是合法 JSON（{e}），记为空", file=sys.stderr)
        assertions = []

    RESULTS.mkdir(parents=True, exist_ok=True)
    path = RESULTS / "ci-latest.json"

    doc = {"tiers": {}, "history": []}
    if path.exists():
        try:
            doc = json.loads(path.read_text(encoding="utf-8"))
        except Exception as e:  # 坏文件不能让 CI 挂
            print(f"[CI][WARN] ci-latest.json 解析失败（{e}），重建", file=sys.stderr)

    doc.setdefault("tiers", {})
    doc.setdefault("history", [])
    doc["updated_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    doc["git_sha"] = os.popen("git rev-parse --short HEAD").read().strip() or "unknown"
    doc["interpreter"] = sys.executable
    doc["tiers"][a.tier] = {
        "status": a.status,
        "duration_s": a.duration,
        "exit_code": a.exit_code,
        "note": a.note,
        "assertions": assertions,
        "recorded_at": doc["updated_at"],
    }
    doc["history"] = (doc["history"] + [{"tier": a.tier, "status": a.status, "at": doc["updated_at"]}])[-200:]

    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(doc, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, path)
    print(f"[CI] 已记录 tier={a.tier} status={a.status} → {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
