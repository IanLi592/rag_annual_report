#!/usr/bin/env python
"""为 RAGAS 冒烟对比挑一组固定题号（按类别分层，确定性）。

为什么不用“前 N 题”：v2 前 36 题全是 numeric，抽出来只测数字题。
这里每类各取前 N 道，且排除 refusal/should_refuse —— evaluate.py 会把
这类题从 RAGAS 打分里过滤掉，只抽到它们会得到全空的分数。

用法：
  python scripts/ci/pick_eval_ids.py --question-set v2 --per-category 2
  # 输出：3,41,77,101,...（逗号分隔，供 evaluate.py --question-ids）
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
SETS = {
    "v1": REPO / "evaluation" / "questions.json",
    "v2": REPO / "evaluation" / "questions_v2.json",
}


def pick(questions: list[dict], per_category: int) -> list[int]:
    groups: dict[str, list[int]] = {}
    for q in questions:
        qtype = q.get("type", "")
        if qtype == "should_refuse":
            continue
        key = q.get("category") or qtype or "unknown"
        groups.setdefault(key, []).append(int(q["id"]))
    picked = []
    for key in sorted(groups):
        picked.extend(sorted(groups[key])[:per_category])
    return sorted(picked)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--question-set", choices=sorted(SETS), default="v2")
    ap.add_argument("--per-category", type=int, default=2)
    a = ap.parse_args()

    if a.per_category < 1:
        print("--per-category 必须 >= 1", file=sys.stderr)
        return 2
    path = SETS[a.question_set]
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        print(f"找不到题集: {path}", file=sys.stderr)
        return 2
    ids = pick(data.get("questions") or [], a.per_category)
    if not ids:
        print("没有可用的非拒答题", file=sys.stderr)
        return 2
    print(",".join(map(str, ids)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
