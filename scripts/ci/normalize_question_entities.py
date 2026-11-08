#!/usr/bin/env python
"""把题集中的股票代码题干改成公司简称（只改 `question`，不动 ground_truth/evidence）。

为什么：125 题里有 100 题写成「000858在2021年的营业收入是多少？」。真实用户
不会这样问，而且 chunk 正文只有「五粮液」没有代码，会让检索指标被题干措辞
人为压低。代码侧 `expand_query_entities` 仍保留（用户真输代码时兜底），
但评测集应当用自然问法。

用法：
  python scripts/ci/normalize_question_entities.py            # dry-run
  python scripts/ci/normalize_question_entities.py --write    # 写回
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

CI_DIR = Path(__file__).resolve().parent
REPO = CI_DIR.parents[1]
QPATH = REPO / "evaluation" / "questions_v2.json"
MANIFEST = REPO / "data" / "manifest.json"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true")
    a = ap.parse_args()

    if not QPATH.exists() or not MANIFEST.exists():
        print(f"[SKIP] 缺 {QPATH} 或 {MANIFEST}")
        return 2
    doc = json.loads(QPATH.read_text(encoding="utf-8"))
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    names = {m["stock_code"]: m.get("company_name", m["stock_code"]) for m in manifest}

    changed = 0
    for q in doc.get("questions", []):
        text = q["question"]
        new = text
        for code, name in names.items():
            if code not in new:
                continue
            new = new.replace(code, "" if name in new else name, 1)
        new = re.sub(r"\s{2,}", " ", new).strip()
        if new != text:
            changed += 1
            if changed <= 5:
                print(f"  {q['id']}: {text} → {new}")
            q["question"] = new

    print(f"共改写 {changed} 题")
    if a.write and changed:
        QPATH.write_text(json.dumps(doc, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"已写回 {QPATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
