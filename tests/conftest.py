"""pytest 路径配置：让 tests/ 能直接 import src/、evaluation/ 与 scripts/ci/ 里的模块。

这些模块不是包（没有 __init__.py），靠 sys.path 注入。
和 src/serve.py:37 加载 rag_pipeline 的方式一致，不改生产代码。
scripts/ci 也加进来，是为了能离线守卫 check_judge.py 的失败分类逻辑。
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
for sub in ("src", "evaluation", "scripts/ci"):
    p = str(REPO / sub)
    if p not in sys.path:
        sys.path.insert(0, p)
