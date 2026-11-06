#!/usr/bin/env sh
# P0-1 提交前静态门：py_compile + 单进程 import 冒烟。零密钥、零网络。
set -u
CI_HOME="$(cd "$(dirname "$0")" && pwd)"; . "$CI_HOME/ci_lib.sh"
cd "$REPO_ROOT" || exit $EXIT_PREREQ

T0=$(date +%s)
ci_python
ensure_results_dir

FAILED=""

# ① py_compile —— 抓 SyntaxError / 缩进错 / 模板字符串提前终止这类
for f in src/*.py evaluation/*.py; do
  [ -f "$f" ] || continue
  if ! "$CI_PY" -m py_compile "$f" 2>/dev/null; then
    FAILED="$FAILED $f"
    ci_fail "py_compile 失败: $f"
    "$CI_PY" -m py_compile "$f" 2>&1 | tail -4 >&2
  fi
done
[ -n "$FAILED" ] || ci_ok "py_compile 通过"

# ② import 冒烟 —— 单进程跑 8 个模块
#    实测分进程要 9.41s，合并成单进程 2.19s（torch/faiss 只加载一次）
#    结论必须走文件，不能走 stdout：pymupdf 会往 **stdout** 打
#    "warning: The `fitz` API is deprecated..."，用 stdout 传 JSON 会被污染解析失败
IMPORT_JSON="$RESULTS_DIR/static-imports.json"
"$CI_PY" - "$IMPORT_JSON" <<'PY'
import importlib, json, sys, time
sys.path.insert(0, "src"); sys.path.insert(0, "evaluation")
MODS = ["parse_pdf", "chunk_documents", "build_index", "download_reports",
        "rag_pipeline", "serve", "evaluate", "compare_strategies"]
bad = []
t0 = time.perf_counter()
for m in MODS:
    try:
        importlib.import_module(m)
    except Exception as e:
        bad.append({"module": m, "error": type(e).__name__, "msg": str(e)[:160]})
with open(sys.argv[1], "w", encoding="utf-8") as f:
    json.dump({"failed": bad, "n": len(MODS), "elapsed": round(time.perf_counter() - t0, 2)}, f)
PY

N_FAIL="$("$CI_PY" -c "import json,sys;print(len(json.load(open(sys.argv[1],encoding='utf-8'))['failed']))" "$IMPORT_JSON" 2>/dev/null)"
if [ -z "$N_FAIL" ]; then
  ci_fail "import 冒烟没产出结果文件 $IMPORT_JSON —— 检查解释器是否可用"
  FAILED="$FAILED import-noresult"
  N_FAIL="?"
elif [ "$N_FAIL" != "0" ]; then
  FAILED="$FAILED import"
  ci_fail "import 冒烟失败 $N_FAIL 个模块:"
  "$CI_PY" -c "
import json,sys
for b in json.load(open(sys.argv[1],encoding='utf-8'))['failed']:
    print(f\"    {b['module']}: {b['error']}: {b['msg']}\")" "$IMPORT_JSON" >&2
else
  ELAPSED="$("$CI_PY" -c "import json,sys;print(json.load(open(sys.argv[1],encoding='utf-8'))['elapsed'])" "$IMPORT_JSON")"
  ci_ok "import 冒烟 8/8 通过（单进程 ${ELAPSED}s）"
fi

T1=$(date +%s); DUR=$((T1 - T0))

if [ -n "$FAILED" ]; then
  "$CI_PY" "$CI_HOME/record.py" --tier static --status FAIL --duration "$DUR" --exit-code 1 --note "失败项:$FAILED" >/dev/null
  ci_fail "静态门未通过（$DUR s）"
  exit $EXIT_ASSERT
fi

"$CI_PY" "$CI_HOME/record.py" --tier static --status PASS --duration "$DUR" --note "py_compile + import 冒烟" >/dev/null
ci_ok "静态门通过（$DUR s）"
exit $EXIT_PASS
