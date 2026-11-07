#!/usr/bin/env sh
# RAGAS 基线对比门（按需 / 改动触发）。
#
# 做两件事：
#   1. 用固定题号跑 evaluation/evaluate.py（v2 默认按类别分层抽 2 题/类，共 6 题）
#   2. 与 evaluation/results/ragas_baseline.json 比四项指标，输出 diff + 逐题回退
#
# 为什么默认只跑 6 题：完整 125 题 RAGAS 约 30~60 分钟、几十元；
# 每次 Chunk/Prompt 改动都全量跑不现实。6 题是冒烟门，全量用 --per-category 12
# 或直接 python evaluation/evaluate.py。
#
# 用法：
#   sh scripts/ci/check_eval_compare.sh                 # 有基线：跑同题号并对比
#   sh scripts/ci/check_eval_compare.sh --update-baseline  # 重跑并覆盖基线
#   sh scripts/ci/check_eval_compare.sh --per-category 4   # 12 题
#   sh scripts/ci/check_eval_compare.sh --strict           # 指标回退>3pp 时退出 1
#   sh scripts/ci/check_eval_compare.sh --question-ids 1,37,73
#
# 首次运行没有基线时会自动建立基线（这次不可能判回归）。
set -u
CI_HOME="$(cd "$(dirname "$0")" && pwd)"; . "$CI_HOME/ci_lib.sh"
cd "$REPO_ROOT" || exit $EXIT_PREREQ

PER_CATEGORY=2
TOLERANCE=0.03
QUESTION_SET=v2
QUESTION_IDS=""
UPDATE=0
STRICT=0

while [ $# -gt 0 ]; do
  case "$1" in
    --per-category) PER_CATEGORY="$2"; shift 2 ;;
    --tolerance)    TOLERANCE="$2"; shift 2 ;;
    --question-set) QUESTION_SET="$2"; shift 2 ;;
    --question-ids) QUESTION_IDS="$2"; shift 2 ;;
    --update-baseline) UPDATE=1; shift ;;
    --strict) STRICT=1; shift ;;
    *) ci_fail "未知参数: $1"; exit $EXIT_PREREQ ;;
  esac
done

ci_python
load_dotenv
ensure_results_dir

# 基线放在 scripts/ci/ 下（受版本控制），不放在 evaluation/results/（被 .gitignore）。
# 这样换机器/新 clone 也能直接比对；基线只保留题号与逐题 RAGAS 分数，不存答案和上下文。
BASELINE_FILE="${RAGAS_BASELINE:-$CI_HOME/ragas_baseline.json}"

if ! ci_key_filled "${DASHSCOPE_API_KEY:-}"; then
  ci_skip "DASHSCOPE_API_KEY 未填或为占位值，RAGAS 对比跳过"
  "$CI_PY" "$CI_HOME/record.py" --tier eval --status SKIP --duration 0 \
    --note "DASHSCOPE_API_KEY 未填" >/dev/null 2>&1
  exit $EXIT_PREREQ
fi
KEY_ENV="$(judge_key_env)"
if [ -z "$KEY_ENV" ]; then
  ci_skip "未知 RAGAS_LLM_PROVIDER=${RAGAS_LLM_PROVIDER:-}（可选 mimo | deepseek）"
  "$CI_PY" "$CI_HOME/record.py" --tier eval --status SKIP --duration 0 \
    --note "未知裁判供应商" >/dev/null 2>&1
  exit $EXIT_PREREQ
fi
eval "JUDGE_KEY=\${$KEY_ENV:-}"
if ! ci_key_filled "$JUDGE_KEY"; then
  ci_skip "裁判 $KEY_ENV 未填或为占位值，RAGAS 对比跳过"
  "$CI_PY" "$CI_HOME/record.py" --tier eval --status SKIP --duration 0 \
    --note "$KEY_ENV 未填" >/dev/null 2>&1
  exit $EXIT_PREREQ
fi

# 题号选择：显式 > 基线记录 > 分层抽样。
# 对比必须同题号，否则是在比两套尺子。
if [ -z "$QUESTION_IDS" ] && [ "$UPDATE" -eq 0 ] && [ -f "$BASELINE_FILE" ]; then
  QUESTION_IDS="$("$CI_PY" - "$BASELINE_FILE" <<'PY'
import json, sys
d = json.load(open(sys.argv[1], encoding="utf-8"))
ids = d.get("question_ids")
if not ids:
    ids = sorted({str(r.get("question_id")) for r in d.get("raw_results", []) if r.get("question_id") is not None})
print(",".join(map(str, ids)))
PY
)"
  QUESTION_SET="$("$CI_PY" - "$BASELINE_FILE" "$QUESTION_SET" <<'PY'
import json, sys
d = json.load(open(sys.argv[1], encoding="utf-8"))
print(d.get("question_set") or sys.argv[2])
PY
)"
fi
if [ -z "$QUESTION_IDS" ]; then
  QUESTION_IDS="$("$CI_PY" "$CI_HOME/pick_eval_ids.py" --question-set "$QUESTION_SET" \
      --per-category "$PER_CATEGORY")" || exit $EXIT_PREREQ
fi
[ -n "$QUESTION_IDS" ] || { ci_skip "没选出题号"; exit $EXIT_PREREQ; }

# 首次运行自动建基线：没有基线就没有“对比”，但至少先把尺子立起来。
if [ ! -f "$BASELINE_FILE" ] && [ "$UPDATE" -eq 0 ]; then
  ci_note "没有基线 $BASELINE_FILE，本次自动建立基线"
  UPDATE=1
fi

MARKER="$RESULTS_DIR/.ragas_eval_marker.$$"
: > "$MARKER"
trap 'rm -f "$MARKER"' EXIT INT TERM

ci_note "RAGAS 题号: $QUESTION_IDS（question-set=$QUESTION_SET）"
ci_note "裁判: ${RAGAS_LLM_PROVIDER:-mimo} / $KEY_ENV"
T0=$(date +%s)
"$CI_PY" evaluation/evaluate.py --question-set "$QUESTION_SET" --question-ids "$QUESTION_IDS"
RC=$?
T1=$(date +%s); DUR=$((T1 - T0))
if [ "$RC" -ne 0 ]; then
  ci_fail "evaluate.py 失败（退出码 $RC，${DUR}s）"
  "$CI_PY" "$CI_HOME/record.py" --tier eval --status FAIL --duration "$DUR" \
    --exit-code "$RC" --note "evaluate.py 失败" >/dev/null 2>&1
  exit "$RC"
fi

LATEST="$("$CI_PY" - "$MARKER" <<'PY'
import pathlib, sys
m = pathlib.Path(sys.argv[1]).stat().st_mtime
ps = [p for p in pathlib.Path("evaluation/results").glob("native_*.json")
      if p.stat().st_mtime >= m - 1]
print(max(ps, key=lambda p: p.stat().st_mtime) if ps else "")
PY
)"
if [ -z "$LATEST" ] || [ ! -f "$LATEST" ]; then
  ci_fail "evaluate.py 退出 0，但没找到本次产物"
  "$CI_PY" "$CI_HOME/record.py" --tier eval --status FAIL --duration "$DUR" \
    --exit-code 1 --note "缺少评估产物" >/dev/null 2>&1
  exit $EXIT_ASSERT
fi

if [ "$UPDATE" -eq 1 ]; then
  "$CI_PY" - "$LATEST" "$BASELINE_FILE" "$QUESTION_IDS" "$QUESTION_SET" <<'PY'
import json, sys, time
src, dst, ids, setname = sys.argv[1:5]
d = json.load(open(src, encoding="utf-8"))
# 只留对比需要的字段：答案/上下文体积大，且可能含模型输出，不提交。
d["raw_results"] = [
    {"question_id": r.get("question_id"),
     "question_type": r.get("question_type"),
     "ragas": r.get("ragas") or {}}
    for r in d.get("raw_results", [])
]
d["question_ids"] = [int(x) for x in ids.split(",") if x.strip()]
d["question_set"] = setname
d["created_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
json.dump(d, open(dst, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
print(f"[CI] 基线已更新 → {dst}")
PY
  ci_ok "RAGAS 基线已建立（${DUR}s）"
  "$CI_PY" "$CI_HOME/record.py" --tier eval --status PASS --duration "$DUR" \
    --note "baseline updated: $QUESTION_IDS" >/dev/null 2>&1
  exit $EXIT_PASS
fi

STRICT_FLAG=""
[ "$STRICT" -eq 1 ] && STRICT_FLAG="--strict"
"$CI_PY" "$CI_HOME/compare_ragas.py" --baseline "$BASELINE_FILE" --latest "$LATEST" \
  --tolerance "$TOLERANCE" $STRICT_FLAG
CMP_RC=$?
if [ "$CMP_RC" -eq 0 ]; then
  "$CI_PY" "$CI_HOME/record.py" --tier eval --status PASS --duration "$DUR" \
    --note "RAGAS compared: $QUESTION_IDS" >/dev/null 2>&1
  exit $EXIT_PASS
elif [ "$CMP_RC" -eq 1 ]; then
  "$CI_PY" "$CI_HOME/record.py" --tier eval --status FAIL --duration "$DUR" \
    --exit-code 1 --note "RAGAS regression: $QUESTION_IDS" >/dev/null 2>&1
  exit $EXIT_ASSERT
else
  ci_skip "RAGAS 对比无法进行（缺少产物或题号不匹配）"
  "$CI_PY" "$CI_HOME/record.py" --tier eval --status SKIP --duration "$DUR" \
    --note "compare skipped" >/dev/null 2>&1
  exit $EXIT_PREREQ
fi
