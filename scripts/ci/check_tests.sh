#!/usr/bin/env sh
# P1-1 单测门：pytest 跑 tests/。零密钥、零网络、零磁盘产物。
set -u
CI_HOME="$(cd "$(dirname "$0")" && pwd)"; . "$CI_HOME/ci_lib.sh"
cd "$REPO_ROOT" || exit $EXIT_PREREQ

T0=$(date +%s)
ci_python
ensure_results_dir

if ! "$CI_PY" -c "import pytest" 2>/dev/null; then
  die_prereq "该解释器里没有 pytest
    安装: \"$CI_PY\" -m pip install pytest"
fi

JUNIT="$RESULTS_DIR/pytest-latest.xml"
"$CI_PY" -m pytest --junitxml="$JUNIT" > "$RESULTS_DIR/pytest-latest.log" 2>&1
RC=$?

# 从末尾几行摘"NNN passed/failed"摘要，别让"没输出"等于"没跑"
SUMMARY="$(tail -n 5 "$RESULTS_DIR/pytest-latest.log" | grep -E "[0-9]+ (passed|failed|failed.*|error)" | tail -1)"
ci_note "pytest: ${SUMMARY:-见 $RESULTS_DIR/pytest-latest.log}"

T1=$(date +%s); DUR=$((T1 - T0))

if [ $RC -ne 0 ]; then
  ci_fail "单测门未通过（退出码 $RC，$DUR s）—— 失败明细："
  grep -E "^(FAILED|ERROR)" "$RESULTS_DIR/pytest-latest.log" | head -20 >&2
  "$CI_PY" "$CI_HOME/record.py" --tier tests --status FAIL --duration "$DUR" --exit-code 1 --note "$SUMMARY" >/dev/null
  exit $EXIT_ASSERT
fi

"$CI_PY" "$CI_HOME/record.py" --tier tests --status PASS --duration "$DUR" --note "$SUMMARY" >/dev/null
ci_ok "单测门通过（$DUR s）"
exit $EXIT_PASS
