#!/usr/bin/env sh
# 一键跑全套。退出码取"最严重"的那个：断言失败(1) > 前置缺失(2) > 通过(0)。
# 用法: run_all.sh [--tier p0|p0p1] [--fast-index]
set -u
CI_HOME="$(cd "$(dirname "$0")" && pwd)"; . "$CI_HOME/ci_lib.sh"
cd "$REPO_ROOT" || exit $EXIT_PREREQ

TIER="p0p1"
FAST=""
while [ $# -gt 0 ]; do
  case "$1" in
    --tier) TIER="$2"; shift 2 ;;
    --fast-index) FAST="--fast"; shift ;;
    *) ci_fail "未知参数: $1"; exit $EXIT_PREREQ ;;
  esac
done

worst=$EXIT_PASS
worse() {
  # 严重度排序：断言失败(1) > 前置缺失(2) > 通过(0)
  if [ "$1" -eq "$EXIT_ASSERT" ]; then
    worst=$EXIT_ASSERT
  elif [ "$1" -eq "$EXIT_PREREQ" ] && [ "$worst" -ne "$EXIT_ASSERT" ]; then
    worst=$EXIT_PREREQ
  fi
}

run() {
  ci_note "── $1 开始 ──"
  shift
  "$@"
  rc=$?
  case $rc in
    0) ci_note "  → 退出码 0（通过）" ;;
    2) ci_note "  → 退出码 2（前置缺失，非代码回归）" ;;
    *) ci_fail "  → 退出码 $rc（断言失败）" ;;
  esac
  worse $rc
  return 0
}

T0=$(date +%s)
ci_python
ci_note "解释器: $CI_PY"
ci_note "档位: $TIER   仓库: $REPO_ROOT   sha: $(ci_sha)"

run "P0-1 静态门"   "$CI_HOME/check_static.sh"
case "$TIER" in
  p0|p0p1|all) run "P0-2 索引门" "$CI_HOME/check_index.sh" $FAST ;;
esac
case "$TIER" in
  p0p1|all)
    run "P1-1 单测门" "$CI_HOME/check_tests.sh"
    run "P1-2 服务门" "$CI_HOME/check_service.sh"
    ;;
esac
T1=$(date +%s)

echo
if [ $worst -eq $EXIT_PASS ]; then
  ci_ok "run_all 全绿，总耗时 $((T1 - T0)) s"
elif [ $worst -eq $EXIT_PREREQ ]; then
  ci_skip "run_all 有阶段因前置缺失被跳过，总耗时 $((T1 - T0)) s —— 结论不完整，别当通过"
else
  ci_fail "run_all 有断言失败，总耗时 $((T1 - T0)) s"
fi
exit $worst
