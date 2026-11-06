#!/usr/bin/env sh
# P0-2 索引门的外壳：调用 check_index.py，把结论并进 CI 记录。
set -u
CI_HOME="$(cd "$(dirname "$0")" && pwd)"; . "$CI_HOME/ci_lib.sh"
cd "$REPO_ROOT" || exit $EXIT_PREREQ

T0=$(date +%s)
ci_python
ensure_results_dir

ASSERT_JSON="$RESULTS_DIR/index-assertions.json"
"$CI_PY" "$CI_HOME/check_index.py" --emit-json "$ASSERT_JSON" "$@"
RC=$?

T1=$(date +%s); DUR=$((T1 - T0))
NOTE="索引一致性 + 评测集完整性 + BM25 探针"
[ "${1:-}" = "--fast" ] && NOTE="$NOTE (--fast，未建 BM25)"

case "$RC" in
  0) "$CI_PY" "$CI_HOME/record.py" --tier index --status PASS --duration "$DUR" --note "$NOTE" >/dev/null
     ci_ok "索引门通过（$DUR s）"; exit $EXIT_PASS ;;
  2) "$CI_PY" "$CI_HOME/record.py" --tier index --status SKIP --duration "$DUR" --note "缺产物，见 §3.2" >/dev/null
     ci_skip "索引门跳过：产物缺失"; exit $EXIT_PREREQ ;;
  *) "$CI_PY" "$CI_HOME/record.py" --tier index --status FAIL --duration "$DUR" --exit-code 1 --note "$NOTE" >/dev/null
     ci_fail "索引门未通过（$DUR s）—— 期望 vs 实际见上方 [CI][FAIL] 行"
     exit $EXIT_ASSERT ;;
esac
