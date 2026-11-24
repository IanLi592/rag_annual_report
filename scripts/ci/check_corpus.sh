#!/usr/bin/env sh
# P0-1.5 语料门：零网络校验 data/raw_pdf 与 manifest.json，排在索引门之前。
#
# 为什么必须排在索引门前面：check_index.py 断言的是"索引与 chunks 自洽"，
# 语料缺了两家公司、或者装着一次断流留下的半截 PDF 时，它照样能自洽地绿 ——
# 上游脏、下游对得上，不等于下游是对的。这一门不到 1 秒、不联网、不要密钥，
# 却是全套里唯一直接看 PDF 文件本身的一道。
#
# 退出码沿用 ci_lib.sh 的三档契约：0 通过 / 1 语料真有缺陷 / 2 还没下载过
# （新克隆的正常状态，不能被当成代码回归）。
set -u
CI_HOME="$(cd "$(dirname "$0")" && pwd)"; . "$CI_HOME/ci_lib.sh"
cd "$REPO_ROOT" || exit $EXIT_PREREQ

T0=$(date +%s)
ci_python
ensure_results_dir

"$CI_PY" "$REPO_ROOT/src/download_reports.py" --verify
RC=$?

T1=$(date +%s); DUR=$((T1 - T0))
NOTE="manifest 与磁盘对账 + PDF 结构完整 + 5 公司 x 3 年齐全 + 孤儿 PDF 与残留 .part"

case "$RC" in
  0) "$CI_PY" "$CI_HOME/record.py" --tier corpus --status PASS --duration "$DUR" --note "$NOTE" >/dev/null
     ci_ok "语料门通过（$DUR s）"; exit $EXIT_PASS ;;
  2) "$CI_PY" "$CI_HOME/record.py" --tier corpus --status SKIP --duration "$DUR" --note "data/ 尚未下载" >/dev/null
     ci_skip "语料门跳过：语料产物不存在（新克隆正常，先跑 python src/download_reports.py）"
     exit $EXIT_PREREQ ;;
  *) "$CI_PY" "$CI_HOME/record.py" --tier corpus --status FAIL --duration "$DUR" --exit-code 1 --note "$NOTE" >/dev/null
     ci_fail "语料门未通过（$DUR s）—— 上面每条「语料缺陷」都要单独处置，别整体重下了事"
     exit $EXIT_ASSERT ;;
esac
