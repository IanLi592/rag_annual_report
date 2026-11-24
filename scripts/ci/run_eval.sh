#!/usr/bin/env sh
# 花钱档的正式入口：载 .env → 校验两个 key → 跑 evaluation/evaluate.py
#
# 为什么需要这个脚本：evaluate.py / compare_strategies.py / build_index.py
# 自己**不读 .env**，只认环境变量。CI 的 shell 载入器（ci_lib.sh）覆盖不到
# 直接 `python evaluation/evaluate.py` 这种手动跑法，会撞
# OSError: 请设置环境变量 DASHSCOPE_API_KEY（本会话实测撞到过一次）。
#
# 用法：
#   sh scripts/ci/run_eval.sh --question-ids 1,2      # 冒烟：2 题，实测 38s
#   sh scripts/ci/run_eval.sh                          # 全量 20 题（会明显花钱+花时间，先想清楚）
#   sh scripts/ci/run_eval.sh --skip-ragas             # 只出题不打 ragas 分
set -u
CI_HOME="$(cd "$(dirname "$0")" && pwd)"; . "$CI_HOME/ci_lib.sh"
cd "$REPO_ROOT" || exit $EXIT_PREREQ

ci_python
load_dotenv

if ! ci_key_filled "${DASHSCOPE_API_KEY:-}"; then
  ci_fail "DASHSCOPE_API_KEY 没填或仍是占位值 —— 检索要 embedding，跑不了"
  ci_fail "填到项目根 .env（已被 .gitignore 忽略）"
  exit $EXIT_PREREQ
fi
# 裁判 key 按供应商解析（默认 MiMo Token Plan）。
# --skip-ragas 时根本不需要裁判，别拦着人出题。
PROVIDER="${RAGAS_LLM_PROVIDER:-mimo}"
KEY_ENV="$(judge_key_env)"
if [ -z "$KEY_ENV" ]; then
  ci_fail "未知的 RAGAS_LLM_PROVIDER=$PROVIDER —— 可选 mimo | deepseek"
  exit $EXIT_PREREQ
fi

case "$*" in
  *--skip-ragas*)
    ci_note "--skip-ragas：只出题不打分，不需要 $KEY_ENV"
    ;;
  *)
    eval "KEY_VAL=\${$KEY_ENV:-}"
    if ! ci_key_filled "$KEY_VAL"; then
      ci_fail "裁判走 $PROVIDER，但 $KEY_ENV 没填或仍是占位值"
      ci_fail "填到项目根 .env；或换档：RAGAS_LLM_PROVIDER=deepseek sh scripts/ci/run_eval.sh"
      ci_fail "只想出题不打分：加 --skip-ragas"
      exit $EXIT_PREREQ
    fi
    ci_note "裁判：$PROVIDER（$KEY_ENV 已就绪）"
    # 连通性探针：离线预检只能验 key 存在，验不到"真能返回可解析内容"。
    # 全量 17 题在 MiMo 上要 14-18 分钟，那种失败必须提前拦住。
    ci_note "裁判连通性探针（一次真实小请求，约 25 token）"
    "$CI_PY" "$CI_HOME/check_judge.py"
    jrc=$?
    if [ $jrc -ne 0 ]; then
      ci_fail "探针未通过（退出码 $jrc）—— 已阻止评估，别白跑十几分钟"
      exit $jrc
    fi
    ;;
esac

ensure_results_dir
ci_note "跑 evaluation/evaluate.py $* （会产生 DashScope 调用；裁判走 $PROVIDER）"
T0=$(date +%s)
"$CI_PY" evaluation/evaluate.py "$@"
RC=$?
T1=$(date +%s)
[ $RC -eq 0 ] && ci_ok "评估完成（$((T1 - T0)) s），产物在 evaluation/results/" \
              || ci_fail "评估失败（退出码 $RC，$((T1 - T0)) s）"
exit $RC
