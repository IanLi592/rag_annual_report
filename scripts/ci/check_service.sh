#!/usr/bin/env sh
# P1-2 服务启停门（CD 判据）：拉起 uvicorn → 轮询 /health 到 200 → 关停。
#
# 为什么"没有密钥"必须当成前置缺失（退出码 2）而不是失败：
# 实测 src/rag_pipeline.py:309 的 __init__ 第一行就调 get_client()，
# 无 DASHSCOPE_API_KEY 时 uvicorn 直接退出码 3、/health 连接被拒（http=000）。
# 那是环境没配好，不是代码回归 —— 混成"失败"会让人去查错方向。
set -u
CI_HOME="$(cd "$(dirname "$0")" && pwd)"; . "$CI_HOME/ci_lib.sh"
cd "$REPO_ROOT" || exit $EXIT_PREREQ

PORT=8011
TIMEOUT_S=90
while [ $# -gt 0 ]; do
  case "$1" in
    --port) PORT="$2"; shift 2 ;;
    --timeout) TIMEOUT_S="$2"; shift 2 ;;
    *) ci_fail "未知参数: $1"; exit $EXIT_PREREQ ;;
  esac
done

T0=$(date +%s)
ci_python
ensure_results_dir
load_dotenv

if ! ci_key_filled "${DASHSCOPE_API_KEY:-}"; then
  ci_skip "DASHSCOPE_API_KEY 没填，或填的是占位示例值 —— 两种都不能算验过："
  ci_skip "  · 空值：rag_pipeline.py:309 的 get_client() 直接抛 EnvironmentError，服务起不来"
  ci_skip "  · 占位值：客户端只是构造、不发请求，服务照样能起来 /health 照样 200 —— 那是假绿"
  ci_skip "修法：项目根 .env 里写真实 key（该文件已被 .gitignore 忽略），见 docs P1-3"
  "$CI_PY" "$CI_HOME/record.py" --tier service --status SKIP --duration 0 --note "DASHSCOPE_API_KEY 未填或为占位值" >/dev/null
  exit $EXIT_PREREQ
fi

LOG="$RESULTS_DIR/service-latest.log"
ci_note "启动 uvicorn serve:app 端口 $PORT（冷启动含索引加载 + BM25 分词 10352 条）"

# 注意两个 Windows 坑：
#  ① 不能用 ( cd src && uvicorn ) & —— $! 是子 shell 的 pid，kill 掉 shell
#     后 python.exe 仍占着端口（实测泄漏过一次 PID 51388）。改成 PYTHONPATH 注入。
#  ② Git Bash 会把 /F /PID 这类参数当路径转换，兜底调 PowerShell 最稳。
PYTHONPATH="$REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}" \
  "$CI_PY" -m uvicorn serve:app --host 127.0.0.1 --port "$PORT" >"$LOG" 2>&1 &
SRV=$!

port_pid() {
  netstat -ano 2>/dev/null | grep ":$PORT " | grep LISTENING | awk '{print $NF}' | head -1
}

stop_server() {
  [ -n "${SRV:-}" ] || return 0
  kill "$SRV" 2>/dev/null || true
  i=0
  while [ "$i" -lt 5 ]; do
    LEFT="$(port_pid)"
    [ -z "$LEFT" ] && return 0
    sleep 1; i=$((i + 1))
  done
  ci_note "kill 没带走 python.exe（PID $LEFT），改用 PowerShell 兜底"
  powershell -NoProfile -Command "Stop-Process -Id $LEFT -Force" >/dev/null 2>&1 || true
  sleep 1
  LEFT2="$(port_pid)"
  [ -z "$LEFT2" ] || ci_fail "端口 $PORT 仍被 PID $LEFT2 占用，请手动结束"
}

trap stop_server EXIT INT TERM

WAITED=0
READY=""
while [ "$WAITED" -lt "$TIMEOUT_S" ]; do
  CODE="$(curl -s -o /dev/null -m 2 -w '%{http_code}' "http://127.0.0.1:$PORT/health" 2>/dev/null || true)"
  if [ "$CODE" = "200" ]; then READY=1; HEALTH_BODY="$(curl -s -m 2 "http://127.0.0.1:$PORT/health" 2>/dev/null || true)"; break; fi
  # 进程已经退了就不用再等
  kill -0 "$SRV" 2>/dev/null || { READY=0; break; }
  sleep 2; WAITED=$((WAITED + 2))
done

T1=$(date +%s); DUR=$((T1 - T0))
stop_server

if [ -n "$READY" ]; then
  "$CI_PY" "$CI_HOME/record.py" --tier service --status PASS --duration "$DUR" --note "冷启动 ${DUR}s" >/dev/null
  ci_ok "服务门通过：/health 返回 ${HEALTH_BODY:-?}，冷启动 ${DUR}s（此数回填 docs §8 待实测项）"
  exit $EXIT_PASS
fi

ci_fail "服务没能在 ${TIMEOUT_S}s 内就绪（退出码 $DUR 秒观测）"
ci_fail "服务日志尾部："
tail -12 "$LOG" >&2
"$CI_PY" "$CI_HOME/record.py" --tier service --status FAIL --duration "$DUR" --exit-code 1 --note "超时或启动崩溃" >/dev/null
exit $EXIT_ASSERT
