# ci_lib.sh —— 被 scripts/ci/check_*.sh 与 scripts/git-hooks/* 共用。
# 只用 POSIX sh 语法：Git 的钩子由 sh.exe 执行，不能假设 bash 特性。
# 详见 docs/cicd_requirements.md（P0-4 解释器钉死 / P0-5 退出码 / P0-6 逃生阀）

# ── 退出码约定（P0-5）─────────────────────────────────────────────────────────
EXIT_PASS=0      # 全部断言通过
EXIT_ASSERT=1    # 断言失败（真回归）
EXIT_PREREQ=2    # 前置缺失：解释器 / 产物 / 密钥，不代表代码有问题

# ── 路径解析 ──────────────────────────────────────────────────────────────────
# 约定：scripts/ci/check_*.sh 里 $0 就在 scripts/ci，可直接推出来；
# 钩子脚本在 scripts/git-hooks，必须先 export CI_HOME 再 source 本文件。
CI_DIR="${CI_HOME:-$(cd "$(dirname "$0")" && pwd)}"
REPO_ROOT="$(git rev-parse --show-toplevel 2>/dev/null || true)"
[ -n "$REPO_ROOT" ] || { echo "[CI][FATAL] 取不到仓库根目录（不在 git 工作区内？）" >&2; exit $EXIT_PREREQ; }
BASELINE="$CI_DIR/baseline.json"
RESULTS_DIR="$REPO_ROOT/evaluation/results"

# ── 输出helper：成功也要有摘要，不能让"没输出"等于"没跑"───────────────────────
ci_note() { printf '[CI] %s\n' "$*"; }
ci_ok()   { printf '[CI][PASS] %s\n' "$*"; }
ci_fail() { printf '[CI][FAIL] %s\n' "$*" >&2; }
ci_skip() { printf '[CI][SKIP] %s\n' "$*" >&2; }

die_prereq() { ci_fail "前置缺失: $*"; ci_fail "修复指引见 docs/cicd_requirements.md §2 P0-4/P1-3"; exit $EXIT_PREREQ; }

# ── P0-4：解释器钉死 ─────────────────────────────────────────────────────────
# 允许 CI_PYTHON 覆盖；默认硬编 ailearning 的 3.12。
# 绝不回落到裸 python —— 本机全局是 3.14.5，缺 16/19 个项目依赖，
# 回落会得到一堆 ModuleNotFoundError 假故障。
CI_DEFAULT_PY="D:/softwares/miniconda/envs/ailearning/python.exe"

ci_python() {
  [ -n "${CI_PY:-}" ] && return 0
  PY="${CI_PYTHON:-$CI_DEFAULT_PY}"
  [ -f "$PY" ] || die_prereq "找不到解释器: $PY
    期望是 conda env ailearning 的 Python 3.12。用 CI_PYTHON=<路径> 覆盖。"

  MINOR="$("$PY" -c 'import sys;print("%d.%d" % sys.version_info[:2])' 2>/dev/null)"
  [ -n "$MINOR" ] || die_prereq "解释器无响应: $PY"
  if [ "$MINOR" != "3.12" ]; then
    die_prereq "解释器版本不对: $PY 是 $MINOR，基线要求 3.12（conda env ailearning）
      用 CI_PYTHON=<3.12 解释器路径> 覆盖，或改 scripts/ci/baseline.json 的 interpreter.python_minor"
  fi
  CI_PY="$PY"
  return 0
}

# ── P1-3：.env 载入（自写解析，零依赖）───────────────────────────────────────
# 规则：KEY=VALUE，一行一对；# 开头为注释；值两侧的成对引号剥掉。
# 必须吃掉行尾 \r —— Windows 上记事本/PowerShell 写出的 .env 常是 CRLF，
# 不剥掉会让 key 变成 "sk-xxx\r"，然后在 API 侧报鉴权失败（极难查）。
load_dotenv() {
  [ -n "${LOAD_ENV_DONE:-}" ] && return 0
  LOAD_ENV_DONE=1
  ENVF="$REPO_ROOT/.env"
  [ -f "$ENVF" ] || return 0
  ci_note "载入 $ENVF"
  while IFS= read -r line || [ -n "$line" ]; do
    line="$(printf '%s' "$line" | tr -d '\r')"
    case "$line" in ''|'#'*) continue ;; esac
    case "$line" in *=*) ;; *) continue ;; esac
    k="${line%%=*}"
    v="${line#*=}"
    k="$(printf '%s' "$k" | tr -d '[:space:]')"
    # 去掉值两侧成对的单/双引号
    case "$v" in
      \"*\") v="${v#\"}"; v="${v%\"}" ;;
      \'*\') v="${v#\'}"; v="${v%\'}" ;;
    esac
    [ -n "$k" ] || continue
    # 已在环境里的不覆盖（命令行显式传入优先于 .env）
    eval "cur=\${$k:-}"
    [ -n "$cur" ] && continue
    export "$k=$v"
  done < "$ENVF"
}

# 判断一个 key 是否"真的填了"。
# 为什么需要：rag_pipeline.get_client() 只构造 OpenAI 客户端、不发网络请求，
# 所以塞一个 README 里的示例值（sk-xxxxxxxx）也能让 uvicorn 正常起来、
# /health 正常返回 200 —— 门禁会绿得毫无意义。占位符必须当"没填"处理。
ci_key_filled() {
  _v="${1:-}"
  [ -n "$_v" ] || return 1
  case "$_v" in
    # DashScope/OpenAI 风格示例值 + MiMo Token Plan 的 tp- 风格示例值
    sk-xxxxxxxx|sk-xxx|sk-\*\*\*|tp-xxxxxxxx|tp-xxx|tp-\*\*\*|\
    xxxxxxxx|CHANGE_ME*|TODO*|FIXME*|你的*) return 1 ;;
  esac
  return 0
}

# ragas 裁判用哪把 key —— 必须和 evaluate.py 的 JUDGE_PRESETS 保持一致。
# 默认 mimo；设 RAGAS_LLM_PROVIDER=deepseek 回到旧尺子。
judge_key_env() {
  case "${RAGAS_LLM_PROVIDER:-mimo}" in
    mimo)                echo "MIMO_API_KEY" ;;
    deepseek|DeepSeek)   echo "DEEPSEEK_API_KEY" ;;
    *)                   echo "" ;;
  esac
}

# ── P1-4：结果目录 ───────────────────────────────────────────────────────────
ensure_results_dir() { mkdir -p "$RESULTS_DIR" 2>/dev/null; }

# 当前 git sha（短），取不到就 unknown
ci_sha() { git -C "$REPO_ROOT" rev-parse --short HEAD 2>/dev/null || echo "unknown"; }

# P0-6 逃生阀：钩子层调用。返回 0 表示"这次放行，不跑检查"。
ci_skip_requested() { [ "${CI_SKIP:-0}" = "1" ]; }
