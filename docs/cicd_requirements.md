# 本地 CI/CD 需求文档（v0.1，待确认）

> 范围：**只在这台 Windows 机器上跑**，不接 GitHub Actions、不做云端并行、不做公网自动部署。
> 本文档是待确认稿。标注 `待实测` 的数字在 M1 实跑后回填；标注 `已实测` 的附证据位置。

---

## 1. 背景与决策前提（全部本机实测得出）

| 事实 | 实测值 | 对方案的影响 |
|---|---|---|
| 项目解释器 | conda env `ailearning` = Python 3.12.14（`.idea/misc.xml` 里 `sdkName = ailearning`） | 钩子必须用绝对路径调它，不能依赖 conda activate |
| 全局 Python | 3.14.5，缺 19 个依赖中的 16 个 | 绝不能用 `python`/`py` 裸调，会跑进错环境 |
| git | 2.54.0.windows.1，支持 `core.hooksPath` | 钩子可以入版本库 |
| 本仓库 `core.filemode` | **false** | 入库的钩子重新 clone 会丢 `+x` 被**静默跳过** → 必须 `git update-index --chmod=+x` |
| 现有钩子 | `.git/hooks` 内只有 `.sample`，零个在用 | 从零开始，无冲突 |
| 测试/检查工具 | `pytest / ruff / flake8 / mypy / black / coverage` **全部未装**；只有标准库 `unittest` | 决定 P0 用零依赖方案 |
| 大产物 | `data` 147 MB、`vectorstore` 56 MB、`models/bge-reranker-base` 1.1 GB，**全部 gitignore** | 云端拿不到，本地直接读现成 —— 这是选本地的核心理由 |
| torch | **`2.14.1+xpu`**（Intel XPU 构建，非 pip 默认索引） | 任何"另建干净 venv"都装不回等价环境，见 P2-1 |
| API 密钥 | `DASHSCOPE_API_KEY` / `DEEPSEEK_API_KEY` 进程 env 与 User 级注册表**均为空**；项目根无 `.env`；README 26–27 行教的是会话级 `$env:` | 要花钱的档必须先解决密钥落盘（D3） |
| 服务启动前提 | `RAGPipeline.__init__` 首行 `get_client()`（`src/rag_pipeline.py:309` → `:70-72` 抛 `EnvironmentError`） | `/health` 门**必须有 key 才能验**，不是免费的 |
| ragas 修复 | 已完成：`langchain-community 0.4.2 → 0.4.1`，`import ragas` 通过 | 依赖钉版是 CI 的前置条件，已满足一半 |

**本地方案换不来的东西**（说清楚再签）：全新环境可复现性的即时保证（用 P2-1 补偿，非每次）、PR 门禁与第三方证据、`--no-verify` 无法防御、机器挂了记录就没了。

---

## 2. 功能清单

### P0 — 没有这几条，这套东西不成立

**P0-1 提交前静态门（pre-commit，秒级，零密钥零网络）**
对 8 个 Python 文件做 `py_compile`；对 `src/*` `evaluation/*` 做 import 冒烟。已实测：`py_compile` 8 文件通过、8 个模块 import 全部 OK。

**P0-2 索引一致性门（pre-push，读本地现成产物）**
断言 `vectorstore/faiss_index.bin` 的 `ntotal` / `d` 与 `faiss_meta.json` 条数三者自洽，并与基线常量对齐；再跑一次真实 BM25 中文检索，断言命中数 > 0 且 Top1 带上 `bm25_score`。这一档**本地独有**（云端没有索引）。
**断言必须先校验字段存在性**：实测踩过 —— `faiss_meta.json` 里**没有** `company_name` / `block_type`（真实字段是 `chunk_id / content / stock_code / year / page_num / section / block_types / is_ocr / strategy / source_file / parent_content / parent_id / heading_level / table_json`），用 `.get()` 取值会得到 `None` 并**静默通过**。规约：每条字段断言前先把缺失字段列成断言失败项，禁止裸 `.get()`。

**P0-3 钩子版本化**
新增 `scripts/git-hooks/{pre-commit,pre-push}` + 一次性 `git config core.hooksPath scripts/git-hooks` + `git update-index --chmod=+x`。附一份 `scripts/git-hooks/README.md` 写明换机器要重跑的两条命令。

**P0-4 解释器钉死**
钩子内所有 Python 调用走 `CI_PYTHON` 变量，默认值硬编 `D:/softwares/miniconda/envs/ailearning/python.exe`；缺失则**明确报错退出**，绝不回落到 `python`（会掉进 3.14 空环境）。

**P0-5 可诊断的输出与退出码约定**
失败必须打印：档位、断言名、期望值 vs 实际值、修复命令。成功也要一行摘要（避免"没输出=不知道跑没跑"）。退出码：`0` 通过 / `1` 断言失败 / `2` 前置缺失（环境、产物、密钥）。

**P0-6 逃生阀**
`CI_SKIP=1 git commit ...` 可跳过（对齐"给了删除入口就要给恢复入口"的一致性要求：能拦也要能明确放行），跳过时钩子打印醒目警告，不静默。

### P1 — 明显增值，不阻塞 P0

**P1-1 纯函数单测集（目标 15 条）**
覆盖无需密钥、无需网络的确定性逻辑：`chunk_documents` 的三种分块策略边界、`serve.py` 的 `_build_source` / `_preview` / `_filter_meta`、`compare_strategies.compute_metrics` 的 hit_rate / mrr。每条**配一次反向验证**（故意改坏被测函数 → 断言必变红 → 还原 → 逐字节比对）。

**P1-2 服务启停门（CD 判据）**
拉起 `uvicorn serve:app` → 轮询 `GET /health` 直到 200 → 关停。启动失败要贴出原始 traceback（**已实测无 key 时就是 `OSError: 请设置环境变量 DASHSCOPE_API_KEY`，uvicorn 退出码 3**）。冷启动构成已实测：faiss 0.05 s + meta 0.10 s + BM25 建库 12.01 s ≈ 12.2 s，另加模块 import；**总时长 `待实测`**（被 key 卡住，见 §8 第 8 行）。轮询超时上限建议 90 s。

**P1-3 密钥落盘**
项目根 `.env`（`.gitignore` 第 12 行已覆盖 `.env`，无需改动）+ 一个 20 行的解析器供钩子读取。**密钥值由你提供，我不猜也不生成占位。**

**P1-4 结果产物**
每次 CI 写 `evaluation/results/ci-<UTC>.json`（`.gitignore:22` 已忽略 `evaluation/results/`）。字段见 §3.3。

### P2 — 收益/成本要你拍板，我不硬推

**P2-1 干净安装冒烟（每周或 `requirements.txt` 变更时）**
新建一次性 venv → `pip install -r requirements.txt` → import 冒烟 → 结果入回收站。价值：抓依赖漂移（今天重装会拿到 `ragas 0.4.3 / torch 2.15.0 / pandas 3.1.0`，与你实验环境两套）。
**成本要讲清**：装不出 `2.14.1+xpu`，且全量约 `待实测` 分钟、约 2 GB（D 盘空闲 398 GB 够）。它证明的是"公共库组合能不能装起来"，不证明 reranker 性能一致。

**P2-2 定时化**：Task Scheduler（已实测 `Get-ScheduledTask` 可用，机器上 189 个任务）或 QoderWork 定时任务。
**P2-3 本地发布脚本**：打 tag → 跑 P0+P1 → 重启服务 → 记一份发布快照。
**P2-4 临时公网演示**：cloudflared/frp 隧道，演示完关。本机未装，需要时再定。
**P2-5 `pre-commit` 框架**：不推荐。要额外装一个包，而收益不超过 P0-3 的裸 sh 脚本。

---

## 3. 数据模型

### 3.1 输入：环境变量

| 变量 | 必需性 | 当前状态 | 被谁消费 |
|---|---|---|---|
| `CI_PYTHON` | 有默认值 | 未设置 → 回落 ailearning 绝对路径 | 所有钩子 |
| `CI_SKIP` | 可选 | — | 逃生阀 |
| `DASHSCOPE_API_KEY` | P1-2 / 评估必需 | 已填（`.env`，去引号后长度 117，真实 embedding 调用 200） | `rag_pipeline.py:70`、`build_index.py:45` |
| `MIMO_API_KEY` | 默认 ragas 裁判（MiMo Token Plan） | **空，等你填**（平台「订阅管理 / API Keys」，`tp-` 前缀，只显示一次） | `evaluate.py` `JUDGE_PRESETS["mimo"]` |
| `DEEPSEEK_API_KEY` | 备用裁判（`RAGAS_LLM_PROVIDER=deepseek`） | 已填（长度 35）；旧结论档位，保留是为了能复现旧尺子 | `JUDGE_PRESETS["deepseek"]` |

### 3.2 输入：磁盘产物与基线常量

CI 的断言基线（"金标准"），来自 README 自述 + 本轮实测，落成 `scripts/ci/baseline.json` 一个文件：

```
chunks_total      7931         # faiss ntotal == meta 条数（2026-11-07 解析去重后；旧值 10352）
embedding_dim     1024         # faiss d
companies         5
years             2021,2022,2023
pdf_count         15           # data/manifest.json 实测为 15 条记录（已实测）
questions_count   20           # evaluation/questions.json（dict，含 version 键，已实测）
reranker_path     models/bge-reranker-base   # 1.1 GB，缺失时检索降级（README 已声明）
```

缺失产物时的行为定义（不许含糊）：`vectorstore` 不在 → P0-2 以退出码 `2` 跳过并提示重建命令，**不算失败也不算通过**；`models` 不在 → 只降级警告。

### 3.3 输出：CI 结果记录

`evaluation/results/ci-<UTC>.json`：

```
{ tier, started_at, duration_s, interpreter, git_sha,
  assertions: [ {name, expect, actual, pass} ],
  skipped_reason: null | "missing_vectorstore" | ... }
```

---

## 4. 接口

### 4.1 CI 阶段入口（新增，均为 sh，Git Bash 下可直跑）

| 脚本 | 入参 | 退出码 | 耗时 |
|---|---|---|---|
| `scripts/ci/check_static.sh` | 无 | 0/1/2 | **已实测 4 s**：`py_compile` 8 文件（0.25 s 级）+ 单进程 import 冒烟 **2.3 s**（分进程跑要 9.41 s，已合并） |
| `scripts/ci/check_index.sh` | `--allow-drift` / `--fast` | 0/1/2 | **已实测完整 11–14 s**（BM25 建库 9.5–12.0 s，波动来自 jieba 缓存冷热）；`--fast` **0.13 s** |
| `scripts/ci/check_tests.sh` | 无 | 0/1/2 | **已实测 4–5 s**（35 条用例本身 1.1–1.4 s，余为解释器启动） |
| `scripts/ci/check_service.sh` | `--port 8011` / `--timeout` | 0/1/2 | 无 key 时**已实测走 SKIP（退出码 2）**；冷启动总时长仍 `待实测`（需 key） |
| `scripts/ci/run_all.sh` | `--tier p0\|p0p1` `--fast-index` | 取最严重码 | **已实测 p0p1 = 20–24 s**（含服务门 SKIP） |

### 4.2 现有脚本被 CI 调用的方式（已实测其 CLI）

- `evaluation/evaluate.py --question-ids 1,2,3 [--skip-ragas]` → 冒烟用单题；`--skip-ragas` 只出题不打分的既有开关（`:246-247`）
- `evaluation/compare_strategies.py --strategies semantic --modes vector_only,hybrid --top-k 4`（`:230-234`）
- `src/serve.py`：`uvicorn serve:app --host 0.0.0.0 --port 8000`

### 4.3 服务端点作为部署门（`src/serve.py`，已实测端点存在）

| 端点 | CI 用途 | 是否花钱 |
|---|---|---|
| `GET /health` | P1-2 唯一判据：pipeline 就绪 | 否（但启动需 key 存在） |
| `POST /query` | P2-3 发布后人工抽测 | 是（embedding + LLM） |
| `POST /query/debug` | 检索链路定位（返回中间结果） | 部分是（向量召回需 embedding） |

---

## 5. 里程碑

**M1 · 骨架能拦错**（本文档确认后一次会话内完成）
P0-1～P0-6 + `baseline.json`。
验收：正常提交放行；故意造一个语法错必拦；`CI_SKIP=1` 能放行且有警告；反向验证（改坏 `check_index` 的基线常量 → 必红 → 还原逐字节一致）。

**M2 · 有断言对象 + 服务门**
P1-1（15 条 + 逐条反向验证）、P1-2、P1-3、P1-4。
验收：`run_all.sh --tier p0p1` 全绿并产出 JSON；拔掉 `DASHSCOPE_API_KEY` 时 `check_service` 给出退出码 2 和原始报错，而不是挂掉。

**M3 · 可复现兜底 + 本地发布**
P2-1、P2-2、P2-3。（P2-4/P2-5 默认不做。）
验收：一次真实"周日干净安装"跑完并留下 JSON 记录；`--allow-drift` 在漂移时仍给出可读差异报告。

---

## 6. 需要你拍板的决策点

**D1｜P0-1 是否引入 ruff？**
推荐 **不引入**。`py_compile` + import 冒烟已能拦住语法错与"组件没 import 就静默消失"这类问题；ruff 能多抓的是风格与未用变量，对这个 2277 行的单人项目，收益不抵"又要装一个包"。真要装，M2 一起装。

**D2｜测试框架用标准库 `unittest` 还是装 `pytest`？**
推荐 **P0 阶段用 `unittest`（零安装），P1-1 落地时若样板太重再换 `pytest`**。代价说清：`unittest` 写参数化用例和 fixture 会啰嗦不少。

**D3｜`.env` 怎么读？**
推荐 **自写 20 行解析器**，不装 `python-dotenv`。格式固定 `KEY=VALUE`、不支持和引号转义以外的花活。另外两条路：装 python-dotenv（生态标准但多一个依赖）；或在 PowerShell `$PROFILE` 里设会话变量（最省事，但钩子从 Git Bash 起的看不到，等于 P1-2 白做）。
**无论哪种，密钥值得你给。**

**D4｜索引基线 10352 / 1024 写死还是动态推？**
推荐 **写死进 `baseline.json` + `--allow-drift` 覆盖**。理由：写死才拦得住"重建索引时静默少了一半 chunk"这种真实事故；动态推等于自己验自己。

**D5｜BM25 建库（10352 条 jieba 分词）能不能每次 pre-push 都跑？**
**已实测：12.01 s**（其中 jieba 词典加载 0.43 s，其余为分词 + BM25 建模；检索本身 0.03 s）。
推荐 **保留在 pre-push**：12 s 对推送前等待可接受，且这一档拦的是索引/检索链路回归，正是你最在意的"可验证"部分。若之后嫌慢，再加 `--fast` 只做 faiss 元数据比对（实测亚秒级：加载 0.05 s + meta 0.10 s）。

**D6｜钩子拦不住怎么办 —— 拦截级别？**
推荐 **pre-commit 阻断 + 明确逃生阀**（P0-6）。不放逃生阀的话，你半夜赶工时会直接 `--no-verify`，那样连日志都没了。

**D7｜CI 结果 JSON 要不要提交入库当趋势历史？**
推荐 **提交一份精简版**（KB 级，只留断言名与 pass/fail + 指标均值），完整报告留本地。价值：评估指标随改动的漂移变成可 diff 的历史，这对你这个带消融实验的项目是真收益。

**D8｜P2-5 `pre-commit` 框架要不要上？**
推荐 **不上**，理由同 D1。

**D9｜服务常驻方式？**
推荐 **先手动起**。真需要长期开再选 Task Scheduler 登录自启（已实测可用）或 WSL systemd（Ubuntu-24.04 已实测在）。不推荐为了这个去装 NSSM/WinSW。

---

## 7. 本文明确不做的事

不接 GitHub Actions；不做 PR 门禁；不做自动公网部署；不改 `requirements.txt` 里除已钉两行之外的版本（D 项决策后再动）；不生成任何占位资源；不在没密钥的情况下假装跑通了端到端。

---

## 8. 本会话实测记录（命令与数字可复跑）

调用解释器一律 `D:/softwares/miniconda/envs/ailearning/python.exe`，工作目录 `D:\projects\rag_annual_report`。

| # | 测什么 | 结果 |
|---|---|---|
| 1 | `py_compile src/*.py evaluation/*.py` | 8 文件全过，**0.25 s** |
| 2 | `src` 6 模块 + `evaluation` 2 模块逐个 import | 全 OK，分进程共 **9.41 s** |
| 3 | `faiss.read_index` → `ntotal` / `d` | **10352 / 1024**，与 `faiss_meta.json` 条数 **10352** 三者自洽 → 基线断言成立 |
| 4 | `rag_pipeline.BM25Store()` 建库 | **12.01 s**（jieba 词典 0.43 s）；日志 `BM25 索引完成` |
| 5 | 真实检索 `"贵州茅台 2021 年营业收入"` top-10 | 命中 **10** 条，耗时 **0.03 s**，Top1 `bm25_score=31.77` |
| 6 | **反向验证**：把期望 ntotal 从 10352 改成 10351 | 断言**如期变红**（`基线不符: ntotal=10352(期望10351)`），证明这条守卫不是空转 |
| 7 | **字段名静默通过演示** | 用 `company_name` / `block_type` 取值 → **None 且不报错**。真实字段为 `chunk_id / content / stock_code / year / page_num / section / block_types / is_ocr / strategy / source_file / parent_content / parent_id / heading_level / table_json`。已据此给 P0-2 加规约 |
| 8 | `uvicorn serve:app --port 8011`（无 key） | **退出码 3**，`OSError: 请设置环境变量 DASHSCOPE_API_KEY`（`rag_pipeline.py:309 → :72`）；`GET /health` → `http=000`（连接被拒，服务没起来） |
| 9 | `evaluate.py --question-ids 1 --skip-ragas`（无 key） | 走到 `evaluate.py:68` 构造 `RAGPipeline` 才因同一个 key 报错，退出码 1。**说明 ragas 的 import 修复没有把脚本堵在前面** |

**这些数字证明到的 / 没证明到的**（不含糊）：
证明到 —— 静态门与索引/检索门的断言逻辑真实有效（含反向验证）；`ragas==0.3.1 + langchain-community==0.4.1` 组合下 `import ragas` 及 `evaluate.py` 需要的 7 个符号、`metric.llm / metric.embeddings` 赋值姿势全部可用。
**没证明到 —— ragas 打分从未真实跑过一次。** 它下游要先出答案（要 embedding + LLM、要花钱），而两个 key 现在都不在机器上。要补这一刀，只差你给一份 `.env`。
**（此段已于 §10 作废：key 已填入 `.env`，端到端与 ragas 打分均已真实跑过。）**

---

## 10. key 填入后的真跑记录（2026-11-05）

`.env` 填了真 key（DashScope 长度 117 / DeepSeek 长度 35，只测长度不打印值）。`git check-ignore -v .env` → `.gitignore:13`，`git ls-files .env` 为空。

| # | 测什么 | 实测 |
|---|---|---|
| 10 | `run_all.sh --tier p0p1` 全档 | **退出码 0，全绿，总耗时 34 s**（静态 4 s / 索引 13 s / 单测 3 s / 服务 14 s）。§8 那句"结论不完整"到此作废 |
| 11 | 服务门正向（真 key 首次） | `/health` → `{"status":"ok","pipeline_ready":true}`，冷启动 **14 s**；端口 8011 跑完无残留监听 |
| 12 | **真实 `POST /query`**（走 embedding→检索→RRF→LLM 生成） | HTTP 200，**往返 2 s**。答案正确：茅台 2023 营收 **14,769,360.50 万元（≈1476.94 亿）**，4 条引用带 `chunk_id` 与页码 |
| 13 | **真实 ragas 打分**（2 题） | **Faithfulness 1.0000 / Answer Relevancy 0.9997 / Context Precision 0.5000 / Context Recall 0.5000**，耗时 **38 s**，产物 `evaluation/results/native_20261105_132851.json` + `.csv` |
| 14 | `run_eval.sh` 守卫 | 塞占位 `sk-xxxxxxxx` → 拒绝，退出码 2（不花钱）；真跑 1 题 `--skip-ragas` → 13 s 通过 |

### 真跑暴露的两个问题（只报，未改）

1. **元数据过滤只作用于一半候选 —— 疑似真 bug。**
   `src/rag_pipeline.py:327` 把 `filter_meta` 传给向量检索，但 `:333` 的
   `BM25Store.search(retrieval_query, TOP_K_RETRIEVE)` **根本没有 filter 形参**（定义在 `:147`）。
   `use_bm25=True` 是默认路径（`serve.py:53` 就是这么构造的），于是 RRF 会把别的年份/公司的块混进来。
   实测证据：请求里写了 `year="2023"`，返回的第 4 条引用是 **`600519_2022_00205`（2022年报 第52页）**。
   这大概率也是 Context Precision / Recall 只有 0.5 的直接原因。
   **改它等于改检索语义**，会让已有实验结论变化，所以没动，等你决策。
2. **`evaluate.py` 等脚本不读 `.env`**，只认环境变量。直接 `python evaluation/evaluate.py` 会撞
   `OSError: 请设置环境变量 DASHSCOPE_API_KEY`（本会话真撞到过一次，退出码 1、2 秒就挂）。
   已补 `scripts/ci/run_eval.sh` 作为花钱档的正式入口（载 `.env` + 校验两个 key + 透传参数），
   **不动生产代码**。想彻底解决就得给 `rag_pipeline.py` 加 Python 侧 dotenv —— 那属生产路径，等你点头。

### 顺带记一笔（非缺陷）

引用里出现的章节串是 `1 / 143 > 第十节 > 二、 财务报表` 这种形状 —— PDF 页眉页码混进了 `section_path`，
还有 `二、非持续的公允` 这种被截断的标题。属解析层噪声，与 README 已声明的"表格块重复"是同一类已知质量问题。

---

## 9. M1 / M2 落地状态（本次会话）

### 已交付并实跑通过

| 项 | 落地物 | 状态 |
|---|---|---|
| P0-1 静态门 | `scripts/ci/check_static.sh` | 实跑 4 s 全绿 |
| P0-2 索引门 | `scripts/ci/check_index.py` + `.sh` | 实跑 11–14 s，**11 条断言全绿**；`--fast` 0.13 s |
| P0-3 钩子版本化 | `scripts/git-hooks/{pre-commit,pre-push}` + `core.hooksPath` + `update-index --chmod=+x`（`git ls-files -s` 实测 `100755`） | `git hook run pre-commit` 实测被调用 |
| P0-4 解释器钉死 | `ci_lib.sh` 的 `ci_python()`，非 3.12 直接退出码 2 | 生效 |
| P0-5 输出与退出码 | `[CI][PASS/FAIL/SKIP]` + 0/1/2 | 生效 |
| P0-6 逃生阀 | `CI_SKIP=1` 实测放行并打印警告 | 生效 |
| P1-1 单测 | `tests/` 4 个文件 **35 条**（原计划 15 条） | 1.1–1.4 s 全绿 |
| P1-3 `.env` 载入 | `ci_lib.sh` 的 `load_dotenv()`（POSIX sh，含 CRLF 剥离） | 已写；**因无 key，正向路径未跑过** |
| P1-4 结果产物 | `scripts/ci/record.py` → `evaluation/results/ci-latest.json` | 实跑生成 |
| 服务门负面路径 | 无 key → SKIP 退出码 2 + 指路 | 实跑复现 |

### 反向验证（6/6 通过，逐条改坏→必红→还原→sha256 一致）

| 被改坏的东西 | 该拦住它的守卫 | 结果 |
|---|---|---|
| `serve.py` 页码哨兵 `if page and page != -1` → `if page` | 单测门 | 红 ✓ |
| `build_chunk_id` 补零 `%05d` → `%04d` | 单测门 | 红 ✓ |
| RRF 默认 `k=60` → `k=40` | 单测门 | 红 ✓ |
| `compute_metrics` 的 `[:top_k]` → `[:top_k*2]` | 单测门 | 红 ✓ |
| `baseline.json` 条数 10352 → 10000 | 索引门 | 红 ✓ |
| 往 `chunk_documents.py` 注入语法错 | 静态门（经 `git hook run pre-commit`） | 红 ✓ |

六个文件还原后 sha256 与改前**逐一一致**；`git status` 复核 `src/` 与 `evaluation/` 零改动。

### 新暴露的两个既有问题 —— 已按你的指示改掉并验证

1. **`serve.py:83-85` 的 `Field(example=...)`（Pydantic V2 废弃）→ 已改 `json_schema_extra={"example": ...}`**。
   验证：`pytest` 摘要从 `35 passed, 3 warnings` 变成 **`35 passed`**；`app.openapi()` 里三个 example 全在
   （`question='贵州茅台2023年营业收入是多少'` / `stock_code='600519'` / `year='2023'`），`required: ['question']` 未变 —— Swagger UI 的示例值承诺没被改坏。
2. **`src/parse_pdf.py` 的 `import fitz` → `import pymupdf`**（含 5 处属性引用；局部变量名 `fitz_doc`/`fitz_page` 刻意没动，免得多出无意义 diff）。
   验证不止"能 import"：拿最小的那份真实 PDF（五粮液 2022，2.1 MB）现场跑 `AnnualReportParser(...).parse()`，
   产出 **803 blocks**，与磁盘上用旧 `fitz` 解析出的结果**块数一致** → 改名行为等价。
   那份 JSON 的 mtime 仍是 2026-11-02 18:55，确认 `parse()` 只返回不落盘、没碰你的数据。
   顺带把 `ARCHITECTURE.md:59,64` 与 `requirements.txt:3` 里 "PyMuPDF/fitz" 的表述同步成现状。

### P1-2 服务门：正向路径已实测

`.env` 已生成（空值占位；`.gitignore:12` 覆盖，`git ls-files .env` 为空 → 不会被提交）。填 key 后即可真跑。

用一次性假 key 跑通**装配路径**，并测出最后那个 `待实测`：

- **冷启动 13 s**，`GET /health` 返回 `{"status":"ok","pipeline_ready":true}`。
- **这条只证明"索引加载 + BM25 建库 + FastAPI 装配"没问题，不证明 key 有效** ——
  `get_client()` 只构造客户端、不发请求。据此加了 `ci_key_filled()`：空值与 `sk-xxxxxxxx` 这类占位一律按"没填"处理
  （退出码 2，并把两种失败模式分别讲清），堵住假绿。
- **抓到并修掉一个我自己引入的真 bug**：原先 `( cd src && uvicorn ) &` 里 `$!` 是子 shell 的 pid，
  `kill` 掉 shell 后 python.exe 仍占着 8011（实测泄漏过一次 PID 51388）。改成 `PYTHONPATH` 注入 +
  `stop_server()` 轮询端口确认释放，超时再用 PowerShell `Stop-Process` 兜底。复跑确认不再泄漏。
  （`serve.py:31-32` 全用 `Path(__file__)` 定位资源，所以换 cwd 不影响静态页与索引。）

### 超出批准范围的改动（可单独回滚）

- **新增 `.gitattributes`**：钉 `*.sh` 与 `scripts/git-hooks/*` 为 `eol=lf`。
  原因：本机 `core.autocrlf=true` 且原先没有 `.gitattributes`，重新 clone 会把钩子脚本变成 CRLF，`sh.exe` 读到 `#!/usr/bin/env sh\r` 直接 `bad interpreter` —— 钩子静默失效（与 apk1handler 的 `gradlew` 事故同源）。
  刻意**没有**给 `*.py` 定规则，避免整仓换行 reflow 污染 diff。不要就 `git rm .gitattributes`。
- **`ci_key_filled()` 占位守卫** 与 **`stop_server()` 端口兜底**：起因是你让我建 `.env` 占位（会引入假绿）和实测到的端口泄漏。不需要可撤。
- **`ARCHITECTURE.md` / `requirements.txt` 里 fitz 字样的文档同步**。

### 仍然欠着的（截至 §10 真跑之后）

- ~~ragas 端到端真实打分~~ → 已跑（2 题，见 §10 第 13 行）。**全量 20 题仍未跑**（成本与耗时明显更高，要不要跑等你说）。
- P2-1 干净安装冒烟：M3。装不出 `torch 2.14.1+xpu` 这件事要在做之前先定口径。
- §10 列的两个真跑发现（BM25 不过滤 / 脚本不读 `.env` 的根治法）：都涉及生产代码，等你决策。
- 提交：`src/`、`evaluation/` 的改动与全部新文件至今仍只是 **staged，未 commit**。

---

## 11. BM25 过滤漏洞的修复（你拍板后执行）

### 改了什么

| 位置 | 改动 |
|---|---|
| `src/rag_pipeline.py`（VectorStore 之前） | 新增共享谓词 `match_filter(item, filter_meta)`，语义与原向量侧**逐字一致**（缺键当 `""`、两侧 `str()` 后比、多键 AND） |
| `src/rag_pipeline.py` `VectorStore.search` | 内联的过滤表达式改为调 `match_filter`（行为不变，只为让两条链路同源） |
| `src/rag_pipeline.py` `BM25Store.search` | 新增 `filter_meta` 形参。做法是把不合规块的分数 `np.where(keep, scores, 0.0)` 置零，**复用下面既有的 `scores[idx] < 1e-9` 判据**自动跳过，不引入新的截断/回填行为 |
| `src/rag_pipeline.py` `RAGPipeline.query` | BM25 调用补上 `filter_meta`（原缺陷现场） |
| `src/serve.py` `/query/debug` | 同上。**注意**：原注释写的是「BM25 检索（不受 filter_meta 限制，搜全库后融合）」，即当时是刻意的；但主链路修好后，调试接口若口径不同就失去"逐步复现真实链路"的意义，所以一并对齐。想保留旧语义就说一声，我回滚这一处 |

**没动**：`evaluation/compare_strategies.py` 的 `retrieve_bm25`/`retrieve_vector` —— 消融实验本来就在测"裸检索"的 hit_rate/mrr，加过滤会把指标人为抬高。这属测量口径，改它等于改已发表结论。

### 守卫与证据

- 新增 `tests/test_filter.py` **13 条**（谓词语义 5 条 + 合成语料上的 BM25 过滤 6 条 + AST 接线守卫 2 条）。
  其中**接线守卫是必要的**：只测 `BM25Store.search` 的能力，删掉调用点的实参照样全绿 ——
  而那正是缺陷原样（能力存在、没接上）。两条 AST 守卫分别钉住 `RAGPipeline.query` 与 `serve.query_debug` 的调用实参个数。
- **反向验证 5/5 通过**（逐条改坏 → 必红 → 还原 → sha256 与改前一致）：
  摘掉分数掩码 / 谓词 `all→any` / 空值分支反转 / 主链路丢实参 / debug 端点丢实参。
  判红一律用**退出码**，不靠匹配 "1 failed" 字符串。
  （第一次跑这个 harness 挂了：现有源文件工作副本是 **CRLF**，我的多行匹配串用的是 `\n` → 匹配不到。
  已改成单行匹配。顺带确认 `HEAD` 里存的是 LF，git 因 `core.autocrlf=true` 已归一化，不是问题。）
- 全套 `pytest`：**48 passed**（35 → 48）。
- **真实语料证据**（10352 块，探针 `"贵州茅台 营业收入 亿元"`）：
  无过滤 top10 年份分布 `{2021, 2022, 2023}`，Top2 就是 `600519_2022_00205`；
  加 `{year: 2021}` 后只剩 `{2021}`，那块 2022 消失。
- **端到端证据**（真 HTTP `POST /query`，`year="2023"`）：修复前引用里混着 `600519_2022_00205`；
  修复后 4 条引用全是 `_2023_`，答案仍是正确的 14,769,360.50 万元，往返 2.55 s。
- 索引门新增 2 条真语料断言（过滤后只剩指定年份 + 对照组确实跨年，防"空断言"）。

### 全量 20 题评估

见 §12（跑完回填）。

---

## 12. 全量 20 题真跑结果（2026-11-05 14:53）

`sh scripts/ci/run_eval.sh`（无参数＝全部 20 题）→ **退出码 0，292 s（约 4.9 分钟）**。
产物 `evaluation/results/native_20261105_145339.json` + `.csv`。

| 指标 | 全量 17 题 | 对比：之前 2 题小样本 |
|---|---|---|
| Faithfulness | **0.7750** | 1.0000 |
| Answer Relevancy | **0.6332** | 0.9997 |
| Context Precision | **0.2729** | 0.5000 |
| Context Recall | **0.3824** | 0.5000 |

20 题里 3 题是 `should_refuse`（`evaluate.py:143` 按注释排除），**实际进 ragas 的是 17 题**。
之前那 2 题是严重偏样本，四个指标全被高估 —— 这也说明"跑一题看看"绝不能当成结论。

### 必须先讲清的一件事：这次修复对上面这组数字**没有影响**

`evaluate.py:74` 调的是 `pipeline.query(q["question"], verbose=False)` —— **不传 filter_meta**。
所以评估走的一直是"不筛选"的检索路径，BM25 过滤漏洞修没修，指标都一样。

- 修复真正生效的地方是 **HTTP 接口带筛选的请求**（前端 UI 选公司/年份 → `stock_code`/`year`），
  已用真实 `POST /query` 端到端验证：同一条 `year=2023` 请求，引用从混入 `600519_2022_00205` 变成 4 条全 `2023`。
- 因此这组 20 题数字与 README/ARCHITECTURE 的口径**可比**（同一条不筛选路径）。

### Context Precision 只有 0.273 —— 但现阶段无法定位是哪几题

`raw_results` 每条只存 `question / answer / contexts / ground_truth / question_id / question_type`，
**逐题 ragas 分数没有落盘**（顶层 `scores` 只有四个均值，CSV 也只有 `answer_len`/`context_count`）。
所以现在只能看均值，不能回答"哪类题拖低的"。这是个真实的可诊断性缺口，
补法很小：在 `evaluate.py` 里把每题的 `score row` 一并写进 JSON。属新改动，等你点头再做。

### 收尾状态

`run_all.sh --tier p0p1` 修完后复跑：**退出码 0 全绿，37 s** —— 静态 3 s / 索引 **13 条断言** 13 s / 单测 **48 条** 4 s / 服务 14 s。
索引门新增的两条真语料断言（过滤后只剩指定年份 + 对照组确实跨年）实测通过：无过滤召回 3 个年份，加 `{year:2021}` 只剩 `{2021}`。

---

## 13. 逐题 ragas 分数落盘

改动 9 行（`evaluation/evaluate.py`）：`run_ragas_eval` 里把四个指标按 dataset 行序 zip 回原 dict，
写进 `r["ragas"]`；`save_results` 的 CSV 增加四列。`None`/`NaN` 原样存 `null`，不塞 0（塞 0 会污染均值口径）。

关键点：`eval_results` 是 `results` 的过滤视图、**共享同一批对象**，所以回填它等于回填落盘的那份，
不需要额外传参。`should_refuse` 题保留空的 `ragas`，下游能区分"没打分"与"打了 0 分"。

### 证据

- **真跑 3 题（32 s）逐题分数确实落下来**，且能反推校验 zip 没有错位：
  precision `(0.5+0.5+0.25)/3 = 0.4167` == 报告的均值 `0.4167`；recall `(0+1+1)/3 = 0.6667` == `0.6667`。
  产物 `evaluation/results/native_20261105_202645.json` / `.csv`（CSV 表头已含四个指标列）。
- 新增 `tests/test_persistence.py` **3 条**（离线、monkeypatch `RESULT_DIR` 到 tmp_path，不污染真实产物目录）。全套 **51 passed**。
- **反向验证 3/3**：摘掉 CSV 指标列 / 把 `r.get("ragas",{})` 退回硬取键（should_refuse 会 KeyError）/ 清空顶层均值。
  其中第一条**第一次做错了**：我替换的是 `context_count` 那行，而真正产生列的是后面的 `**{m: ...}`，
  等于什么都没改，测试正确地保持绿色 —— 是变异无效，不是守卫无效。换成把指标名元组改成 `()` 后如期变红，
  点名失败用例 `test_csv_has_one_column_per_metric`。

### 已经能看到的信号

3 道 `simple_fact`（最容易的一类）的 context_precision 是 **0.5 / 0.5 / 0.25** ——
说明 0.273 那个均值不是被难题拖的，而是**上下文里每 4 块只有约 1 块被裁判认为有用**，是检索精度本身的问题。
要按题型定量归因还得重跑一次全量（旧那份 14:53 的产物里没有逐题分数）。

**注意 §12 的那组 20 题数字仍是"无逐题分数"版本**，本次改动不会追溯生效。

---

## 14. ragas 裁判从 DeepSeek 换成小米 MiMo Token Plan

代码只有 4 处引用旧供应商（`evaluate.py` 的 URL / 模型名 / key / base_url），已重构成可切换预设：

```
JUDGE_PRESETS = { mimo: {token-plan-cn 端点, mimo-v2.6-flash, MIMO_API_KEY},
                  deepseek: {api.deepseek.com, deepseek-v4-flash, DEEPSEEK_API_KEY} }
resolve_judge()   # 默认 mimo；RAGAS_LLM_PROVIDER 切档；MIMO_BASE_URL / MIMO_MODEL 可覆盖
```

三件设计决定：

1. **Embedding 不换**。MiMo Token Plan 的公开资料只有 chat/工具调用/多模态/TTS/ASR，**没有 embeddings 端点**；
   而 `answer_relevancy` 等指标要算向量相似度。已用一条守卫钉死（`test_embeddings_stay_on_dashscope`）。
2. **保留 DeepSeek 作为可切档**，因为 ARCHITECTURE.md 那批结论是在 DeepSeek 裁判上跑的 —— 换裁判等于换尺子，
   得留一条回到旧口径做归因的路。
3. **`main()` 加了裁判预检**，在生成答案之前就校验 key/档位并 `exit 2`。理由很实际：全量 20 题实测 292 s 且要花钱，
   不该在答案全跑完之后才报"缺 key"。

### 验证

- 新增 `tests/test_judge_config.py` **9 条**（离线）；全套 **60 passed**（51 → 60）；`run_all` 全绿 29 s。
- **反向验证 5/5**，每条都点名抓到它的用例：旧 base_url 被顺手"修正" / 默认档被改 / `tp-` 占位符没认出 /
  预检失败却不拦 / Embedding 被顺手切走。
- **DeepSeek 档实跑 1 题（29 s，退出码 0）**：日志显示裁判为 `deepseek / deepseek-v4-flash`，
  分数 F=1.0 / AR=1.0 / CP=0.5 / CR=0.0 —— 与 §13 那道题一字不差，说明重构没动判定逻辑。
- `run_eval.sh` 改为按档位查对应 key：缺 `MIMO_API_KEY` → 退出码 **2**（实测）；`--skip-ragas` 时不再强要裁判 key。

### 还没做 / 做不到

- **`mimo-v2.6-flash` 这个模型 ID 没被真实验证过**：`.env` 里 `MIMO_API_KEY` 还空着，没法发真实请求。
  我查到的资料里出现的是 `mimo-v2.6-flash-rl` 和 `mimo-v2.5`。你填好 key 后跑一题冒烟即可确认；
  若返回体解析失败（`reading choices` 那类），基本就是 ID 要带 `-rl`，用 `MIMO_MODEL` 覆盖试即可，不用改源码。
- **`temperature=0` 对 MiMo 是否可用没验证**：社区资料提到 V2.6 RL 系列推荐 `temperature=1.0 / top_p=0.95`，
  个别服务会直接拒绝 `temperature=0`。现在沿用了原裁判的 `temperature=0`（判定可复现性更重要），冒烟时一起看。
- 文档同步了 README/USAGE_GUIDE/ARCHITECTURE 共 5 处；`USAGE_GUIDE.md:548-551` 那段"生成侧也可换 DeepSeek"的提示
  **没动** —— 它讲的是 `rag_pipeline.py` 的生成模型（现在仍走 DashScope `qwen-plus`），不在本次范围内。

### 一次操作失误的如实记录

给 `.env` 加 MiMo 那行时，我用 `DASHSCOPE_API_KEY=`（原行的**前缀子串**）做锚点，并在新内容里写了一段**我自己虚构的 key 文本**，
结果把真值挤到了假串后面，等于破坏了正在使用的 DashScope key。发现后立即用精确切除还原，
并做了三重确认：去引号后长度回到实测过的 117、`DEEPSEEK` 那行未受影响（35+2 引号=37）、
**真实 embedding 调用返回 HTTP 200 / 1024 维**。随后改用只匹配注释行的锚点，不再触碰任何 key 行。
教训：改含密钥的文件时，锚点必须避开被改动行的前缀，且绝不往文件里写任何臆造的值。

---

## 15. MiMo 裁判冒烟验证（2026-11-06，key 填好后）

`MIMO_API_KEY` 已填（长度 51，`tp-` 前缀，非占位）。四项结论，都有实测支撑：

1. **模型 ID 可用，且我之前资料里的 `-rl` 担心是多余的。** `GET /v1/models` 返回权威列表：
   `mimo-v2.5-asr`、`mimo-v2.5-tts`、`mimo-v2.5-tts-voiceclone`、`mimo-v2.5-tts-voicedesign`、
   **`mimo-v2.6-flash`**、`mimo-v2.6-pro`。代码里写的 `mimo-v2.6-flash` 正在列表上；pro 档叫
   `mimo-v2.6-pro`（不是社区文章里的 `mimo-v2.5-pro`）。
2. **`temperature=0` 服务端接受**（HTTP 200），不需要改采样参数。
3. **一度出现 `content` 空、回复全在 `reasoning_content` —— 查明是小 `max_tokens` 的产物，不是温度问题。**
   `max_tokens=16/256` 时"1+1"这类短问题返回空 content；换成接近裁判真实问法 + `max_tokens=512`，
   `temp=0`／`temp=1.0`／`enable_thinking=False` 三种配置各跑 3 次，**0/3 空回复**。
   残留风险：思考链吃满预算时仍可能返回空 content，ragas 会拿不到判定。
4. **真实 1 题冒烟通过**：`sh scripts/ci/run_eval.sh --question-ids 1` → 退出码 **0**、0 条 Traceback，
   四个指标全部产出，逐题分数也正常落盘。

### 换裁判确实换尺子（同一道题，两个裁判）

| 裁判 | faithfulness | answer_relevancy | context_precision | **context_recall** |
|---|---|---|---|---|
| DeepSeek | 1.0 | 1.0 | 0.5 | **0.0** |
| MiMo | 1.0 | 1.0 | 0.5 | **0.5** |

只有 context_recall 不一致。**n=1 不能下结论**，但它直接证明了 §14 里那句"换裁判等于换尺子"不是理论担忧 ——
和 ARCHITECTURE.md 旧数字对比时，必须先固定裁判档位。

### 耗时（决定全量成本）

1 题全程 **105 s**（vs DeepSeek 同题 29 s）。拆出来：建索引+生成答案 55 s（走 DashScope，与裁判无关），
MiMo 打 4 个指标 **49 s**。按此粗估全量 17 题 ≈ **14–18 分钟**（ragas 会并发部分指标，实际可能更短），
比 DeepSeek 那次的 292 s 明显贵。要不要跑全量，先看这个数。

### 顺带修的一处不实文案

`run_eval.sh` 原本固定打印"会产生 DashScope + DeepSeek 调用"，默认档已是 MiMo —— 改成按实际档位播报
（`DashScope 调用；裁判走 $PROVIDER`）。

### 建议但没做（等你点头）

加一个**裁判连通性探针**进 `run_eval.sh` 预检：发一次几 token 的真实请求，断言 `content` 非空。
现在的预检只验 key 存在与档位合法（离线），验不到"裁判真能返回可解析内容" —— 而上面第 3 条说明这个失败模式真实存在，
14 分钟的全量跑之后才发现就太晚了。成本约 25 token。**（已实现，见 §16。）**

---

## 16. 探针落地 + MiMo 全量评估与归因（2026-11-06）

### 探针已实现（就是上面"建议但没做"那条）

`scripts/ci/check_judge.py`，已接进 `run_eval.sh` 的预检之后、真正评估之前。
**配置一律来自 `evaluate.resolve_judge()`** —— 探针不许自己拼一份，否则改了预设而探针还在测老地址，探针就变成假绿灯（有守卫钉这点）。

实测一次：HTTP 200，往返 4.5 s，content=`通了`，45 token，退出码 0。

- 退出码分档：鉴权/网络/模型 ID 问题 → **2**（前置缺失）；连通且鉴权过但 **content 为空** → **1**（这才是 ragas 不可用）。
  后者单列一档，因为它是 §15 实测到的真实失败模式，且性质不同 —— 环境没问题，是模型输出形态不能用。
- `PROBE_MAX_TOKENS = 512` 被测试钉住：用小预算探出来的"空 content"是参数的锅，会把好配置误判成坏。
- `main()` 才载 `.env`，`run()` 保持纯逻辑 —— 否则测试会被开发者本机 key 悄悄救活，结果随机器状态漂移。

**离线守卫 11 条**（`tests/test_judge_probe.py`）：空 content / 401 / 404 / 网络错 / 配置同源 / 预算下限 / 缺 key 不发请求 / `.env` 解析三规则。反向验证 **4/4**。

### 全量 20 题（MiMo 裁判）实测

**393 秒（6.5 分钟）**，不是我上一轮估的 14–18 分钟 —— 粗估错了：ragas 把 17 题 × 4 指标并成 68 个任务并发跑（实测 1.1–4.8 s/步）。

| 指标 | DeepSeek 裁判 | MiMo 裁判 | 差 |
|---|---|---|---|
| Faithfulness | 0.7750 | **0.8676** | +0.093 |
| Answer Relevancy | 0.6332 | 0.6251 | −0.008 |
| Context Precision | 0.2729 | **0.1993** | −0.074 |
| Context Recall | 0.3824 | **0.4454** | +0.063 |

**尺子确实换了**：MiMo 对忠实度更宽松、对上下文精度更苛刻。跟 ARCHITECTURE.md 旧数字对比时必须标明档位。

### 归因（`scripts/ci/analyze_scores.py`，17 题有分数 / 3 题 should_refuse 无分数）

| 题型 | n | faithfulness | answer_relevancy | **context_precision** | context_recall |
|---|---|---|---|---|---|
| simple_fact | 5 | 1.000 | 0.977 | 0.478 | 0.900 |
| precise_number | 5 | 0.758 | 0.594 | **0.200** | 0.400 |
| cross_doc_compare | 4 | 0.802 | **0.205** | **0.000** | **0.000** |
| time_trend | 3 | 0.917 | 0.651 | **0.000** | 0.357 |

0.199 这个均值的来源已经很明确：

1. **`cross_doc_compare` 四题精度与召回全 0** —— 对比题需要同时命中两家公司的块，`ctx=4` 的上下文里显然只装得下一家。
   这是结构性缺陷，不是调参能解决的：要么对比题走"每家各取 top-k 再合并"，要么给这类问题单独放大上下文预算。
2. **`time_trend` 精度也为 0** —— 趋势题要跨 2021/2022/2023 三年，同一个根因（一次检索装不下多年）。
3. **`precise_number` 精度 0.2** —— 具体金额在表格里；而 README 已声明表格内容在 `table`/`text` 块里重复约 4,491 次，
   重复块会挤占 top-k 里的有效位置。
4. `simple_fact` 是唯一健康的档位（0.478 / 0.900），说明基础检索没问题，问题集中在**需要多块组合的题型**。

### 又一次"测试没覆盖到的分支"教训

`analyze_scores.py` 第一版带一个 `NameError`（嵌套推导式里引用了不存在的 `k`），
**6 条测试全绿却当场崩在真实数据上** —— 因为所有测试都显式传 `--file`，自动发现那条分支一次没跑过。
已重写该分支并补 4 条测试覆盖：选中带分数的文件、忽略 §13 之前的无分数旧产物、只有旧产物时判 SKIP(2)、
半截 JSON 不整体崩掉。

顺带把 `analyze_scores.py` 做成自带自检：用逐题分数重算均值，与顶层 `scores` 比对。
全量 17 题上四个指标**全部一致** —— 这是比 §13 那次 3 题强得多的证据，证明 zip 回填没错位。

### 状态

全套 **80 passed**；`run_all --tier p0` 全绿 34 s。产物 `evaluation/results/native_20261106_093451.json`
与日志 `full_mimo_run.log`。仍未 commit。
