# 本地 CI/CD 操作速查

只讲怎么跑。设计理由、实测依据、待拍板项在 `docs/cicd_requirements.md`（下称"需求文档"）。

前置：仓库根目录、Git Bash、`.env` 里有 key（`.env` 已被 gitignore，不会提交）。

---

## 一条命令跑全套

```sh
sh scripts/ci/run_all.sh --tier p0p1
```

实测 35–80 s（波动来自机器负载）。想快就 `--tier p0`（静态+语料+索引），再配 `--fast-index` 把索引门从约 26 s 压到 0.1 s。

**平时不用手动跑**：`git commit` 自动跑静态门，`git push` 自动跑索引门 + 单测门。服务门和评估因为要冷启动/花钱，故意不进钩子。语料门只挂在 `run_all.sh` 上（不到 1 s、零网络），没进钩子是因为它拦的是**产物**而不是代码，提交时代码改了语料没动是常态。

---

## 五道门分别是什么

| 门 | 命令 | 查什么 | 需要 key |
|---|---|---|---|
| 静态 | `sh scripts/ci/check_static.sh` | 27 个 `.py` 做 `py_compile` + 单进程 import 冒烟 8/8 | 否 |
| 语料 | `sh scripts/ci/check_corpus.sh` | `data/raw_pdf` + `manifest.json` 对账：15 条齐全、每份 PDF 头尾结构完整、清单外孤儿 PDF、残留 `.part`。**排在索引门之前**——索引门只证明派生物自洽，语料缺半它照样绿 | 否 |
| 索引 | `sh scripts/ci/check_index.sh` | 13 条断言：7931×1024 自洽、必需字段存在、评测集题数与题型分布、真实 BM25 检索命中、**过滤语义在真实语料上生效** | 否 |
| 单测 | `sh scripts/ci/check_tests.sh` | 270 条纯函数守卫（分块/引用格式化/RRF/指标计算/落盘/裁判配置/探针分类/Qwen-VL 响应解析/拒答分类/索引选择/页眉页脚噪声/**年报下载与语料校验**） | 否 |
| 服务 | `sh scripts/ci/check_service.sh --port 8011` | 拉起 uvicorn → 轮询 `/health` 到 200 → 关停并确认端口释放 | 是 |

语料门就是 `python src/download_reports.py --verify` 的一层外壳，可以单独手敲。索引门的 `--allow-drift` 用于"你确实改了语料重建了索引"之后复核；长期基线改 `scripts/ci/baseline.json`。

---

## 评估与归因（花钱档）

```sh
sh scripts/ci/run_eval.sh --question-ids 1,2,3   # 冒烟，实测 32–105 s
sh scripts/ci/run_eval.sh                        # 全量 20 题，实测 393 s
python scripts/ci/analyze_scores.py --top 8      # 归因，不花钱，只读最新产物
```

`run_eval.sh` 会先跑裁判连通性探针（约 45 token）再放行 —— 避免全量跑到一半才发现裁判返回空 content。

切裁判档位：

```sh
RAGAS_LLM_PROVIDER=deepseek sh scripts/ci/run_eval.sh --question-ids 1
```

换集群或试模型 ID 不用改源码，加环境变量 `MIMO_BASE_URL` / `MIMO_MODEL` 即可。

---

## 退出码

| 码 | 含义 | 下一步 |
|---|---|---|
| 0 | 通过 | —— |
| 1 | **断言失败，是真回归** | 看 `[CI][FAIL]` 行的"期望 vs 实际"，命令会直接指向改动的地方 |
| 2 | **前置缺失**（没 key / 没语料 / 没索引 / 没 pytest） | 不是代码问题。语料门会让你先跑 `python src/download_reports.py`，索引门会给重建命令，服务门会让你填 `.env` |

## 逃生阀

```sh
CI_SKIP=1 git commit -m "..."
```

会打印醒目警告。别用 `--no-verify` —— 那个连日志都不留。

---

## 换机器 / 重新 clone 必做两条

```sh
git config core.hooksPath scripts/git-hooks
git update-index --chmod=+x scripts/git-hooks/pre-commit scripts/git-hooks/pre-push
```

**不跑不会报错，就是不执行。** 本仓库 `core.filemode=false`，clone 会丢可执行位。
自检：`git ls-files -s scripts/git-hooks` 应显示 `100755`。详见 `scripts/git-hooks/README.md`。

---

## 结果在哪看

| 路径 | 内容 |
|---|---|
| `evaluation/results/ci-latest.json` | 四道门的状态、耗时、断言明细 |
| `evaluation/results/native_<时间戳>.json` | 评估产物，含**逐题** ragas 分数 |
| `evaluation/results/native_<时间戳>.csv` | 同上表格版，四个指标各一列，排序即得归因 |
| `evaluation/results/pytest-latest.log` | 单测完整输出 |
| `evaluation/results/service-latest.log` | 服务启动日志 |

均在 gitignore 内，不会提交。

---

## 五个真会咬人的坑

1. **别用裸 `python`。** 本机全局是 3.14.5 且缺项目依赖，会得到一堆假 `ModuleNotFoundError`。
   CI 内部已钉死 conda env `ailearning` 的 3.12；手动跑脚本用完整路径或先 `conda activate ailearning`。
2. **带中文的 JSON 别写在 curl 命令行上**（编码会坏，报 400 `error parsing the body`），
   用 `--data-binary @文件`。
3. **Git Bash 没有 `pgrep`**，命令未找到会让 `||` 分支误报"进程已结束"。查进程走 PowerShell 脚本文件。
4. **内联 PowerShell 别夹中文**，PS5 按 ANSI 解析会炸引号；写成 `.ps1` 文件再执行。
5. **改了检索/分块逻辑后指标会变**，`analyze_scores.py` 会自检逐题分数与顶层均值是否自洽 ——
   对不上就是落盘错位，别信那张表。

---

## 想加新守卫

放 `tests/`，遵循既有约定：不联网、不依赖 key、不读磁盘大产物。
每条新守卫都要做反向验证（故意把被测代码改坏 → 确认变红 → 还原 → sha256 比对一致），
做法与已有 8 组测试一致，例子见 `tests/test_filter.py` 和 `tests/test_judge_probe.py`。
特别注意：**每个分支都要被跑到**。曾经有 6 条测试全绿却漏了一个 `NameError`，
因为所有测试都显式传了 `--file`，自动发现那条分支一次没执行过。
