# git 钩子（本地 CI 入口）

钩子脚本在版本库里，但 **`core.hooksPath` 是本机配置，不会被 clone 带过来**。
换机器 / 重新 clone 后必须重跑下面两条，否则钩子一个都不会执行 —— 而且**不会报错，就是不跑**。

```sh
git config core.hooksPath scripts/git-hooks
git update-index --chmod=+x scripts/git-hooks/pre-commit scripts/git-hooks/pre-push
```

第二条是 Windows 上的命门：本仓库实测 `core.filemode=false`，git 不跟可执行位，
重新 clone 出来的钩子脚本会丢掉 `100755`，Git 直接跳过它。

自检（应当输出 `100755`）：

```sh
git ls-files -s scripts/git-hooks
```

## 各钩子跑什么

| 钩子 | 阶段 | 内容 | 实测耗时 |
|---|---|---|---|
| `pre-commit` | 提交前 | P0-1 静态门：27 个 `.py` 做 `py_compile` + 单进程 import 冒烟 8/8 | 见 `docs/cicd_requirements.md` §8 |
| `pre-push` | 推送前 | P0-1.5 语料门（零网络，`manifest` 与 `data/raw_pdf` 对账）+ P0-2 索引门（含 BM25 建库 + 真实检索）+ P1-1 单测门 | ≈ 1 s + ≈ 11 s + ≈ 2 s |
| `pre-push`（条件） | 推送前 | 改动 `parse_pdf.py` / `chunk_documents.py` / `rag_pipeline.py` / 题集时，跑 RAGAS 6 题冒烟对比 | ≈ 2~4 min，报告不拦人 |

语料门放 pre-push 而不放 pre-commit：它校验的是**产物**而不是代码，提交时代码改了语料没动是
常态；但推出去之前必须保证别人 clone 完能复现 —— 半截 PDF 或缺两家公司的 manifest，索引门
照样全绿（详见 `docs/cicd_requirements.md` §17）。

RAGAS 对比门的基线是 `scripts/ci/ragas_baseline.json`。第一次跑（或主动
`--update-baseline`）会建立基线；没有 `DASHSCOPE_API_KEY` / 裁判 key 时跳过，
不影响推送。`CI_EVAL=0 git push` 关闭自动对比，`CI_EVAL_STRICT=1` 把指标回退当失败。

服务门（P1-2，需 `DASHSCOPE_API_KEY`）不在钩子里，手动：

```sh
sh scripts/ci/run_all.sh --tier p0p1        # 全套
sh scripts/ci/run_all.sh --tier p0 --fast-index   # 快速自检
sh scripts/ci/run_all.sh --tier eval        # 只跑 RAGAS 基线对比
```

## 逃生阀

```sh
CI_SKIP=1 git commit -m "..."
CI_SKIP=1 git push
```

会打印醒目警告后放行，并在 `evaluation/results/ci-latest.json` 留下缺口 —— 用它就得自己负责。
不要用 `--no-verify`：那个连日志都不留。

## 依赖

- 解释器：默认 `D:/softwares/miniconda/envs/ailearning/python.exe`（Python 3.12，实测项目运行环境）。
  用 `CI_PYTHON=<路径>` 覆盖。**版本不是 3.12 会直接以退出码 2 拒绝**，不会回落到全局 3.14（缺依赖，会得到假故障）。
- 单测门需要 `pytest`；缺了会报退出码 2 并给出安装命令。
