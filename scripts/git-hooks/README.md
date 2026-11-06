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
| `pre-commit` | 提交前 | P0-1 静态门：8 文件 `py_compile` + 单进程 import 冒烟 | 见 `docs/cicd_requirements.md` §8 |
| `pre-push` | 推送前 | P0-2 索引门（含 BM25 建库 + 真实检索）+ P1-1 单测门（35 条） | ≈ 11 s + ≈ 2 s |

服务门（P1-2，需 `DASHSCOPE_API_KEY`）不在钩子里，手动：

```sh
sh scripts/ci/run_all.sh --tier p0p1        # 全套
sh scripts/ci/run_all.sh --tier p0 --fast-index   # 快速自检
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
