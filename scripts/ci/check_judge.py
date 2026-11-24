#!/usr/bin/env python
"""裁判连通性探针：花约 25 token 发一次真实请求，断言裁判能返回可解析的非空 content。

为什么需要：`resolve_judge()` 的预检只能验"key 存在 + 档位合法"，是离线的。
而 §15 实测到一个真实失败模式 —— MiMo 是思考型模型，输出预算被 reasoning 吃满时
`content` 会是空串（`reasoning_content` 里有话）。ragas 拿不到 content 就出不了分，
那种情况在全量 17 题跑完 14 分钟后才暴露，代价太大。

探针覆盖预检覆盖不到的三件事：
  1. key 真的能通过鉴权（401 类）
  2. 模型 ID 真的存在（404 / 返回体没有 choices）
  3. 返回的 content 非空可解析 —— 这条是 ragas 能不能用的前提

配置一律走 evaluate.resolve_judge()，与真实评估同一份事实来源，避免探针自己漂。

用法：python check_judge.py [--timeout 90]
退出码：0 可用 / 1 鉴权与连通都过但 content 为空（裁判不可用）/ 2 前置失败（key、网络、模型 ID）
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

CI_DIR = Path(__file__).resolve().parent
REPO = CI_DIR.parents[1]
for p in (str(REPO / "src"), str(REPO / "evaluation"), str(CI_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)

EXIT_PASS, EXIT_ASSERT, EXIT_PREREQ = 0, 1, 2

# 与 ragas 实际问法保持同量级的输出预算。刻意不用小 max_tokens：
# 实测 16/256 时思考链会吃满预算导致 content 空，那是参数问题不是裁判问题。
PROBE_PROMPT = "只回答两个字：通了"
PROBE_MAX_TOKENS = 512


def load_dotenv_if_present() -> None:
    """裸跑本文件时也能工作（run_eval.sh 已经在 shell 层载过一次）。

    与 ci_lib.sh 的 load_dotenv 语义一致：KEY=VALUE、# 注释、剥成对引号、剥 CR、
    已存在的环境变量优先（命令行显式传入 > .env）。
    不这样做的后果就是之前踩过的"直接 python evaluate.py 报缺 key"。
    """
    envf = REPO / ".env"
    if not envf.exists():
        return
    for line in envf.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        k, v = k.strip(), v.strip()
        if len(v) >= 2 and v[0] == v[-1] and v[0] in "\"'":
            v = v[1:-1]
        os.environ.setdefault(k, v)


def probe(cfg: dict, timeout: int) -> tuple[str, int, object]:
    """发一次真实请求，返回 (content, reasoning 字符数, usage)。异常向上抛，由 run 分类。"""
    from openai import OpenAI

    client = OpenAI(api_key=os.environ[cfg["key_env"]], base_url=cfg["base_url"], timeout=timeout)
    resp = client.chat.completions.create(
        model=cfg["model"],
        messages=[{"role": "user", "content": PROBE_PROMPT}],
        max_tokens=PROBE_MAX_TOKENS,
        temperature=0,          # 与 evaluate.py 构造裁判时的参数保持一致
    )
    msg = resp.choices[0].message
    content = (msg.content or "").strip()
    reasoning = len(getattr(msg, "reasoning_content", None) or "")
    return content, reasoning, resp.usage


def run(timeout: int) -> int:
    from evaluate import resolve_judge

    try:
        cfg = resolve_judge()
    except (ValueError, OSError) as e:
        print(f"[CI][SKIP] 裁判配置不可用: {e}")
        return EXIT_PREREQ

    print(f"[CI] 探针目标：{cfg['name']} / {cfg['model']} @ {cfg['base_url']}")
    t0 = time.perf_counter()
    try:
        content, reasoning, usage = probe(cfg, timeout)
    except Exception as e:                                    # noqa: BLE001 —— 分类靠消息文本
        name = type(e).__name__
        msg = str(e)[:300]
        low = msg.lower()
        if "401" in low or "api_key" in low or "unauthorized" in low or "invalid" in low:
            print(f"[CI][FAIL] 鉴权失败（{name}）：{msg}")
            print(f"  => {cfg['key_env']} 不对/已过期，或 Key 与 Base URL 不是同一集群"
                  "（交叉搭配是最常见的 401 来源）")
        elif "404" in low or "not found" in low or "model" in low and "exist" in low:
            print(f"[CI][FAIL] 模型 ID 不被接受（{name}）：{msg}")
            print(f"  => 用 {cfg['name'].upper()}_MODEL 覆盖试其它 ID；"
                  "列表见 GET /v1/models")
        else:
            print(f"[CI][FAIL] 请求失败（{name}）：{msg}")
            print("  => 先确认 Base URL 与网络可达，再怀疑模型 ID")
        return EXIT_PREREQ

    dur = time.perf_counter() - t0
    tok = getattr(usage, "total_tokens", "?") if usage is not None else "?"
    print(f"[CI] 往返 {dur:.1f}s | usage tokens={tok} | reasoning 长度={reasoning}")

    if not content:
        print(f"[CI][FAIL] 鉴权与连通都过了，但 content 为空"
              f"（回复疑似全在 reasoning_content，长度 {reasoning}）")
        print("  => ragas 拿不到判定文本，全量跑只会得到无意义分数。")
        print(f"     临时绕法：{cfg['name'].upper()}_MODEL 换个非思考档，"
              "或在 evaluate.py 的裁判构造处显式关思考。")
        return EXIT_ASSERT

    print(f"[CI][PASS] 裁判可用，content={content[:24]!r}")
    return EXIT_PASS


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--timeout", type=int, default=90)
    a = ap.parse_args()
    # 只在 CLI 入口载 .env：run() 保持纯逻辑，测试不会被开发者本机的 .env 污染
    load_dotenv_if_present()
    return run(a.timeout)


if __name__ == "__main__":
    sys.exit(main())
