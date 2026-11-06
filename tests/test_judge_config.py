"""RAGAS 裁判可切换配置的回归守卫（离线，不联网）。

约束：Embedding 恒走 DashScope —— MiMo Token Plan 的公开资料里没有 embeddings 端点，
而 answer_relevancy 之类别的就是向量相似度。所以这里既钉住"裁判可换"，
也钉住"embedding 不许被顺手换掉"。
"""
from __future__ import annotations

import pytest

import evaluate

JUDGE_ENVS = ("MIMO_API_KEY", "DEEPSEEK_API_KEY", "RAGAS_LLM_PROVIDER",
              "MIMO_BASE_URL", "MIMO_MODEL", "DEEPSEEK_BASE_URL", "DEEPSEEK_MODEL")


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    for v in JUDGE_ENVS:
        monkeypatch.delenv(v, raising=False)


def test_default_judge_is_mimo_token_plan(monkeypatch):
    monkeypatch.setenv("MIMO_API_KEY", "tp-real-looking-key")
    cfg = evaluate.resolve_judge()
    assert cfg["name"] == "mimo"
    assert cfg["base_url"] == "https://token-plan-cn.xiaomimimo.com/v1"
    assert cfg["model"] == "mimo-v2.6-flash"
    assert cfg["key_env"] == "MIMO_API_KEY"


def test_deepseek_preset_keeps_old_ruler_verbatim(monkeypatch):
    """旧结论是在这套参数上跑出来的，切换档位必须能一字不差回到旧口径。"""
    monkeypatch.setenv("RAGAS_LLM_PROVIDER", "DeepSeek")     # 大小写不敏感
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-old-key")
    cfg = evaluate.resolve_judge()
    assert cfg["base_url"] == "https://api.deepseek.com"
    assert cfg["model"] == "deepseek-v4-flash"
    assert cfg["key_env"] == "DEEPSEEK_API_KEY"


def test_unknown_provider_lists_choices(monkeypatch):
    monkeypatch.setenv("RAGAS_LLM_PROVIDER", "openai")
    with pytest.raises(ValueError) as e:
        evaluate.resolve_judge()
    assert "deepseek" in str(e.value) and "mimo" in str(e.value)


@pytest.mark.parametrize("bad", ["", "sk-xxxxxxxx", "tp-xxxxxxxx"])
def test_missing_or_placeholder_key_fails_fast(monkeypatch, bad):
    """缺 key 必须自己报错，不能让 openai SDK 抛一句看不懂的鉴权失败。"""
    monkeypatch.setenv("MIMO_API_KEY", bad)
    with pytest.raises(EnvironmentError) as e:
        evaluate.resolve_judge()
    assert "MIMO_API_KEY" in str(e.value)
    assert "RAGAS_LLM_PROVIDER=deepseek" in str(e.value)   # 给出退路


def test_env_overrides_take_precedence(monkeypatch):
    """换集群 / 试模型 ID 不该动源码。"""
    monkeypatch.setenv("MIMO_API_KEY", "tp-x")
    monkeypatch.setenv("MIMO_BASE_URL", "https://token-plan-sgp.xiaomimimo.com/v1")
    monkeypatch.setenv("MIMO_MODEL", "mimo-v2.6-flash-rl")
    cfg = evaluate.resolve_judge()
    assert cfg["base_url"].endswith("sgp.xiaomimimo.com/v1")
    assert cfg["model"] == "mimo-v2.6-flash-rl"


def test_embeddings_stay_on_dashscope():
    """裁判换成 MiMo 后，embedding 仍必须是 DashScope —— 否则 answer_relevancy 直接崩。"""
    assert evaluate.EMBED_MODEL == "text-embedding-v3"
    assert evaluate.DASHSCOPE_URL == "https://dashscope.aliyuncs.com/compatible-mode/v1"


def test_preflight_blocks_before_paying(monkeypatch):
    """main() 必须在建索引/生成答案之前就因缺 key 退出。
    答案生成是真花钱的（全量 20 题实测 292 s），先验后花钱。"""
    monkeypatch.setattr("sys.argv", ["evaluate.py", "--question-ids", "1"])

    def boom(*a, **k):
        raise AssertionError("预检失败却仍然去跑 RAG 了 —— 会白花钱")

    monkeypatch.setattr(evaluate, "run_native_rag", boom)
    monkeypatch.delenv("MIMO_API_KEY", raising=False)

    with pytest.raises(SystemExit) as e:
        evaluate.main()
    assert e.value.code == 2
