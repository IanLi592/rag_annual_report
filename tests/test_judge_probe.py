"""裁判连通性探针 check_judge.py 的离线守卫（不发真实请求）。

探针存在的意义是拦住三类"离线预检看不到"的失败：鉴权、模型 ID、content 为空。
这里逐个把这三类喂给它，确认它分类正确、退出码符合 CI 约定。
"""
from __future__ import annotations

import os
from pathlib import Path

import check_judge as cj
import pytest

KEY_ENVS = ("MIMO_API_KEY", "DEEPSEEK_API_KEY", "RAGAS_LLM_PROVIDER",
            "MIMO_BASE_URL", "MIMO_MODEL")


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    for v in KEY_ENVS:
        monkeypatch.delenv(v, raising=False)
    monkeypatch.setenv("MIMO_API_KEY", "tp-looking-real")


class FakeUsage:
    total_tokens = 25


def fake_probe(content, reasoning=0, usage=None):
    def _p(cfg, timeout):
        _p.seen = cfg
        return content, reasoning, usage if usage is not None else FakeUsage()
    _p.seen = None
    return _p


def test_nonempty_content_passes(monkeypatch, capsys):
    monkeypatch.setattr(cj, "probe", fake_probe("通了"))
    assert cj.run(30) == cj.EXIT_PASS
    assert "裁判可用" in capsys.readouterr().out


def test_empty_content_is_assert_failure(monkeypatch, capsys):
    """鉴权过、连通过、但 content 空 —— §15 实测到的那个失败模式，判 1 不判 2。"""
    monkeypatch.setattr(cj, "probe", fake_probe("", reasoning=70))
    assert cj.run(30) == cj.EXIT_ASSERT
    out = capsys.readouterr().out
    assert "content 为空" in out and "reasoning" in out


def test_auth_failure_is_prereq(monkeypatch, capsys):
    def boom(cfg, timeout):
        raise RuntimeError("Error code: 401 - {'error': {'message': 'Invalid API key'}}")
    monkeypatch.setattr(cj, "probe", boom)
    assert cj.run(30) == cj.EXIT_PREREQ
    assert "鉴权失败" in capsys.readouterr().out


def test_unknown_model_id_is_prereq(monkeypatch, capsys):
    def boom(cfg, timeout):
        raise RuntimeError("Error code: 404 - model 'mimo-v2.6-fla' not found")
    monkeypatch.setattr(cj, "probe", boom)
    assert cj.run(30) == cj.EXIT_PREREQ
    assert "模型 ID" in capsys.readouterr().out


def test_network_failure_is_prereq(monkeypatch, capsys):
    def boom(cfg, timeout):
        raise OSError("connection refused")
    monkeypatch.setattr(cj, "probe", boom)
    assert cj.run(30) == cj.EXIT_PREREQ
    assert "请求失败" in capsys.readouterr().out


def test_probe_uses_the_same_judge_config_as_evaluation(monkeypatch):
    """单一事实来源：探针必须用 evaluate.resolve_judge()，不许自己拼一份配置。
    否则改了预设而探针还在测老地址，探针就变成了假绿灯。"""
    seen = fake_probe("通了")
    monkeypatch.setattr(cj, "probe", seen)
    cj.run(30)
    assert seen.seen["name"] == "mimo"
    assert seen.seen["model"] == "mimo-v2.6-flash"
    assert seen.seen["base_url"] == "https://token-plan-cn.xiaomimimo.com/v1"

    monkeypatch.setenv("MIMO_MODEL", "mimo-v2.6-pro")
    seen2 = fake_probe("通了")
    monkeypatch.setattr(cj, "probe", seen2)
    cj.run(30)
    assert seen2.seen["model"] == "mimo-v2.6-pro", "覆盖的模型 ID 没被探针采用"


def test_probe_requests_enough_output_budget(monkeypatch):
    """小 max_tokens 会让思考型模型把预算吃满、content 变空 —— 那是参数的锅。
    钉住探针用的是接近真实裁判的量级。"""
    assert cj.PROBE_MAX_TOKENS >= 512


def test_missing_key_short_circuits_before_network(monkeypatch):
    """key 没填时不该去发请求（会白跑），应直接走配置层退出码 2。"""
    monkeypatch.delenv("MIMO_API_KEY")
    called = []
    def boom(cfg, timeout):
        called.append(1); return "x", 0, FakeUsage()
    monkeypatch.setattr(cj, "probe", boom)
    assert cj.run(30) == cj.EXIT_PREREQ
    assert called == []


def test_run_does_not_read_the_developers_dotenv(monkeypatch):
    """run() 必须与本机 .env 隔离，否则上面那条"缺 key"测试会被真实 key 悄悄救活，
    测试结果随机器状态漂移 —— 这类隐患比 bug 本身难查。"""
    monkeypatch.delenv("MIMO_API_KEY")
    monkeypatch.setattr(cj, "REPO", Path("Z:/definitely-not-a-real-path-xyz"))
    assert cj.run(30) == cj.EXIT_PREREQ      # 没载到任何 .env，仍旧判缺 key


def test_dotenv_loader_strips_quotes_and_keeps_existing(monkeypatch, tmp_path):
    """CLI 侧的 .env 解析：剥成对引号、剥 CR、已存在的环境变量优先。"""
    envf = tmp_path / ".env"
    envf.write_bytes(b'PROBE_TEST_A="value-a"\r\nPROBE_TEST_B=value-b\r\n# c\r\n')
    monkeypatch.setattr(cj, "REPO", tmp_path)
    monkeypatch.setenv("PROBE_TEST_A", "already-set")
    cj.load_dotenv_if_present()
    assert os.environ["PROBE_TEST_A"] == "already-set", "不该覆盖显式传入的环境变量"
    assert os.environ["PROBE_TEST_B"] == "value-b"
    assert '"' not in os.environ.get("PROBE_TEST_A", "")
