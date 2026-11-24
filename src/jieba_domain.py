"""A 股领域词表的统一加载入口。

设计要点（和 match_filter 同一个原则：**两条 BM25 链路必须同源**）：
  - `rag_pipeline.BM25Store` 用一份
  - `compare_strategies.retrieve_bm25`（消融实验）用同一份
否则线上检索和消融结论会建立在不同的分词口径上，那种差异最难查。

作用范围是**中文分词**，不是 embedding —— text-embedding-v3 的分词在模型内部，
外部词典影响不到它。所以本模块只服务于 BM25 关键词召回。

A/B 开关：环境变量 `JIEBA_DOMAIN_DICT=0` 时不加载，用于量化词表收益。
"""
from __future__ import annotations

import os
from pathlib import Path

# 默认词表；用 DOMAIN_DICT_PATH 可指向别的版本（A/B 不同词表规模用）
DEFAULT_DICT = Path(__file__).parent / "domain_dict.txt"

_loaded = False
_stats: dict = {"loaded": 0, "path": str(DEFAULT_DICT), "enabled": True}


def dict_path() -> Path:
    p = os.getenv("DOMAIN_DICT_PATH", "").strip()
    return Path(p) if p else DEFAULT_DICT


def load_domain_dict(jieba_module=None) -> dict:
    """把词表灌进 jieba。幂等：重复调用只加载一次。

    必须在任何 cut() 之前调用 —— jieba 的前缀词典一旦建好，中途加词不会生效
    （除非 resetDictionary，那个不在公开 API 里，别依赖）。
    """
    global _loaded
    enabled = os.getenv("JIEBA_DOMAIN_DICT", "1").strip() != "0"
    _stats["enabled"] = enabled
    if not enabled:
        # 关闭时报告 loaded=0，否则 A/B 的对照组看起来"也加载了"，测出来的差异就是假的
        _stats["loaded"] = 0
        return dict(_stats)

    if _loaded:
        return dict(_stats)

    if jieba_module is None:
        import jieba
        jieba_module = jieba

    path = dict_path()
    _stats["path"] = str(path)
    if not path.exists():
        raise FileNotFoundError(
            f"领域词表缺失: {path}\n  它必须随代码入库（不是 data/ 下的运行产物）")

    words = [ln.strip() for ln in path.read_text(encoding="utf-8").splitlines()
             if ln.strip() and not ln.strip().startswith("#")]
    if not words:
        raise ValueError(f"领域词表为空: {path}")

    # dict_freq=False → 保留文件里的词频（我们没写词频，jieba 会用默认构造）
    for w in words:
        jieba_module.add_word(w, freq=20000)
    _loaded = True
    _stats["loaded"] = len(words)
    return dict(_stats)


def is_loaded() -> bool:
    return _loaded


def reset_for_test() -> None:
    """仅供测试重置幂等标记，生产代码不要调。"""
    global _loaded
    _loaded = False
