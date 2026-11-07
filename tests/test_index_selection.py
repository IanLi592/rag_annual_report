"""索引选择与父块拼装的守卫。

`--index-dir` / RAG_INDEX_DIR 让线上在 semantic 与 structured 索引之间切换；
父块拼装让 structured 的"子块召回、父块送 LLM"真正生效。
这两处最容易出现"文件换成 structured 了，但 parents 没载入"的静默降级。
"""
from __future__ import annotations

import json

import pytest

import rag_pipeline as rp


# ── resolve_chunk_content 优先级 ─────────────────────────────────────────────

def test_parent_table_wins_over_inline_and_self():
    item = {"chunk_id": "c1", "parent_id": "p1",
            "parent_content": "内联父块", "content": "子块"}
    out = rp.resolve_chunk_content(item, {"p1": "查表父块"})
    assert out == "查表父块"


def test_inline_parent_content_is_fallback():
    item = {"parent_id": "p-missing", "parent_content": "内联父块", "content": "子块"}
    assert rp.resolve_chunk_content(item, {}) == "内联父块"


def test_child_content_is_last_resort():
    assert rp.resolve_chunk_content({"content": "子块"}, {}) == "子块"


def test_empty_parent_table_does_not_crash():
    assert rp.resolve_chunk_content({}, None) == ""


def test_build_context_uses_parents_for_llm_text():
    retrieved = [{"chunk_id": "c1", "parent_id": "p1", "content": "子块",
                  "stock_code": "600519", "year": "2023", "page_num": 38}]
    ctx, cits = rp.build_context(retrieved, {"p1": "父块全文，含完整表格"})
    assert "父块全文，含完整表格" in ctx
    assert "子块" not in ctx
    assert cits[0]["chunk_id"] == "c1"


def test_build_context_without_parents_is_backward_compatible():
    retrieved = [{"chunk_id": "c1", "content": "正文", "stock_code": "6", "year": "1"}]
    ctx, _ = rp.build_context(retrieved)
    assert "正文" in ctx


# ── 索引目录与父块载入 ──────────────────────────────────────────────────────

class FakeVectorStore:
    def __init__(self, client, index_dir=None):
        self.index_dir = index_dir
        self.meta_list = [{"strategy": "structured"}]


@pytest.fixture
def fake_pipeline_env(monkeypatch):
    monkeypatch.setattr(rp, "get_client", lambda: object())
    monkeypatch.setattr(rp, "VectorStore", FakeVectorStore)


def test_index_dir_is_used_and_parents_loaded(fake_pipeline_env, tmp_path):
    (tmp_path / "parents.json").write_text(json.dumps([
        {"parent_id": "p1", "content": "父块一"},
        {"parent_id": "p2", "content": "父块二"},
    ], ensure_ascii=False), encoding="utf-8")

    pipe = rp.RAGPipeline(use_bm25=False, use_rerank=False, index_dir=tmp_path)
    assert pipe.index_dir == tmp_path
    assert pipe.vec_store.index_dir == tmp_path
    assert pipe.parents == {"p1": "父块一", "p2": "父块二"}
    assert pipe.strategy == "structured", "strategy 要从索引 meta 读出来，/health 依赖它"


def test_missing_parents_file_keeps_semantic_behavior(fake_pipeline_env, tmp_path):
    pipe = rp.RAGPipeline(use_bm25=False, use_rerank=False, index_dir=tmp_path)
    assert pipe.parents == {}, "semantic 索引没有 parents.json，不能因此启动失败"


def test_default_index_dir_is_main_vectorstore(fake_pipeline_env):
    pipe = rp.RAGPipeline(use_bm25=False, use_rerank=False)
    assert pipe.index_dir == rp.VECTORSTORE_DIR
    assert pipe.vec_store.index_dir is None or pipe.vec_store.index_dir == rp.VECTORSTORE_DIR
