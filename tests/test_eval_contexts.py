"""守卫：evaluate.ragas_contexts 必须与 build_context 的「来源标签 + 正文」口径一致。

钉住这次改动的核心行为：送给 RAGAS 的上下文要带上来源标签（含 stock_code），
不能退回裸正文；且逐条不增不减、空输入返回空表、父块拼装走 resolve_chunk_content。
每个断言只盯一个不变量，便于「反向突变」时每个突变恰好让且仅让一个用例变红。

纯离线：只用字典 + 真实 build_context/resolve_chunk_content，不建索引、不联网。
"""
from evaluate import ragas_contexts


def _item(**over):
    base = {
        "chunk_id": "000858_2021_001",
        "content": "正文内容ABC",
        "stock_code": "000858",
        "year": "2021",
        "page_num": 10,
        "section": "第二节 主要财务指标",
    }
    base.update(over)
    return base


def test_labeled_prefix():
    """上下文带来源标签，且标签里有 stock_code、年份「年报」与章节名（裸正文会没有）。"""
    ctxs = ragas_contexts([_item()], {})
    assert ctxs[0].startswith("[1] ")
    assert "000858" in ctxs[0] and "年报" in ctxs[0]
    assert "第二节 主要财务指标" in ctxs[0]


def test_content_kept():
    """标签之外，检索到的正文本身仍完整保留在上下文里。"""
    it = _item()
    ctxs = ragas_contexts([it], {})
    assert it["content"] in ctxs[0]


def test_count_preserved():
    """几条检索结果就产出几条上下文，不多不少。"""
    ctxs = ragas_contexts([_item(), _item()], {})
    assert len(ctxs) == 2


def test_empty():
    """空检索返回空表，而不是含一个空字符串的列表（否则会让 RAGAS 误判有上下文）。"""
    assert ragas_contexts([], {}) == []


def test_parent_resolution():
    """有父块时拼的是父块全文、不是子块正文（与作答模型看到的父块一致）。"""
    child = _item(content="子块正文XYZ", parent_id="P1")
    ctxs = ragas_contexts([child], {"P1": "父块完整正文WWW"})
    assert "父块完整正文WWW" in ctxs[0]
    assert "子块正文XYZ" not in ctxs[0]
