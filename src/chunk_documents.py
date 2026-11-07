"""
文档分块脚本：对解析后的年报做分块处理

三种分块策略：
  策略A  固定大小分块  —— 最简单，但会切断句子/表格
  策略B  语义分块      —— 按段落/章节边界切，保留语义完整性
  策略C  层级分块      —— 父子块：父块用于召回上下文，子块用于精确匹配

企业级 RAG 通常用 B 或 C，
让学生先跑通 A，再体会 B/C 在召回效果上的区别。

输出格式说明：
  每个 chunk 是一个 dict，包含：
    - chunk_id      唯一标识
    - content       文本内容（供 embedding）
    - metadata      元信息（供过滤/溯源）
      - stock_code  股票代码
      - year        年份
      - page_num    来源页码
      - section     章节路径（字符串）
      - block_type  text/table/title
      - is_ocr      是否 OCR 结果
      - strategy    分块策略名
"""

import json
import uuid
import logging
from pathlib import Path
from typing import Iterator

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

PARSED_DIR = Path(__file__).parent.parent / "data" / "parsed"
CHUNKS_DIR = Path(__file__).parent.parent / "data" / "chunks"
CHUNKS_DIR.mkdir(parents=True, exist_ok=True)


# ── 策略 A：固定大小分块 ──────────────────────────────────────────────────────

def chunk_fixed(
    text: str,
    chunk_size: int = 500,
    overlap: int = 50,
) -> Iterator[str]:
    """
    按字符数切块，相邻块有重叠。
    缺点：无视句子/段落边界，表格会被切断。
    优点：实现最简单，块大小可预测。
    """
    start = 0
    while start < len(text):
        end = start + chunk_size
        yield text[start:end]
        start += chunk_size - overlap


# ── 策略 B：语义分块 ──────────────────────────────────────────────────────────

def chunk_semantic(
    blocks: list[dict],
    max_chunk_size: int = 800,
    min_chunk_size: int = 100,
) -> Iterator[dict]:
    """
    按解析结构分块：遇到标题强制切块，段落尽量合并到 max_chunk_size 以内。

    优点：保留语义完整性，章节边界清晰。
    缺点：块大小不均匀（财务报表单个表格可能很大）。
    """
    buffer_blocks = []
    buffer_len    = 0
    buffer_meta   = {}

    def flush(buf: list[dict]) -> dict | None:
        if not buf:
            return None
        content = "\n\n".join(b["content"] for b in buf)
        # 元信息取第一个块的
        meta = {
            "page_num":   buf[0]["page_num"],
            "section":    " > ".join(buf[0]["section_path"]) if buf[0]["section_path"] else "",
            "block_types": list({b["block_type"] for b in buf}),
            "is_ocr":     any(b["is_ocr"] for b in buf),
            "heading_level": buf[0].get("heading_level", 0),
        }
        return {"content": content, "metadata": meta}

    for block in blocks:
        btype = block["block_type"]
        blen  = len(block["content"])

        # 标题块：强制先 flush，标题单独作为一个小块（不用于 embedding，
        # 但保留在 chunk 里作为上下文前缀）
        if btype == "title":
            if buffer_blocks:
                result = flush(buffer_blocks)
                if result and len(result["content"]) >= min_chunk_size:
                    yield result
                buffer_blocks = []
                buffer_len    = 0

        # 表格块：单独成块（不与文字合并，防止 embedding 效果变差）
        if btype == "table":
            if buffer_blocks:
                result = flush(buffer_blocks)
                if result and len(result["content"]) >= min_chunk_size:
                    yield result
                buffer_blocks = []
                buffer_len    = 0
            yield {
                "content": block["content"],
                "metadata": {
                    "page_num": block["page_num"],
                    "section":  " > ".join(block["section_path"]),
                    "block_types": ["table"],
                    "is_ocr":   block["is_ocr"],
                    "heading_level": block.get("heading_level", 0),
                }
            }
            continue

        # 文字块：累积，超过 max_chunk_size 就 flush
        if buffer_len + blen > max_chunk_size and buffer_blocks:
            result = flush(buffer_blocks)
            if result and len(result["content"]) >= min_chunk_size:
                yield result
            buffer_blocks = []
            buffer_len    = 0

        buffer_blocks.append(block)
        buffer_len += blen

    # 尾部剩余
    if buffer_blocks:
        result = flush(buffer_blocks)
        if result and len(result["content"]) >= min_chunk_size:
            yield result


# ── 策略 C：层级分块（父子块） ────────────────────────────────────────────────

def chunk_hierarchical(
    blocks: list[dict],
    parent_size: int = 2000,
    child_size:  int = 400,
    overlap:     int = 50,
) -> Iterator[dict]:
    """
    两级结构：
      父块（parent）：大段落，用于给 LLM 提供足够上下文
      子块（child）：小段落，用于向量检索（更精确）

    检索时：命中子块 → 取父块 id → 给 LLM 读父块
    这就是所谓的 "小to大检索"（Small-to-Big Retrieval）

    输出的每个 child 块里带 parent_id 字段。
    """
    full_text  = "\n\n".join(b["content"] for b in blocks if b["content"].strip())
    pages_map  = {b["content"]: b["page_num"] for b in blocks}

    # 先切父块
    parents = []
    start   = 0
    while start < len(full_text):
        end     = min(start + parent_size, len(full_text))
        content = full_text[start:end]
        parent_id = str(uuid.uuid4())[:8]
        parents.append({
            "parent_id": parent_id,
            "content":   content,
            "start":     start,
            "end":       end,
        })
        start += parent_size - overlap

    # 再从父块里切子块
    for parent in parents:
        p_content = parent["content"]
        p_id      = parent["parent_id"]
        c_start   = 0
        while c_start < len(p_content):
            c_end     = min(c_start + child_size, len(p_content))
            child_content = p_content[c_start:c_end]
            yield {
                "content":  child_content,
                "metadata": {
                    "parent_id":   p_id,
                    "parent_content": p_content,   # 存全量，供 LLM 读
                    "block_types": ["text"],
                    "is_ocr":      False,
                    "section":     "",
                    "page_num":    -1,             # 层级分块时页码对应关系复杂，暂置-1
                }
            }
            c_start += child_size - overlap


# ── 策略 D：结构化分块（表格摘要注入 + 章节树 Parent-Child）───────────────────

def clean_table(raw: list) -> list:
    """去掉全空行/全空列，返回规整后的二维表。"""
    if not raw:
        return []
    rows = [[("" if c is None else str(c)).strip() for c in r] for r in raw]
    rows = [r for r in rows if any(r)]
    if not rows:
        return []
    width = max(len(r) for r in rows)
    rows = [r + [""] * (width - len(r)) for r in rows]
    keep_cols = [j for j in range(width) if any(r[j] for r in rows)]
    return [[r[j] for j in keep_cols] for r in rows]


def summarize_table(raw: list, section: str, max_rows: int = 10) -> str:
    """把二维表压成一段**可 embedding 的自然语言摘要**。

    为什么需要：表格拍平成 markdown 管道文本后，embedding 看到的是大量孤立的
    数字与竖线，语义信号很弱（这正是"核心财务指标检索不准"的一个来源）。
    摘要把"列名: 值"配对写出来，让向量召回能按指标名命中。
    原始结构另存 table_json 供精确提取，两者不互相替代。

    格式约定（第一列当行标签，其余列与表头配对）：
      【表格摘要】<章节>。行「营业收入」：2023年: 147,693,604,994.14，2022年: 124,099,843,771.99；…
    早先版本把第一列也当普通列配对，产出"项目: 营业收入"这种无信息量的对子。
    """
    t = clean_table(raw)
    if not t:
        return ""
    header, body = t[0], t[1:]
    lines = []
    for r in body[:max_rows]:
        label = r[0]
        pairs = [f"{h}: {v}" for h, v in zip(header[1:], r[1:]) if h and v]
        if pairs:
            lines.append(f"行「{label}」：" + "，".join(pairs))
    head = f"【表格摘要】{section}。" if section else "【表格摘要】"
    return head + "；".join(lines)


def chunk_structured(
    blocks: list[dict],
    max_child:  int = 800,
    min_child:  int = 100,
    parent_cap: int = 2000,
) -> tuple[list[dict], list[dict]]:
    """
    章节树 Parent-Child 分块，返回 (children, parents)。

    与策略 C 的区别（C 是按字符窗口硬切，父块整份复制进每个子块）：
      1. 树来自 section_path 的真实层级，显式分级 H1/H2/H3
      2. 父块 = 一个章节的内容拼接（带章节标题作全局背景），**单独存储**，
         子块只留 parent_id —— 修掉 C 的存储冗余
      3. 表格子块的 content 是摘要文本，raw 结构进 metadata.table_json
      4. 检索时"子块召回、父块拼装"：rag_pipeline 用 parent_id 查父块内容

    parents 元素：{"parent_id", "content", "section", "level", "page_start"}
    children 元素：与 semantic 同构（content/metadata），metadata 多 parent_id/level
    """
    import uuid

    def level_of(path):
        return min(len(path), 3) if path else 1

    # 按章节分组（用 section_path 全路径作键，保留出现顺序）
    groups: dict[tuple, list[dict]] = {}
    order: list[tuple] = []
    for b in blocks:
        key = tuple(b.get("section_path") or ["(无章节)"])
        if key not in groups:
            groups[key] = []
            order.append(key)
        groups[key].append(b)

    children, parents = [], []
    for key in order:
        grp = groups[key]
        section = " > ".join(key)
        # 优先用解析阶段显式标出的 H1/H2/H3；旧 parsed JSON 没有该字段时退回路径深度
        levels = [b.get("heading_level", 0) for b in grp if b.get("heading_level", 0)]
        level = max(levels) if levels else level_of(list(key))

        # 子块：段落累积 / 表格单独 / 标题并入下一段
        buf, buf_len = [], 0
        pending_titles: list[str] = []

        def flush_buf():
            nonlocal buf, buf_len
            if not buf:
                return None
            content = "\n\n".join(buf)
            buf, buf_len = [], 0
            return content

        child_chunks: list[dict] = []
        for b in grp:
            bt, txt = b["block_type"], b["content"]
            if bt == "title":
                pending_titles.append(txt)
                continue
            if bt == "table":
                pre = flush_buf()
                if pre and len(pre) >= min_child:
                    child_chunks.append(_mk_child(pre, b, pending_titles, level))
                pending_titles = []
                summ = summarize_table(b.get("raw_table"), section)
                if summ and len(summ) >= min_child // 2:
                    # 子块内容 = 摘要 + 原始 markdown 表格。
                    # 为什么两份都要：旧实现里表格文字同时存在于 table 块和 text 块，
                    # 去掉重复后 BM25 Hit@4 实测从 0.380 升到 0.400（§11），
                    # 说明这份“重复”并不是必要的召回增强。块内同时带摘要与原始
                    # markdown 不增加块数，既保留语义信号，又保留 table_json 的结构。
                    child_chunks.append({
                        "content": summ + "\n" + b["content"],
                        "metadata": {
                            "page_num": b["page_num"],
                            "section": section,
                            "block_types": ["table"],
                            "is_ocr": bool(b.get("is_ocr")),
                            "heading_level": b.get("heading_level", 0) or level,
                            "table_json": clean_table(b.get("raw_table")),
                        },
                    })
                continue
            # text
            # 单个块就超过 max_child 时必须先拆 —— 否则一个 6000 字的段落会
            # 原样变成一个子块（旧 semantic 策略也有这个毛病），父块上限随之失效
            if len(txt) > max_child:
                pre = flush_buf()
                if pre and len(pre) >= min_child:
                    child_chunks.append(_mk_child(pre, b, pending_titles, level))
                pending_titles = []
                for s0 in range(0, len(txt), max_child):
                    piece = txt[s0:s0 + max_child]
                    if len(piece) >= min_child:
                        child_chunks.append(_mk_child(piece, b, [], level))
                continue
            if buf_len + len(txt) > max_child and buf:
                c = flush_buf()
                if c and len(c) >= min_child:
                    child_chunks.append(_mk_child(c, b, pending_titles, level))
                pending_titles = []
            buf.append(txt)
            buf_len += len(txt)
        tail = flush_buf()
        if tail and len(tail) >= min_child:
            child_chunks.append(_mk_child(tail, grp[-1], pending_titles, level))

        if not child_chunks:
            continue

        # 父块：章节内容拼接，超 parent_cap 就按子块边界切多段
        pid = str(uuid.uuid4())[:8]
        acc, acc_len, page_start = [], 0, grp[0]["page_num"]
        for c in child_chunks:
            piece = c["content"]
            if acc_len + len(piece) > parent_cap and acc:
                parents.append({"parent_id": pid, "content": "\n\n".join(acc),
                                "section": section, "level": level,
                                "heading_level": level,
                                "page_start": page_start})
                pid = str(uuid.uuid4())[:8]
                acc, acc_len, page_start = [], 0, c["metadata"]["page_num"]
            acc.append(piece)
            acc_len += len(piece)
            c["metadata"]["parent_id"] = pid
        if acc:
            parents.append({"parent_id": pid, "content": "\n\n".join(acc),
                            "section": section, "level": level,
                            "heading_level": level,
                            "page_start": page_start})
        children.extend(child_chunks)

    return children, parents


def _mk_child(content: str, anchor: dict, titles: list[str], heading_level: int = 0) -> dict:
    prefix = (" / ".join(titles) + "\n") if titles else ""
    return {
        "content": prefix + content,
        "metadata": {
            "page_num": anchor["page_num"],
            "section": " > ".join(anchor.get("section_path") or []),
            "block_types": ["text"],
            "is_ocr": bool(anchor.get("is_ocr")),
            "heading_level": heading_level,
        },
    }


# ── 主流程 ────────────────────────────────────────────────────────────────────

STRATEGY = "semantic"   # 改成 "fixed" 或 "hierarchical" 体验不同策略

def build_chunk_id(stock_code: str, year: str, idx: int) -> str:
    return f"{stock_code}_{year}_{idx:05d}"


def process_file(parsed_path: Path, strategy: str = STRATEGY):
    with open(parsed_path, encoding="utf-8") as f:
        data = json.load(f)

    meta   = data.get("meta", {})
    blocks = data.get("blocks", [])

    stock_code = meta.get("stock_code", "unknown")
    year       = meta.get("year", "unknown")

    logger.info(f"分块 {parsed_path.name}  策略={strategy}  blocks={len(blocks)}")

    # 根据策略生成 chunks
    raw_chunks = []

    if strategy == "fixed":
        full_text = "\n\n".join(b["content"] for b in blocks)
        for text_chunk in chunk_fixed(full_text):
            raw_chunks.append({
                "content":  text_chunk,
                "metadata": {"block_types": ["text"], "is_ocr": False, "section": "", "page_num": -1}
            })

    elif strategy == "semantic":
        for chunk in chunk_semantic(blocks):
            raw_chunks.append(chunk)

    elif strategy == "hierarchical":
        for chunk in chunk_hierarchical(blocks):
            raw_chunks.append(chunk)

    elif strategy == "structured":
        children, parents = chunk_structured(blocks)
        raw_chunks = children
        # 父块单独存 sidecar：子块只带 parent_id，不再整份复制父内容
        # （策略 C 的存储冗余就是那么来的）
        for p in parents:
            p["stock_code"] = stock_code
            p["year"]       = year
            p["source_file"] = parsed_path.name
        ppath = CHUNKS_DIR / f"{parsed_path.stem}_structured_parents.json"
        with open(ppath, "w", encoding="utf-8") as f:
            json.dump(parents, f, ensure_ascii=False, indent=2)
        logger.info(f"  → 父块 {len(parents)} 个，已保存 {ppath.name}")

    else:
        raise ValueError(f"未知策略: {strategy}")

    # 补充公共元信息
    result = []
    for idx, chunk in enumerate(raw_chunks):
        chunk_id = build_chunk_id(stock_code, year, idx)
        chunk["chunk_id"]              = chunk_id
        chunk["metadata"]["stock_code"] = stock_code
        chunk["metadata"]["year"]       = year
        chunk["metadata"]["strategy"]   = strategy
        chunk["metadata"]["source_file"] = parsed_path.name
        result.append(chunk)

    # 保存
    out_path = CHUNKS_DIR / f"{parsed_path.stem}_{strategy}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    logger.info(f"  → {len(result)} 个 chunk，已保存 {out_path.name}")
    return result


def main():
    import argparse
    ap = argparse.ArgumentParser(description="年报分块")
    ap.add_argument("--strategy", default=None,
                    choices=["fixed", "semantic", "hierarchical", "structured"],
                    help="默认取模块级 STRATEGY（semantic）")
    args = ap.parse_args()
    strategy = args.strategy or STRATEGY

    parsed_files = sorted(PARSED_DIR.glob("*.json"))
    if not parsed_files:
        logger.error("没有找到解析结果，请先运行 parse_pdf.py")
        return

    all_chunks = []
    for path in parsed_files:
        chunks = process_file(path, strategy=strategy)
        all_chunks.extend(chunks)

    # 合并所有公司的 chunk 到一个文件，方便统一建索引
    combined_path = CHUNKS_DIR / f"all_{strategy}.json"
    with open(combined_path, "w", encoding="utf-8") as f:
        json.dump(all_chunks, f, ensure_ascii=False, indent=2)

    logger.info(f"\n合并完成：共 {len(all_chunks)} 个 chunk → {combined_path}")

    # structured 策略额外合并父块 sidecar
    if strategy == "structured":
        all_parents = []
        for p in sorted(CHUNKS_DIR.glob("*_structured_parents.json")):
            if p.name.startswith("all_"):
                continue
            all_parents.extend(json.load(open(p, encoding="utf-8")))
        pout = CHUNKS_DIR / "all_structured_parents.json"
        with open(pout, "w", encoding="utf-8") as f:
            json.dump(all_parents, f, ensure_ascii=False, indent=2)
        ids = {c["metadata"].get("parent_id") for c in all_chunks}
        orphan = [i for i in ids if i and not any(p["parent_id"] == i for p in all_parents)]
        logger.info(f"父块合计 {len(all_parents)} 个 → {pout.name}；"
                    f"孤儿 parent_id {len(orphan)} 个")
        if orphan:
            logger.error(f"  存在子块指向不存在的父块: {orphan[:5]}")

    # 简单统计
    avg_len = sum(len(c["content"]) for c in all_chunks) / max(len(all_chunks), 1)
    logger.info(f"平均 chunk 长度: {avg_len:.0f} 字符")

    table_count = sum(1 for c in all_chunks if "table" in c["metadata"].get("block_types", []))
    ocr_count   = sum(1 for c in all_chunks if c["metadata"].get("is_ocr"))
    logger.info(f"其中表格块: {table_count}  OCR块: {ocr_count}")


if __name__ == "__main__":
    main()
