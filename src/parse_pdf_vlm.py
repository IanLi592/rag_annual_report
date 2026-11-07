"""Qwen-VL 版式解析分支。

定位：只处理**规则解析容易坏掉的页**，不是替换整个 PDF 流水线。
  - 表格密集页（≥2 张表，或表格面积占比 ≥20%）
  - 扫描页（文字极少但有图片）
  - 显式指定的页（--vlm-pages）
  - 抽样验证（--vlm-sample N，每份 PDF 取前 N 张复杂页）

设计约束：
  - 默认关闭（VLM_MODE=off），避免一跑 parse_pdf.py 就产生不可控的 token 费用。
  - 有硬上限 VLM_MAX_PAGES（默认 60/份），超过后回退规则解析并只告警一次。
  - 结果按页缓存到 data/vlm_cache/，重跑不重复计费。
  - 失败一律回退规则解析，不阻断整份年报。

输出与 parse_pdf.ParsedBlock 对齐，由 parse_pdf 负责包装成 dataclass。
"""
from __future__ import annotations

import base64
import json
import logging
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import pymupdf

logger = logging.getLogger(__name__)

DASHSCOPE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
VLM_MODEL = os.getenv("VLM_MODEL", "qwen-vl-max")
VLM_DPI = int(os.getenv("VLM_DPI", "150"))
VLM_MAX_PAGES = int(os.getenv("VLM_MAX_PAGES", "60"))
VLM_CACHE_DIR = Path(os.getenv("VLM_CACHE_DIR", str(Path(__file__).parent.parent / "data" / "vlm_cache")))

CHAPTER_RE = re.compile(r"^第[一二三四五六七八九十百]+章")
SECTION_RE = re.compile(r"^第[一二三四五六七八九十百]+节")
ITEM_RE = re.compile(r"^[一二三四五六七八九十]+、")
NUMBERED_RE = re.compile(r"^\d+\.\s")


# ── 配置与页类型判断 ─────────────────────────────────────────────────────────


@dataclass
class VlmConfig:
    """VLM 分支的开关与预算。"""

    mode: str = field(default_factory=lambda: os.getenv("VLM_MODE", "off").strip().lower())
    max_pages: int = field(default_factory=lambda: VLM_MAX_PAGES)
    pages: Optional[set[int]] = None          # 1-based 页码；显式指定时优先于 mode
    sample: int = 0                           # 每份 PDF 最多取 N 张复杂页
    dpi: int = field(default_factory=lambda: VLM_DPI)
    cache_dir: Path = field(default_factory=lambda: VLM_CACHE_DIR)

    def enabled(self) -> bool:
        return bool(self.pages) or self.mode != "off"

    def __post_init__(self):
        if self.mode not in ("off", "complex", "all"):
            raise ValueError(f"VLM_MODE 必须是 off/complex/all，收到 {self.mode!r}")
        if self.max_pages < 0:
            raise ValueError("VLM_MAX_PAGES 不能为负")


def is_complex_page(plumb_page, fitz_page, text: str) -> bool:
    """判断一页是否值得交给 VLM。

    规则解析真正会坏的是两类：表格密集页、扫描页。普通正文页继续走规则解析，
    既省钱又保持现有块结构。
    """
    if len((text or "").strip()) < 100:
        try:
            if fitz_page.get_images(full=True):
                return True
        except Exception:
            pass

    try:
        tables = plumb_page.find_tables()
    except Exception:
        return False
    if not tables:
        return False

    page_area = max(float(plumb_page.width) * float(plumb_page.height), 1.0)
    table_area = 0.0
    for t in tables:
        try:
            x0, y0, x1, y1 = t.bbox
            table_area += max(0.0, x1 - x0) * max(0.0, y1 - y0)
        except Exception:
            continue
    return len(tables) >= 2 or (table_area / page_area) >= 0.20


# ── 客户端与渲染 ─────────────────────────────────────────────────────────────


def get_vlm_client():
    """懒创建 DashScope OpenAI 兼容客户端；没有 key 就返回 None，由调用方回退。"""
    api_key = os.getenv("DASHSCOPE_API_KEY")
    if not api_key:
        return None
    from openai import OpenAI

    return OpenAI(api_key=api_key, base_url=DASHSCOPE_URL)


def render_page_png(pdf_path: Path, page_num: int, dpi: int = VLM_DPI) -> bytes:
    """把 1-based 页码渲染成 PNG。"""
    doc = pymupdf.open(str(pdf_path))
    try:
        page = doc[page_num - 1]
        mat = pymupdf.Matrix(dpi / 72, dpi / 72)
        pix = page.get_pixmap(matrix=mat)
        return pix.tobytes("png")
    finally:
        doc.close()


# ── Prompt 与响应解析 ────────────────────────────────────────────────────────

VLM_PROMPT = """你是中国上市公司年报的版式解析器。请只解析这一页，输出 JSON 数组，不要输出解释。
数组元素格式：
{"block_type":"title|text|table","content":"...","heading_level":1|2|3|0,"table":{"header":["..."],"rows":[["..."]]}}

规则：
1. 标题按章节层级给 heading_level：第X章=1，第X节=2，一、/1.=3，其他=0。
2. 表格必须保留行列结构，放在 table.header / table.rows；content 写一句简短摘要。
3. 不要合并跨栏，不要把表格拍平成一段文字。
4. 无法识别的页返回 []。
只输出 JSON 数组。"""


def _strip_code_fence(text: str) -> str:
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.S)
    text = re.sub(r"\s*```$", "", text, flags=re.S)
    return text.strip()


def _extract_json_array(raw) -> list:
    """从模型输出里取出 JSON 数组；兼容 content 为分段列表的情况。"""
    if isinstance(raw, list):
        raw = "".join(
            part.get("text", "") if isinstance(part, dict) else str(part)
            for part in raw
        )
    text = _strip_code_fence(str(raw or ""))
    if not text:
        raise ValueError("VLM 返回空内容")
    start, end = text.find("["), text.rfind("]")
    if start < 0 or end < start:
        raise ValueError(f"VLM 输出里没有 JSON 数组: {text[:120]!r}")
    data = json.loads(text[start:end + 1])
    if isinstance(data, dict):
        data = data.get("blocks", [])
    if not isinstance(data, list):
        raise ValueError("VLM JSON 顶层不是数组")
    return data


def _normalize_table(raw) -> list[list[str]]:
    """把 VLM 的 {header, rows} 或二维数组统一成二维数组。"""
    if raw is None:
        return []
    if isinstance(raw, dict):
        header = raw.get("header") or raw.get("columns") or []
        rows = raw.get("rows") or raw.get("data") or []
        table = [list(header)] + [list(r) for r in rows]
    elif isinstance(raw, list):
        table = [list(r) if isinstance(r, (list, tuple)) else [str(r)] for r in raw]
    else:
        return []
    cleaned = []
    for row in table:
        cleaned.append(["" if c is None else str(c).strip() for c in row])
    return [r for r in cleaned if any(r)]


def _table_to_markdown(table: list[list[str]]) -> str:
    if not table:
        return ""
    width = max(len(r) for r in table)
    rows = [r + [""] * (width - len(r)) for r in table]
    header = rows[0]
    lines = ["| " + " | ".join(header) + " |",
             "| " + " | ".join(["---"] * width) + " |"]
    for row in rows[1:]:
        lines.append("| " + " | ".join(row[:width]) + " |")
    return "\n".join(lines)


def _classify_heading(title: str, level: int = 0) -> int:
    if level in (1, 2, 3):
        return level
    t = title.strip()
    if CHAPTER_RE.match(t):
        return 1
    if SECTION_RE.match(t):
        return 2
    if ITEM_RE.match(t) or NUMBERED_RE.match(t):
        return 3
    return 0


def parse_vlm_response(raw, page_num: int, section_path: list[str]) -> list[dict]:
    """把模型返回的 JSON 规范化成 ParsedBlock 的 dict 形态。

    这是纯函数，单测不碰网络。任何不合法元素都跳过，而不是让整页失败。
    """
    blocks: list[dict] = []
    for item in _extract_json_array(raw):
        if not isinstance(item, dict):
            continue
        btype = str(item.get("block_type") or item.get("type") or "text").lower()
        if btype not in ("title", "text", "table"):
            btype = "text"
        content = str(item.get("content") or "").strip()
        heading_level = _classify_heading(content, int(item.get("heading_level") or 0))

        raw_table = None
        if btype == "table":
            raw_table = _normalize_table(item.get("table") or item.get("raw_table"))
            if raw_table:
                content = content or _table_to_markdown(raw_table)
            if not content and not raw_table:
                continue
        elif not content:
            continue

        blocks.append({
            "block_type": btype,
            "content": content,
            "page_num": page_num,
            "section_path": list(section_path),
            "is_ocr": False,
            "heading_level": heading_level,
            "parser": "vlm",
            "raw_table": raw_table,
        })
    return blocks


def parse_page_vlm(
    pdf_path: Path,
    page_num: int,
    section_path: list[str],
    client=None,
    dpi: int = VLM_DPI,
) -> list[dict]:
    """调用 Qwen-VL 解析一页（1-based 页码）。异常由调用方回退。"""
    client = client or get_vlm_client()
    if client is None:
        raise RuntimeError("DASHSCOPE_API_KEY 未设置，无法调用 Qwen-VL")

    png = render_page_png(pdf_path, page_num, dpi=dpi)
    b64 = base64.b64encode(png).decode("ascii")
    resp = client.chat.completions.create(
        model=VLM_MODEL,
        messages=[{
            "role": "user",
            "content": [
                {"type": "text", "text": VLM_PROMPT},
                {"type": "image_url",
                 "image_url": {"url": f"data:image/png;base64,{b64}"}},
            ],
        }],
        temperature=0,
    )
    raw = resp.choices[0].message.content
    blocks = parse_vlm_response(raw, page_num, section_path)
    if not blocks:
        raise ValueError(f"VLM 第 {page_num} 页没有解析出任何块")
    return blocks
