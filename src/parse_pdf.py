"""
PDF 解析脚本：将原始年报 PDF 转换为结构化文本

企业级 RAG 的真实挑战：
  1. 数字 PDF vs 扫描件：处理方式完全不同
  2. 表格提取：年报里大量财务报表，直接按文字流提取会乱序
  3. 页眉/页脚噪声：每页都有公司名、页码，必须去除
  4. 章节识别：利用字体大小/加粗猜测标题层级
  5. 输出格式：保留元信息（页码、章节路径），供后续溯源用

依赖安装：
  pip install pdfplumber pymupdf pytesseract pillow
  # tesseract-ocr 需要单独安装并配置 PATH
  # Windows: https://github.com/UB-Mannheim/tesseract/wiki
"""

import re
import json
import argparse
import logging
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Optional

import pdfplumber          # 擅长表格提取
import pymupdf             # PyMuPDF，擅长文字+图片提取。
                          # 不用旧别名 fitz —— 它会往 stdout 打
                          # "warning: The `fitz` API is deprecated..."，
                          # 破坏"从子进程 stdout 取结构化输出"的写法（CI 已踩过）

# OCR 依赖可选（需要同时安装 pytesseract 包 + tesseract-ocr 二进制）
try:
    from PIL import Image
    import pytesseract
    OCR_AVAILABLE = True
except ImportError:
    OCR_AVAILABLE = False

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# VLM 分支：默认关闭，只有显式 --vlm-mode / VLM_MODE 才启用。
# 页类型判断、prompt、响应解析都在 parse_pdf_vlm，这里只负责路由与回退。
from parse_pdf_vlm import VlmConfig, get_vlm_client, is_complex_page, parse_page_vlm, cache_namespace

RAW_DIR    = Path(__file__).parent.parent / "data" / "raw_pdf"
PARSED_DIR = Path(__file__).parent.parent / "data" / "parsed"
PARSED_DIR.mkdir(parents=True, exist_ok=True)

# 如果 tesseract 不在 PATH，手动指定（Windows 常见路径）
# pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"


# ── 数据结构 ──────────────────────────────────────────────────────────────────

@dataclass
class ParsedBlock:
    """
    一个解析块 = 年报里的一段连续内容（文字段落 or 表格）

    保留 page_num 和 section_path 非常重要——
    RAG 答案引用时能告诉用户"来自第38页，财务报告/资产负债表"
    """
    block_type:   str            # "text" | "table" | "title"
    content:      str            # 文字内容（表格转为 markdown）
    page_num:     int
    section_path: list[str]      # ["第三章 管理层讨论", "一、经营情况概述"]
    is_ocr:       bool = False   # 是否经过 OCR，质量可能有误
    heading_level: int = 0       # 0=非标题；1/2/3=H1/H2/H3
    parser:       str = "rule"   # rule | vlm | ocr
    raw_table:    Optional[list] = field(default=None, repr=False)  # 原始表格数据


# ── 工具函数 ──────────────────────────────────────────────────────────────────

# 年报里常见的章节标题模式（粗略匹配，不求完美）
CHAPTER_PATTERNS = [
    re.compile(r"^第[一二三四五六七八九十百]+[章节]"),     # 第一章、第三节
    re.compile(r"^[一二三四五六七八九十]、"),               # 一、二、
    re.compile(r"^\d+\.\s"),                                # 1. 2.
]

# 页眉/页脚：公司名 [+ 年份] + 年度报告。尾缀只放开“全文”一种 —— 修复前产物里
# 这类行实测 890 条、全部是「全文」尾缀（不带尾缀的另有 1,452 行，旧正则本来就滤）；
# 语料中另有“…年度报告正文/原件/摘要/。”这类真实句子，放宽尾缀集合等于删正文。
# 同一条正则同时用于正文过滤（NOISE_PATTERNS）与章节栈过滤
# （is_non_section_title），避免两处再次出现“一边漏、一边也漏”的漂移。
_REPORT_FOOTER_RE = re.compile(r"^.{1,40}年度报告(?:\s*全文)?\s*$")

NOISE_PATTERNS = [
    _REPORT_FOOTER_RE,                # 页眉/页脚：公司名+年度报告（含“全文”尾缀）
    re.compile(r"^\d+\s*$"),          # 独立页码
    re.compile(r"^—\s*\d+\s*—$"),    # — 38 —
    re.compile(r"^\d+\s*/\s*\d+$"),   # 62 / 143（页脚页码，避免混进 section_path）
]


def is_noise_line(line: str) -> bool:
    line = line.strip()
    if len(line) < 2:
        return True
    return any(p.match(line) for p in NOISE_PATTERNS)


def is_title_line(line: str, fontsize: Optional[float] = None, is_bold: bool = False) -> bool:
    """
    判断一行是否是标题。
    有字体信息时用字体大小，没有时用文字规律。
    """
    if fontsize and fontsize >= 14:
        return True
    if is_bold and len(line.strip()) < 50:
        return True
    return any(p.match(line.strip()) for p in CHAPTER_PATTERNS)


def classify_heading(title: str, fontsize: Optional[float] = None, is_bold: bool = False) -> int:
    """把标题分成 H1/H2/H3（返回 1/2/3；非标题返回 0）。

    优先用年报的编号规律，字体大小只作 fallback —— 很多年报的章标题字号
    并不比正文大多少，但“第X章/第X节/一、”是稳定信号。
    """
    t = title.strip()
    if re.match(r"^第[一二三四五六七八九十百]+章", t):
        return 1
    if re.match(r"^第[一二三四五六七八九十百]+节", t):
        return 2
    if re.match(r"^[一二三四五六七八九十]+、", t):
        return 3
    if re.match(r"^\d+\.\s", t):
        return 3
    if fontsize and fontsize >= 18:
        return 1
    if fontsize and fontsize >= 14:
        return 2
    if is_bold:
        return 3
    return 0


# ── 章节栈的噪声过滤 ─────────────────────────────────────────────────────────
# 这些“标题”会被字体/加粗规则误判，且一旦进入 section_path 就会长期留在栈顶：
#   · 目录页的“目录” / “第一节 ……… 3”（点线）
#   · 封面公司名、落款日期
# 实测未过滤时（2026-11-07 旧产物），主索引 63.5% 的 chunk 章节根是“目录/公司名/日期”，
# structured 的 parents.json 有 57% 同样被污染。修复 + 11-10/11-16 三轮过滤后，
# 2026-11-23 在当前盘上产物重测：主索引与 structured 的章节根污染率均为 0%（见 §3.3 策略 D），
# 63.5%/57% 仅作为修复前对照保留。
_TOC_TITLE_RE    = re.compile(r"^目\s*录$")
_TOC_LEADER_RE   = re.compile(r"\.{6,}")
# 落款日期。数字与“年月日”之间允许空格：五粮液/海康的 PDF 里实测是
# `2022 年4 月29 日` `2021 年12 月31 日` 这种带空格写法（修复前产物实测 33 块
# 因此进过栈，涉及 4 个不同日期串）。
# 整行锚定不能放宽 —— 语料里 `于2021年12月31日`、`截至2021年12月31日止12个月`
# 这类正文片段共 10 块，锚一丢就会整句被误杀。
_DATE_TITLE_RE   = re.compile(
    r"^[\d〇○零一二三四五六七八九十]{2,4}\s*年"
    r"\s*[\d〇○零一二三四五六七八九十]{1,2}\s*月"
    r"\s*[\d〇○零一二三四五六七八九十]{1,2}\s*日$")
_COMPANY_TITLE_RE = re.compile(r"^.{2,30}(?:股份有限公司|有限责任公司|有限公司)$")
_CHAPTER_RE      = re.compile(r"^第[一二三四五六七八九十百]+章")
_SECTION_RE      = re.compile(r"^第[一二三四五六七八九十百]+节")


def is_non_section_title(title: str) -> bool:
    """标题出现在版面上，但不应该进入章节路径（目录/封面/落款/页脚）。"""
    t = title.strip()
    return bool(
        _TOC_TITLE_RE.fullmatch(t)
        or _DATE_TITLE_RE.fullmatch(t)
        or _COMPANY_TITLE_RE.fullmatch(t)
        or _REPORT_FOOTER_RE.match(t)
        or _TOC_LEADER_RE.search(t)
    )


def is_chapter_title(title: str) -> bool:
    return bool(_CHAPTER_RE.match(title.strip()))


def is_section_title(title: str) -> bool:
    return bool(_SECTION_RE.match(title.strip()))


def is_toc_page(lines: list[str]) -> bool:
    """目录页：至少两行带点线。命中后整页标题都不推进章节栈。"""
    return sum(1 for line in lines if _TOC_LEADER_RE.search(line)) >= 2



def table_to_markdown(table: list[list]) -> str:
    """把 pdfplumber 提取的表格转成 markdown 格式，方便 LLM 理解。"""
    if not table:
        return ""

    # 清洗单元格：None 变空字符串，去掉换行
    rows = []
    for row in table:
        cleaned = [str(cell or "").replace("\n", " ").strip() for cell in row]
        rows.append(cleaned)

    if not rows:
        return ""

    # 构建 markdown 表格
    header = rows[0]
    lines  = ["| " + " | ".join(header) + " |"]
    lines.append("| " + " | ".join(["---"] * len(header)) + " |")
    for row in rows[1:]:
        # 对齐列数（有些 PDF 表格行列不整齐）
        while len(row) < len(header):
            row.append("")
        lines.append("| " + " | ".join(row[:len(header)]) + " |")

    return "\n".join(lines)


def detect_if_scanned(page: pymupdf.Page, text: str) -> bool:
    """
    启发式判断：文字极少但图片多 → 很可能是扫描页。
    年报中扫描件多见于附件（审计报告原件）。
    """
    if len(text.strip()) > 50:
        return False
    image_list = page.get_images(full=True)
    return len(image_list) > 0


def ocr_page(page: pymupdf.Page, dpi: int = 200) -> str:
    """对扫描页做 OCR（中文）。需要 pytesseract + tesseract-ocr 二进制。"""
    if not OCR_AVAILABLE:
        return "[扫描页，OCR 不可用（未安装 pytesseract/tesseract），内容跳过]"
    try:
        mat  = pymupdf.Matrix(dpi / 72, dpi / 72)
        clip = page.rect
        pix  = page.get_pixmap(matrix=mat, clip=clip)
        img  = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        text = pytesseract.image_to_string(img, lang="chi_sim+eng")
        return text
    except Exception as e:
        logger.warning(f"  OCR 失败，跳过此页: {e}")
        return "[扫描页，OCR 失败，内容跳过]"


# ── 主解析逻辑 ────────────────────────────────────────────────────────────────

class AnnualReportParser:
    """
    年报 PDF 解析器。

    策略：
      - 用 pdfplumber 提取表格（它的表格算法更准）
      - 用 PyMuPDF 提取带字体信息的文字（用于判断标题）
      - 对扫描页降级为 OCR
    """

    def __init__(self, pdf_path: Path, meta: dict = None, vlm_config: VlmConfig = None):
        self.pdf_path = pdf_path
        self.meta     = meta or {}
        self.blocks: list[ParsedBlock] = []
        self._section_stack: list[str] = []
        self._section_levels: list[int] = []
        # VLM 默认关闭；只有显式 --vlm-mode / VLM_MODE 才启用
        self.vlm_config = vlm_config or VlmConfig()
        self._vlm_pages_used = 0
        self._vlm_sample_seen = 0
        self._vlm_budget_warned = False

    def _section_level(self, title: str, level: int = 0) -> int:
        """把标题落到有效的 H1/H2/H3；目录/封面噪声返回 0。

        年报有两种编号习惯：有的用“第X章”做一级、第X节做二级；有的直接
        用“第X节”做一级。后者如果按 H2 处理，会把栈顶的“目录/公司名”
        一直当成父级。这里显式区分：栈顶是第X章时第X节才是 H2，否则是 H1。
        """
        if is_non_section_title(title):
            return 0
        if is_section_title(title):
            if self._section_stack and is_chapter_title(self._section_stack[0]):
                return 2
            return 1
        return level or classify_heading(title)

    def _update_section(self, title: str, level: int = 0):
        """维护章节栈：根据编号层级推断层次，并显式记录 H1/H2/H3。"""
        level = self._section_level(title, level)
        if level <= 0:
            return
        if level == 1 or not self._section_stack:
            self._section_stack = [title]                              # H1
            self._section_levels = [1]
        elif level == 2:
            self._section_stack = self._section_stack[:1] + [title]     # H2
            self._section_levels = self._section_levels[:1] + [2]
        elif level == 3:
            # H3 的截断点 = 「栈里最后一个真实 H2 之后」，而不是硬切 [:2]。
            # 硬切隐含假设栈永远是 [H1, H2, H3] 三层，但这份语料普遍是「第X节」
            # 直接当 H1、「一、xxx」就是最低层（修复前全库仅 3 个 chunk 带 H2 层）。
            # 于是每来一个新 H3，上一个同级标题会被当成父级永久留在栈中间：
            #   第二节 公司简介 > 一、公司信息 > 四、注册变更情况   ← 中间层是过期同级
            # 修复前 parsed 实测 2,957 块的路径里同时挂着两个不同的「一、」样式。
            cut = 1                                          # 没有真实 H2 就只留 H1
            for i, lv in enumerate(self._section_levels):
                if lv == 2:
                    cut = i + 1                              # 切在最后一个真实 H2 之后
            self._section_stack = self._section_stack[:cut] + [title]      # H3
            self._section_levels = self._section_levels[:cut] + [3]
        else:
            # 没有编号规律的标题保留旧行为：最多保留三级 + 当前标题
            self._section_stack = self._section_stack[:3] + [title]
            self._section_levels = self._section_levels[:3] + [3]


    def _should_use_vlm(self, page_num: int, raw_text: str, plumb_page, fitz_page) -> bool:
        cfg = self.vlm_config
        if cfg.pages is not None:
            return (page_num + 1) in cfg.pages
        if cfg.mode == "off":
            return False
        candidate = True if cfg.mode == "all" else is_complex_page(
            plumb_page, fitz_page, raw_text)
        if not candidate:
            return False
        if cfg.sample:
            if self._vlm_sample_seen >= cfg.sample:
                return False
            self._vlm_sample_seen += 1
        return True

    def _try_vlm(self, page_num: int, raw_text: str, plumb_page, fitz_page):
        """尝试 VLM 解析一页；任何失败都返回 None，让调用方回退规则解析。"""
        if not self._should_use_vlm(page_num, raw_text, plumb_page, fitz_page):
            return None

        cfg = self.vlm_config
        if cfg.max_pages and self._vlm_pages_used >= cfg.max_pages:
            if not self._vlm_budget_warned:
                logger.warning(
                    f"  VLM 已达到预算上限 {cfg.max_pages} 页，后续页回退规则解析"
                    f"（调大 VLM_MAX_PAGES 或 --vlm-max-pages 可继续）")
                self._vlm_budget_warned = True
            return None

        cache_path = (cfg.cache_dir / cache_namespace()
                      / self.pdf_path.stem / f"page_{page_num + 1:04d}.json")
        if cache_path.exists():
            try:
                data = json.loads(cache_path.read_text(encoding="utf-8"))
                logger.debug(f"  第{page_num+1}页命中 VLM 缓存")
                return [ParsedBlock(**b) for b in data]
            except Exception as e:
                logger.warning(f"  第{page_num+1}页 VLM 缓存损坏，重新解析: {e}")

        client = get_vlm_client()
        if client is None:
            if not self._vlm_budget_warned:
                logger.warning("  未设置 DASHSCOPE_API_KEY，VLM 分支回退规则解析")
                self._vlm_budget_warned = True
            return None

        try:
            data = parse_page_vlm(
                self.pdf_path, page_num + 1, list(self._section_stack),
                client, dpi=cfg.dpi,
            )
        except Exception as e:
            logger.warning(
                f"  第{page_num+1}页 VLM 解析失败，回退规则解析: "
                f"{type(e).__name__}: {e}")
            return None

        self._vlm_pages_used += 1
        try:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            cache_path.write_text(
                json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        except Exception as e:
            logger.warning(f"  VLM 缓存写入失败（不影响本次解析）: {e}")

        logger.info(f"  第{page_num+1}页使用 Qwen-VL 解析，{len(data)} 个块")
        return [ParsedBlock(**b) for b in data]

    def _parse_page_rule(self, page_num: int, fitz_page, plumb_page, raw_text: str = None):
        """规则解析一页（原有逻辑，抽出来供 VLM 失败时回退）。"""
        raw_text = raw_text if raw_text is not None else fitz_page.get_text("text")
        is_scanned = detect_if_scanned(fitz_page, raw_text)

        if is_scanned:
            logger.debug(f"  第{page_num+1}页：检测到扫描件，启动 OCR")
            ocr_text = ocr_page(fitz_page)
            self.blocks.append(ParsedBlock(
                block_type="text",
                content=ocr_text,
                page_num=page_num + 1,
                section_path=list(self._section_stack),
                is_ocr=True,
                parser="ocr",
            ))
            return

        # ── 2. 提取表格（用 pdfplumber）──
        # 表格只「排队」，不立刻入库。旧版在这里直接 append，于是同一页内所有表格
        # 永远比所有标题先入库，盖的是「进入本页之前」的章节栈。五粮液 2021 第 5 页
        # 实测（pdfplumber/PyMuPDF 原始坐标）：
        #   y= 97.67 标题 第二节 公司简介和主要财务指标
        #   y=142.83 标题 一、公司信息
        #   y=162.04 表格 股票简称 | 五粮液 | 股票代码 | 000858
        #   y=450.32 标题 二、联系人和联系方式
        #   y=469.65 表格 董事会秘书 | 证券事务代表   …… 四张表四张标题交替排列
        # 旧逻辑把这四张表全标成上一页结束时的「第一节 重要提示和释义 > 释 义」。
        # 修复前 parsed 全库 4,491 个表格块里 3,059 个（68.1%）同页存在排在其后的
        # 标题，即暴露于该缺陷。改法：一页内只有一条阅读流，谁在上面谁先入库，
        # 表格入库时才用「此刻」的章节栈盖章（见 _flush_tables_above）。
        # 注：table_obj.extract() 与 page.extract_tables() 等价 —— pdfplumber 0.11.10
        # 里后者就是 [t.extract(**text_settings) for t in find_tables()]，抽样 5 页
        # 逐字比对相等；这里合成一次遍历，顺便保证行数据与 bbox 天然对齐。
        table_bboxes   = []
        pending_tables = []            # [[top_y, markdown, rows]]，被下方文字追平时入库
        for table_obj in plumb_page.find_tables():
            table_bboxes.append(table_obj.bbox)      # 后续跳过这些区域的文字提取
            rows = table_obj.extract()
            md   = table_to_markdown(rows)
            if md:
                pending_tables.append([table_obj.bbox[1], md, rows])
        pending_tables.sort(key=lambda item: item[0])

        def _flush_tables_above(y: float) -> None:
            """把起点在 y 之上的表格按 y 顺序入库，用当前章节栈盖章。"""
            if not pending_tables:
                return
            ready    = [t for t in pending_tables if t[0] < y]
            if not ready:
                return
            pending_tables[:] = [t for t in pending_tables if t[0] >= y]
            for _top_y, md, rows in ready:
                self.blocks.append(ParsedBlock(
                    block_type="table",
                    content=md,
                    page_num=page_num + 1,
                    section_path=list(self._section_stack),
                    raw_table=rows,
                ))

        # ── 3. 提取文字（跳过表格区域，逐行处理）──
        # 注意：table_bboxes 必须真的用上。旧版收集了它却从未在下面的循环里判断，
        # 导致表格文字既进 table 块、又流进 text 块 —— 实测摘要页 77% 的文字行
        # 落在表格 bbox 内，全量约 4,491 个块是这种重复（README 记为已知问题）。
        page_dict = fitz_page.get_text("dict", flags=pymupdf.TEXT_PRESERVE_WHITESPACE)
        current_para_lines = []
        page_lines = [
            "".join(span["text"] for span in line.get("spans", [])).strip()
            for block in page_dict.get("blocks", [])
            if block.get("type") == 0
            for line in block.get("lines", [])
        ]
        in_toc_page = is_toc_page(page_lines)

        def _in_table(bbox) -> bool:
            if not bbox or not table_bboxes:
                return False
            return any(bbox[0] < b[2] and bbox[2] > b[0] and
                       bbox[1] < b[3] and bbox[3] > b[1] for b in table_bboxes)

        for block in page_dict.get("blocks", []):
            if block.get("type") != 0:   # 0=文字，1=图片
                continue

            for line in block.get("lines", []):
                line_text = "".join(
                    span["text"] for span in line.get("spans", [])
                ).strip()

                if not line_text or is_noise_line(line_text):
                    continue

                if _in_table(line.get("bbox")):
                    continue   # 表格内容只保留 table 块那一份

                # 阅读顺序：这条线已经在某些表格下方，那些表格先入库（在此刻的章节
                # 栈下盖章，所以位于表格之上的标题已经生效、位于其下的还没有）。
                # 必须在 _update_section 之前调用，否则表格会被它下方的标题污染。
                # bbox 缺失时按页面顶部处理（y=0 → 谁都不冲刷，退化成旧行为）。
                _flush_tables_above((line.get("bbox") or [0, 0, 0, 0])[1])

                # 判断是否标题
                spans    = line.get("spans", [])
                fontsize = spans[0].get("size", 0) if spans else 0
                is_bold  = any("Bold" in span.get("font", "") for span in spans)

                if is_title_line(line_text, fontsize, is_bold):
                    # 先把积累的段落存起来
                    if current_para_lines:
                        self.blocks.append(ParsedBlock(
                            block_type="text",
                            content="\n".join(current_para_lines),
                            page_num=page_num + 1,
                            section_path=list(self._section_stack),
                        ))
                        current_para_lines = []

                    if in_toc_page:
                        # 目录页的条目不是真实章节：保留 title 块供分块切边界，
                        # 但不推进章节栈，heading_level 记 0。
                        level = 0
                    else:
                        level = self._section_level(
                            line_text, classify_heading(line_text, fontsize, is_bold))
                        self._update_section(line_text, level)
                    self.blocks.append(ParsedBlock(
                        block_type="title",
                        content=line_text,
                        page_num=page_num + 1,
                        section_path=list(self._section_stack),
                        heading_level=level,
                    ))
                else:
                    current_para_lines.append(line_text)

        # 最后一段
        if current_para_lines:
            self.blocks.append(ParsedBlock(
                block_type="text",
                content="\n".join(current_para_lines),
                page_num=page_num + 1,
                section_path=list(self._section_stack),
            ))

        # 页尾收尾：还压在队列里的表格都排在本页最后一条可见文字之下，用本页最终
        # 章节栈盖章。少了这一步，纯表格页（表格一直排到页底）会整张丢掉表格。
        # 已知残留：段落块要等段落结束才入库，所以「起点落在段落中间」的表格仍会
        # 排在该段落块之前 —— 与修复前方向一致（修复前是全部表格都在全部文字之前），
        # 彻底纠正要把段落按表格边界切开，涉及块正文本身，不在本次修复范围。
        _flush_tables_above(float("inf"))

    def parse(self) -> list[ParsedBlock]:
        logger.info(f"开始解析: {self.pdf_path.name}")

        # 同时打开两个解析器
        plumber_doc = pdfplumber.open(self.pdf_path)
        fitz_doc    = pymupdf.open(str(self.pdf_path))
        self._vlm_pages_used = 0
        self._vlm_sample_seen = 0

        try:
            for page_num in range(len(fitz_doc)):
                fitz_page   = fitz_doc[page_num]
                plumb_page  = plumber_doc.pages[page_num]
                raw_text    = fitz_page.get_text("text")

                vlm_blocks = self._try_vlm(page_num, raw_text, plumb_page, fitz_page)
                if vlm_blocks is not None:
                    # VLM 页同样要推进章节栈：混合模式下，下一页的规则解析
                    # 必须知道这一页新出现的标题，否则 section_path 会停在旧章节。
                    for b in vlm_blocks:
                        if b.block_type == "title":
                            # VLM 没有字体/加粗信息，不能信它给的无编号标题 H1：
                            # 实测 qwen3-vl-plus 会把“合并利润表”标成 heading_level=1，
                            # 一旦当 H1 就会把“第十节 财务报告”顶掉。这里只认编号规律，
                            # 无编号标题（报表名等）不准推进章节栈。
                            b.heading_level = self._section_level(
                                b.content, classify_heading(b.content))
                            self._update_section(b.content, b.heading_level)
                        b.section_path = list(self._section_stack)
                    self.blocks.extend(vlm_blocks)
                    continue

                self._parse_page_rule(page_num, fitz_page, plumb_page, raw_text)
        finally:
            plumber_doc.close()
            fitz_doc.close()

        logger.info(f"  解析完成: {len(self.blocks)} 个块")
        return self.blocks

    def save(self):
        """将解析结果保存为 JSON，保留所有元信息。"""
        stem     = self.pdf_path.stem
        out_path = PARSED_DIR / f"{stem}.json"

        output = {
            "meta":   self.meta,
            "source": str(self.pdf_path),
            "blocks": [asdict(b) for b in self.blocks],
        }
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(output, f, ensure_ascii=False, indent=2)
        logger.info(f"  已保存 → {out_path}")


# ── 主流程 ────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="PDF → 结构化 blocks（可选 Qwen-VL 复杂页解析）")
    ap.add_argument("--vlm-mode", choices=["off", "complex", "all"], default=None,
                    help="VLM 路由：off（默认）/complex（表格密集页+扫描页）/all（全部页）")
    ap.add_argument("--vlm-max-pages", type=int, default=None,
                    help="每份 PDF 的 VLM 页数上限，默认取 VLM_MAX_PAGES（60）")
    ap.add_argument("--vlm-pages", type=str, default=None,
                    help="显式指定页码（1-based，逗号分隔），优先于 --vlm-mode")
    ap.add_argument("--vlm-sample", type=int, default=0,
                    help="每份 PDF 只取前 N 张复杂页做 VLM 抽样，0=不限制")
    ap.add_argument("--vlm-dpi", type=int, default=None, help="VLM 渲染 DPI，默认 150")
    args = ap.parse_args()

    vlm = VlmConfig()
    if args.vlm_mode is not None:
        vlm.mode = args.vlm_mode
    if args.vlm_max_pages is not None:
        vlm.max_pages = args.vlm_max_pages
    if args.vlm_pages:
        vlm.pages = {int(x) for x in args.vlm_pages.split(",") if x.strip()}
    if args.vlm_sample:
        vlm.sample = args.vlm_sample
    if args.vlm_dpi is not None:
        vlm.dpi = args.vlm_dpi
    if vlm.enabled():
        logger.info(
            f"VLM 已启用：mode={vlm.mode} max_pages={vlm.max_pages} "
            f"sample={vlm.sample} pages={sorted(vlm.pages) if vlm.pages else 'auto'}")

    manifest_path = RAW_DIR.parent / "manifest.json"

    if manifest_path.exists():
        with open(manifest_path, encoding="utf-8") as f:
            manifest = json.load(f)
    else:
        # 没有 manifest 就直接扫目录
        manifest = [
            {"filename": p.name, "stock_code": "", "year": ""}
            for p in RAW_DIR.glob("*.pdf")
        ]

    if not manifest:
        logger.error("没有找到任何 PDF，请先运行 download_reports.py")
        return

    for item in manifest:
        pdf_path = RAW_DIR / item["filename"]
        if not pdf_path.exists():
            logger.warning(f"文件不存在，跳过: {pdf_path}")
            continue

        parser = AnnualReportParser(pdf_path, meta=item, vlm_config=vlm)
        parser.parse()
        parser.save()

    logger.info(f"\n全部解析完成，结果在 {PARSED_DIR}")


if __name__ == "__main__":
    main()
