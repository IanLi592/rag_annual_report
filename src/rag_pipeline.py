"""
RAG 问答流水线（原生实现）

流程：
  (可选) 查询改写
        ↓
  向量检索（DashScope embedding + FAISS）
        +
  BM25 关键词检索（jieba + rank_bm25）
        ↓
  RRF 融合排名
        ↓
  (可选) CrossEncoder Rerank
        ↓
  相关性阈值过滤（过低则拒绝回答）
        ↓
  LLM 生成（DashScope qwen-plus）+ 引用标注

使用方式：
  python rag_pipeline.py                            # 交互式
  python rag_pipeline.py --query "茅台2023年营收"
  python rag_pipeline.py --query "..." --stock 600519 --year 2023
  python rag_pipeline.py --query "..." --query-rewrite   # 开启查询改写
  python rag_pipeline.py --query "..." --no-bm25 --no-rerank  # 消融测试

依赖：
  pip install faiss-cpu rank_bm25 jieba openai numpy
  export DASHSCOPE_API_KEY="sk-xxx"
"""

import os
import re
import json
import logging
import argparse
import numpy as np
from pathlib import Path
from typing import Optional
from openai import OpenAI

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

BASE_DIR        = Path(__file__).parent.parent
# 默认主索引；RAG_INDEX_DIR 可指向 vectorstore/structured 等 A/B 索引
VECTORSTORE_DIR = Path(os.getenv("RAG_INDEX_DIR") or (BASE_DIR / "vectorstore"))
INDEX_PATH      = VECTORSTORE_DIR / "faiss_index.bin"
META_PATH       = VECTORSTORE_DIR / "faiss_meta.json"
# structured 分块的父块内容单独存放；子块只带 parent_id，避免整份复制
PARENTS_PATH    = VECTORSTORE_DIR / "parents.json"

DASHSCOPE_URL   = "https://dashscope.aliyuncs.com/compatible-mode/v1"
EMBED_MODEL     = "text-embedding-v3"
EMBED_DIM       = 1024
LLM_MODEL       = "qwen-plus"        # 可换 qwen-turbo（更快）/ qwen-max（更强）

TOP_K_RETRIEVE  = 10    # 初始召回数
TOP_K_RERANK    = 4     # Rerank 后保留数（给 LLM）
SCORE_THRESHOLD = 0.25  # 最高得分低于此值时触发拒绝回答（归一化余弦相似度）

SYSTEM_PROMPT = """你是一个专业的财务分析助手，专门回答关于中国上市公司年度报告的问题。

回答规则：
1. 只根据【参考资料】中的内容回答，不得引用或编造资料外的数据
2. 若参考资料不足以支撑回答，直接说"根据提供的资料无法回答此问题"
3. 引用具体数据时，在句末标注来源编号，如：营业收入为1476亿元[1]
4. 数字要精确，不得四舍五入或模糊表达
5. 回答简洁，重点突出，避免无关废话"""


# ── DashScope 客户端 ──────────────────────────────────────────────────────────

def get_client() -> OpenAI:
    api_key = os.getenv("DASHSCOPE_API_KEY")
    if not api_key:
        raise EnvironmentError("请设置环境变量 DASHSCOPE_API_KEY")
    return OpenAI(api_key=api_key, base_url=DASHSCOPE_URL)


# ── 向量检索 ──────────────────────────────────────────────────────────────────

def match_filter(item: dict, filter_meta: Optional[dict]) -> bool:
    """
    元数据过滤谓词。向量侧与 BM25 侧共用这一份，避免两条召回链路各持一套口径。

    语义（与原向量侧逐字一致）：
      - filter_meta 为空/None → 全通过
      - 每个键都必须存在且字符串化后相等（缺键当 ""，因此按 stock_code 过滤会把
        没有该字段的块排除掉）
    """
    if not filter_meta:
        return True
    return all(str(item.get(k, "")) == str(v) for k, v in filter_meta.items())


def company_name_map(meta_list: list[dict]) -> dict[str, str]:
    """stock_code → 公司简称。

    来源是索引 meta 的 `source_file`（`600519_2023_贵州茅台_...json`）：
    不额外维护一份公司表，换数据集时不会与现实脱节。
    """
    mapping: dict[str, str] = {}
    for item in meta_list:
        code = str(item.get("stock_code", ""))
        if not code:
            continue
        parts = Path(str(item.get("source_file", ""))).stem.split("_")
        if len(parts) >= 3 and parts[2]:
            mapping.setdefault(code, parts[2])
    return mapping


_CODE_RE = re.compile(r"(?<!\d)(\d{6})(?!\d)")


def expand_query_entities(question: str, meta_list: list[dict]) -> str:
    """把问题里的股票代码补成「公司简称+代码」，只扩检索 query，不改原问题。

    实测改造前的 36 道数值题全写成「000858 在 2021 年的营业收入是多少」，而 chunk
    正文只有「五粮液」没有股票代码：Top-4 经常是别的公司。补上简称后
    semantic Hit@4 0.500→1.000、structured 0.306→0.972（rerank top-4）。
    题集已改为自然公司名，此函数保留给线上用户输入代码时兜底。
    问题里已经出现公司简称时不重复插入，未知代码原样保留。
    """
    names = company_name_map(meta_list)
    if not names:
        return question

    def repl(m: re.Match) -> str:
        code = m.group(1)
        name = names.get(code)
        if not name or name in question:
            return m.group(0)
        return f"{name}{code}"

    return _CODE_RE.sub(repl, question)


def infer_filter_from_question(question: str, meta_list: list[dict]) -> dict:
    """从问题文本里推断 stock_code / year，用于表格精确取数的兜底。

    只做**高置信**推断：问题里出现完整股票代码、年报里的公司全称，或唯一的
    四位年份。推断不出来的维度不填，绝不猜。
    """
    filt: dict[str, str] = {}

    codes = {
        str(item.get("stock_code", ""))
        for item in meta_list
        if item.get("stock_code") and str(item["stock_code"]) in question
    }
    if len(codes) == 1:
        filt["stock_code"] = codes.pop()
    elif not codes:
        # 公司名取自 source_file 的第三段（600519_2023_贵州茅台_...）
        name_to_code = {n: c for c, n in company_name_map(meta_list).items()}
        matched = {code for name, code in name_to_code.items() if name in question}
        if len(matched) == 1:
            filt["stock_code"] = matched.pop()
        # 问题里同时提到多家公司时不锁公司 —— 锁哪家都可能答错

    years = sorted(set(re.findall(r"(?<!\d)(20\d{2})(?!\d)", question)))
    if len(years) == 1:
        filt["year"] = years[0]
    return filt


class VectorStore:
    def __init__(self, client: OpenAI, index_dir: str | Path | None = None):
        import faiss
        self.client = client
        self.index_dir = Path(index_dir) if index_dir else VECTORSTORE_DIR
        index_path = (self.index_dir / "faiss_index.bin") if index_dir else INDEX_PATH
        meta_path = (self.index_dir / "faiss_meta.json") if index_dir else META_PATH
        self.index     = faiss.read_index(str(index_path))
        with open(meta_path, encoding="utf-8") as f:
            self.meta_list = json.load(f)
        logger.info(
            f"FAISS 索引加载完成，共 {self.index.ntotal} 条向量（{self.index_dir}）")

    def _embed_query(self, query: str) -> np.ndarray:
        resp = self.client.embeddings.create(
            model=EMBED_MODEL, input=[query], dimensions=EMBED_DIM
        )
        vec = np.array([resp.data[0].embedding], dtype="float32")
        vec = vec / np.maximum(np.linalg.norm(vec, axis=1, keepdims=True), 1e-9)
        return vec

    def search(
        self,
        query: str,
        top_k: int = TOP_K_RETRIEVE,
        filter_meta: Optional[dict] = None,
    ) -> list[dict]:
        """
        向量检索，可选元数据过滤（按 stock_code / year）。
        filter_meta 示例：{"stock_code": "600519", "year": "2023"}
        """
        query_vec = self._embed_query(query)
        # 多取一些再过滤
        scores, indices = self.index.search(query_vec, top_k * 4)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0 or idx >= len(self.meta_list):
                continue
            item = dict(self.meta_list[idx])
            item["vec_score"] = float(score)

            if not match_filter(item, filter_meta):
                continue

            results.append(item)
            if len(results) >= top_k:
                break
        return results


# ── BM25 关键词检索 ───────────────────────────────────────────────────────────

class BM25Store:
    """
    基于 jieba + rank_bm25 的关键词检索。
    对精确数字、专有名词（如"净利润""研发费用"）效果优于纯向量检索。
    首次初始化会分词整个语料库，约需数秒。
    """
    def __init__(self, meta_path: str | Path | None = None):
        from rank_bm25 import BM25Okapi
        import jieba
        from jieba_domain import load_domain_dict

        # 必须在任何分词之前注入，否则前缀词典已建好、后加的词不生效
        dict_stat = load_domain_dict(jieba)

        with open(Path(meta_path) if meta_path else META_PATH, encoding="utf-8") as f:
            self.meta_list = json.load(f)

        logger.info(
            f"构建 BM25 索引（分词中，请稍候）... "
            f"领域词表={'已加载 ' + str(dict_stat['loaded']) + ' 词' if dict_stat['loaded'] else ('被 JIEBA_DOMAIN_DICT=0 关闭' if not dict_stat['enabled'] else '未启用')}")
        tokenized   = [list(jieba.cut(item["content"])) for item in self.meta_list]
        self.bm25   = BM25Okapi(tokenized)
        self.jieba  = jieba
        logger.info("BM25 索引完成")

    def search(
        self,
        query: str,
        top_k: int = TOP_K_RETRIEVE,
        filter_meta: Optional[dict] = None,
    ) -> list[dict]:
        """
        BM25 关键词检索，支持与向量侧完全相同的元数据过滤。

        排序必须**确定性**：语料里存在大量内容完全相同的块（表格文字曾同时进
        table/text 两处），BM25 会对它们给出完全相同的分数。np.argsort 默认
        quicksort 不稳定，同分块的先后顺序随实现漂移 —— 实测同一份数据在
        "argsort 版"与"sorted 版"聚合下 Hit@4 差出 9 个百分点（0.38 vs 0.288），
        所有 A/B 结论都因此不可信。这里改成 (-score, 原始下标) 的稳定排序。
        """
        tokens = list(self.jieba.cut(query))
        scores = self.bm25.get_scores(tokens)

        if filter_meta:
            keep = np.array(
                [match_filter(item, filter_meta) for item in self.meta_list], dtype=bool
            )
            scores = np.where(keep, scores, 0.0)

        order = np.lexsort((np.arange(len(scores)), -scores))   # 先按 -score，同分按下标
        top_idx = order[:top_k]

        results = []
        for idx in top_idx:
            if scores[idx] < 1e-9:  # 完全不相关（含被过滤掉）的跳过
                continue
            item = dict(self.meta_list[idx])
            item["bm25_score"] = float(scores[idx])
            results.append(item)
        return results


# ── RRF 融合 ──────────────────────────────────────────────────────────────────

def reciprocal_rank_fusion(
    vec_results: list[dict],
    bm25_results: list[dict],
    k: int = 60,
) -> list[dict]:
    """
    Reciprocal Rank Fusion（RRF）。
    公式：score(d) = Σ 1/(k + rank_i(d))，k=60 为经验值。
    将向量召回和 BM25 召回的排名合并，互补各自的盲区。
    """
    rrf_scores: dict[str, float] = {}
    chunk_map:  dict[str, dict]  = {}

    for rank, item in enumerate(vec_results, 1):
        cid = item["chunk_id"]
        rrf_scores[cid] = rrf_scores.get(cid, 0) + 1 / (k + rank)
        chunk_map[cid]  = item

    for rank, item in enumerate(bm25_results, 1):
        cid = item["chunk_id"]
        rrf_scores[cid] = rrf_scores.get(cid, 0) + 1 / (k + rank)
        chunk_map[cid]  = item

    sorted_cids = sorted(rrf_scores, key=lambda x: -rrf_scores[x])
    results = []
    for cid in sorted_cids:
        item = dict(chunk_map[cid])
        item["rrf_score"] = rrf_scores[cid]
        results.append(item)
    return results


# ── CrossEncoder Rerank（可选）────────────────────────────────────────────────

_RERANKER = None          # 模块级单例
_RERANKER_FAILED = False


def get_reranker():
    """懒加载 CrossEncoder 单例。

    为什么必须缓存：旧实现每次 rerank() 都 CrossEncoder(model_name) —— 实测
    模型加载 5.8 s，等于每个请求白付 5.8 秒。这大概就是 serve.py 把 rerank
    关成 False 的真实原因。
    """
    global _RERANKER, _RERANKER_FAILED
    if _RERANKER is not None or _RERANKER_FAILED:
        return _RERANKER
    try:
        from sentence_transformers import CrossEncoder
        model_path = Path(__file__).parent.parent / "models" / "bge-reranker-base"
        model_name = str(model_path) if model_path.exists() else "BAAI/bge-reranker-base"
        _RERANKER = CrossEncoder(model_name)
        logger.info(f"Reranker 已加载并缓存：{model_name}")
    except Exception as e:
        _RERANKER_FAILED = True
        logger.warning(f"Reranker 加载失败，后续请求将跳过精排: {type(e).__name__}: {e}")
        _RERANKER = None
    return _RERANKER


def rerank(
    query: str,
    candidates: list[dict],
    top_k: int = TOP_K_RERANK,
    min_score: float | None = None,
) -> list[dict]:
    """
    用 CrossEncoder 对候选集二次精排，并可选地**过滤长尾低分文档**。

    模型：BAAI/bge-reranker-base（中文，约 1.1 GB）；不可用时降级为直接截断。

    min_score 是本次新增的真实过滤能力：分数低于该阈值的文档不进上下文。
    注意语义 —— 旧实现只排序+截断，**永远返回 top_k 条**，所以"重排削减上下文
    Token"这个说法在旧代码下不成立。阈值取值必须由分数分布决定，
    不能为了凑某个削减比例反推（见 scripts/ci/measure_rerank.py）。
    """
    reranker = get_reranker()
    if reranker is None:
        return candidates[:top_k]

    try:
        pairs  = [(query, c["content"]) for c in candidates]
        scores = reranker.predict(pairs)
        for item, score in zip(candidates, scores):
            item["rerank_score"] = float(score)
        ranked = sorted(candidates, key=lambda x: -x.get("rerank_score", 0))
        if min_score is not None:
            kept = [c for c in ranked if c["rerank_score"] >= min_score]
            dropped = len(ranked) - len(kept)
            if dropped:
                logger.info(f"  Rerank 过滤掉 {dropped} 条低于 {min_score} 的长尾文档")
            return kept[:top_k]
        return ranked[:top_k]
    except Exception as e:
        logger.warning(f"Rerank 失败，使用 RRF 原始排序: {e}")

    return candidates[:top_k]


# ── 查询改写 ──────────────────────────────────────────────────────────────────

def rewrite_query(query: str, client: OpenAI) -> str:
    """
    用 LLM 将模糊的用户问题改写为更适合检索的精确表述。
    示例：
      原始：茅台最近怎么样
      改写：贵州茅台2023年营业收入净利润同比增长率经营情况

    使用 qwen-turbo（最快最便宜），不需要高质量推理。
    """
    resp = client.chat.completions.create(
        model="qwen-turbo",
        messages=[
            {
                "role": "system",
                "content": (
                    "你是检索查询优化专家。将用户的问题改写为更适合从年度报告中检索信息的精确查询语句。"
                    "保留关键实体（公司名、年份、财务指标），扩展相关关键词，不要超过50字。"
                    "直接输出改写后的查询语句，不要解释。"
                ),
            },
            {"role": "user", "content": query},
        ],
        temperature=0,
    )
    rewritten = resp.choices[0].message.content.strip()
    logger.info(f"查询改写: {query!r} → {rewritten!r}")
    return rewritten


# ── LLM 生成 ──────────────────────────────────────────────────────────────────

def resolve_chunk_content(item: dict, parents: dict | None = None) -> str:
    """取一条检索结果最终送进 LLM 的正文。

    优先级：父块表（structured sidecar）→ 内联 parent_content（旧 hierarchical）
    → 子块自身。`build_context` 与 `/query/debug` 必须共用这一份口径。
    """
    parents = parents or {}
    return (parents.get(item.get("parent_id"))
            or item.get("parent_content")
            or item.get("content", ""))


_CELL_WS = re.compile(r"\s+")
_LABEL_PREFIX = re.compile(r"^(?:(?:其中|加|减)[:：]?|[一二三四五六七八九十]+、)")
_LABEL_UNIT = re.compile(r"[（(](?:元|万元|亿元|%|％)[）)]$")
_CJK_RE = re.compile(r"[一-鿿]")
# 解析残留：单元格首列偶尔带"的/在/和"这类介词，去掉后再当指标名用。
_LABEL_FILLER = re.compile(r"^(?:的|在|和|与|及|为|是)+")
# 增减/比例/附注列不是指标绝对值；补位只回答"是多少"，这些列会干扰取值。
_SKIP_COL = re.compile(r"增减|变动|比例|附注|本年比上年")

# 同一指标在不同年份报表里的首列写法不同（2022 年报用"资产总计"而不是
# "总资产"）。补位只在这里做同义映射，且映射后仍要求首列精确相等。
_TABLE_LABEL_ALIASES = {
    "总资产": ("资产总计",),
    "资产总额": ("资产总计",),
    "归属于上市公司股东的净资产": (
        "归属于母公司所有者权益合计",
        "归属于母公司股东权益",
    ),
}


def _norm_cell(value) -> str:
    """表格单元格归一化：去空白，便于“营业收入 / 其中：营业收入”这类匹配。"""
    return _CELL_WS.sub("", str(value if value is not None else ""))


def _norm_label(value) -> str:
    """行标签归一化：去掉“其中：/加：/减：”前缀与“（元）”这类单位后缀。"""
    return _LABEL_UNIT.sub("", _LABEL_PREFIX.sub("", _norm_cell(value)))


def _collect_table_rows(items: list[dict], key: str) -> list[dict]:
    """按归一化行标签扫描 `table_json`，只认首列。"""
    rows = []
    for item in items:
        for row in item.get("table_json") or []:
            if not row:
                continue
            # 只有首列有内容的多为跨列标题（如"（一）营业收入的确认"），不是数据行
            if not any(_norm_cell(c) for c in row[1:]):
                continue
            first = _norm_label(row[0])
            if not first:
                continue
            # 长文本单元格不是行标签（实测有整段审计说明被当成“营业收入”行）；
            # 允许“一、营业收入”“营业收入（元）”这类少量前后缀，但不允许整段文字。
            if len(first) > len(key) + 8:
                continue
            if key == first or key in first or first in key:
                rows.append({
                    "chunk_id":   item.get("chunk_id", ""),
                    "stock_code": item.get("stock_code", ""),
                    "year":       item.get("year", ""),
                    "page_num":   item.get("page_num", -1),
                    "section":    item.get("section", ""),
                    "row":        [("" if c is None else str(c)) for c in row],
                })
    return rows


def lookup_table_value(
    retrieved: list[dict],
    row_label: str,
    fallback_pool: Optional[list[dict]] = None,
) -> list[dict]:
    """从检索结果的 `table_json` 里按行标签精确提取整行。

    这是“表格结构化为 JSON 入库”的消费端：向量检索负责找到相关表格，
    这里直接读原始二维结构，避免让 LLM 从 Markdown 文本里重新数格子。
    只在**首列**匹配，防止某个数值恰好等于标签造成误命中。

    `fallback_pool`：检索候选里没有命中时的兜底池（按 stock_code/year 收窄的
    全部表格块）。报表页的 chunk 以大段数字为主，向量相似度经常输给审计意见
    之类的叙述页 —— 实测“贵州茅台2023年利润表营业收入”的 Top-10 里没有
    合并利润表，但按行标签直接在 600519/2023 的表格块里扫就能精确命中。
    """
    key = _norm_label(row_label)
    if not key:
        return []
    rows = _collect_table_rows(retrieved, key)
    if rows or not fallback_pool:
        return rows

    seen = set()
    for r in _collect_table_rows(fallback_pool, key):
        sig = (r["chunk_id"], tuple(r["row"]))
        if sig not in seen:
            seen.add(sig)
            rows.append(r)
    return rows


def build_context(retrieved: list[dict], parents: dict | None = None) -> tuple[str, list[dict]]:
    """将检索结果组装为 Prompt 上下文，返回上下文字符串和引用列表。

    parents：{parent_id: 父块全文}。structured 分块下子块只带 parent_id，
    父块内容在这里按 id 查表拼装（"子块召回、父块拼装"）。
    查不到就退回 parent_content（旧 hierarchical 的内联存法），再退回子块自身。
    """
    parents = parents or {}
    parts     = []
    citations = []

    for i, item in enumerate(retrieved, 1):
        stock   = item.get("stock_code", "")
        year    = item.get("year", "")
        page    = item.get("page_num", "")
        section = item.get("section", "")

        label = f"[{i}] {stock} {year}年报"
        if section:
            label += f" · {section}"
        if page and page != -1:
            label += f" · 第{page}页"

        content = resolve_chunk_content(item, parents)
        parts.append(f"{label}\n{content}")
        citations.append({"index": i, "source": label, "chunk_id": item.get("chunk_id", "")})

    return "\n\n---\n\n".join(parts), citations


def call_llm(query: str, context: str, client: OpenAI) -> str:
    user_msg = (
        f"【参考资料】\n{context}\n\n"
        f"【问题】\n{query}\n\n"
        "请根据参考资料回答，并在引用数据处标注来源编号（如[1]）。"
    )
    resp = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": user_msg},
        ],
        temperature=0.1,
    )
    return resp.choices[0].message.content


# ── 完整流水线 ────────────────────────────────────────────────────────────────

class RAGPipeline:
    def __init__(
        self,
        use_bm25:        bool = True,
        use_rerank:      bool = True,
        use_query_rewrite: bool = False,
        rerank_min_score: float | None = None,
        index_dir:       str | Path | None = None,
    ):
        self.client        = get_client()
        self.index_dir     = Path(index_dir) if index_dir else VECTORSTORE_DIR
        self.vec_store     = VectorStore(
            self.client, index_dir=self.index_dir if index_dir else None)
        self.use_bm25      = use_bm25
        self.use_rerank    = use_rerank
        self.use_qr        = use_query_rewrite
        # 长尾过滤阈值。默认 None = 只排序+截断，不丢上下文。
        # 0.7 曾在 25 题子集上看似"质量不掉"，但 100 题实测 Hit@4 0.560→0.410，
        # 已不再是默认值；要启用必须先用 scripts/ci/measure_rerank.py 重定。
        self.rerank_min_score = rerank_min_score
        self.bm25_store    = (
            BM25Store(meta_path=self.index_dir / "faiss_meta.json")
            if use_bm25 and index_dir else
            (BM25Store() if use_bm25 else None)
        )
        # 父块查表：文件不存在（semantic 索引）时为空 dict，行为与旧版完全一致
        parents_path = self.index_dir / "parents.json" if index_dir else PARENTS_PATH
        self.parents: dict = {}
        if parents_path.exists():
            with open(parents_path, encoding="utf-8") as f:
                self.parents = {p["parent_id"]: p["content"] for p in json.load(f)}
            logger.info(f"已载入父块 {len(self.parents)} 个（子块召回、父块拼装）")
        self.strategy = (
            self.vec_store.meta_list[0].get("strategy", "")
            if self.vec_store.meta_list else ""
        )

    def table_pool(
        self,
        filter_meta: Optional[dict] = None,
        question: str = "",
    ) -> list[dict]:
        """表格精确取数的兜底池：收窄到 stock_code/year 后的全部表格块。

        必须有 stock_code 才返回 —— 15 份年报都有“营业收入”，只按年份或全库
        扫会跨公司串数，宁可返回空也不返回错公司的数。
        """
        filt = dict(filter_meta or {})
        for k, v in infer_filter_from_question(
                question, self.vec_store.meta_list).items():
            filt.setdefault(k, v)
        if not filt.get("stock_code"):
            return []
        return [item for item in self.vec_store.meta_list
                if item.get("table_json") and match_filter(item, filt)]

    def retrieve(
        self,
        question: str,
        filter_meta: Optional[dict] = None,
        verbose: bool = False,
    ) -> dict:
        """执行一次完整检索（改写 → 向量/BM25 → RRF → 精排）。

        唯一检索入口：`query()` 与 `serve.query_debug()` 都用它，保证调试接口
        不会因为复制了一份逻辑而和主链路越走越远。
        """
        # ① 查询实体归一 + 可选改写
        #    股票代码只扩检索/重排 query，送 LLM 的仍是原问题（见 query()）
        expanded_question = expand_query_entities(question, self.vec_store.meta_list)
        retrieval_query = (
            rewrite_query(expanded_question, self.client) if self.use_qr
            else expanded_question
        )

        # ② 向量检索
        vec_results = self.vec_store.search(retrieval_query, TOP_K_RETRIEVE, filter_meta)
        if verbose:
            logger.info(f"向量召回: {len(vec_results)} 条，最高分={vec_results[0]['vec_score']:.3f}" if vec_results else "向量召回: 0 条")

        # ③ BM25 + RRF 融合
        bm25_results = []
        if self.use_bm25 and self.bm25_store:
            bm25_results = self.bm25_store.search(retrieval_query, TOP_K_RETRIEVE, filter_meta)
            candidates   = reciprocal_rank_fusion(vec_results, bm25_results)
            if verbose:
                logger.info(f"BM25 召回: {len(bm25_results)} 条，RRF 后: {len(candidates)} 条")
        else:
            candidates = vec_results

        # ④ Rerank（与线上同阈值，默认不过滤）
        #    用实体归一后的问题：代码题若只给“000858”，重排同样需要公司名
        if self.use_rerank:
            final = rerank(expanded_question, candidates, TOP_K_RERANK, self.rerank_min_score)
        else:
            final = candidates[:TOP_K_RERANK]

        if verbose:
            logger.info(f"最终使用 {len(final)} 条上下文")

        return {
            "query":        retrieval_query,
            "vec_results":  vec_results,
            "bm25_results": bm25_results,
            "candidates":   candidates,
            "final":        final,
        }

    def _table_context_items(
        self,
        question: str,
        filter_meta: Optional[dict] = None,
    ) -> list[dict]:
        """把问题里点名的指标行，按 `table_json` 精确取出来补进上下文。

        为什么需要：自然语言问题“五粮液2021年的总资产是多少”在 RRF top-4 里
        经常命中同一份年报的叙述段（行业情况、审计报告），而摘要表块因为数字
        密集排不进前四 —— 实测 10 道可答题里 5 道因此误拒。表格行直接按首列
        标签取数，不依赖排序，能补上这个召回盲区。

        只在公司/年份能高置信推断时启用；最多补 1 行，避免上下文膨胀。
        """
        pool = self.table_pool(filter_meta, question)
        if not pool:
            return []
        # 只处理单年点查：跨年趋势题需要多年份上下文，补一行本年数据
        # 反而会挤占“逐年对比”的证据（实测 RAGAS Context Precision 下滑）。
        filt = dict(filter_meta or {})
        for k, v in infer_filter_from_question(
                question, self.vec_store.meta_list).items():
            filt.setdefault(k, v)
        if not filt.get("year"):
            return []
        qn = _norm_cell(question)
        names_map = company_name_map(self.vec_store.meta_list)
        company = names_map.get(pool[0].get("stock_code", ""), "") if pool else ""
        labels: set[str] = set()
        aliases: set[str] = set()
        cell_labels: set[str] = set()
        for item in pool:
            for row in item.get("table_json") or []:
                if not row:
                    continue
                label = _LABEL_FILLER.sub("", _norm_label(row[0]))
                cell_labels.add(label)
                # 公司名碎片（"宁德"/"时代"）也会出现在问题里，不能当指标名
                if company and (label in company or company in label):
                    continue
                # 表格首列常混着 "2021年" "1" 这类序号/年份，它们在问题里
                # 也出现且往往比指标名更长，必须排除，否则会用年份去取行。
                if (len(label) >= 2 and not label[0].isdigit()
                        and len(_CJK_RE.findall(label)) >= 2
                        and label in qn):
                    labels.add(label)
        # 同义写法（"总资产" vs "资产总计"）与直接命中一起排序：
        # 先试更长的指标名；精确命中但整行为空（跨 chunk 断开的资产负债表）
        # 时继续试下一个，直到取到带数值的行。
        for canonical, names in _TABLE_LABEL_ALIASES.items():
            if canonical in qn:
                aliases.update(a for a in names if a in cell_labels)
        # 排序：完整指标名（≥3 字）优先，其次同义写法，最后才是"资产/总额"
        # 这类碎片。海康 2023 的"总资产"和收购标的的"资产总计"同时有值，
        # 别名不能压过直接命中的主要指标行。
        good = sorted((x for x in labels if len(x) >= 3), key=len, reverse=True)
        short = sorted((x for x in labels if len(x) < 3), key=len, reverse=True)
        candidates = good + sorted(aliases, key=len, reverse=True) + short
        if not candidates:
            return []

        rows: list[dict] = []
        for label in candidates:
            for name in (label, *_TABLE_LABEL_ALIASES.get(label, ())):
                found = [r for r in lookup_table_value(pool, name)
                         if _norm_label(r["row"][0]) == _norm_label(name)
                         and any(_norm_cell(v) for v in r["row"][1:])]
                if found:
                    rows = found[:1]
                    break
            if rows:
                break
        if not rows:
            return []

        by_chunk = {item.get("chunk_id"): item for item in pool}
        out = []
        for r in rows:
            src = by_chunk.get(r["chunk_id"], {})
            table = src.get("table_json") or []
            header = table[0] if table else []
            # header[0] 通常是"项目/年份"占位，真正的列名从下标 1 开始；
            # 行标签必须显式写出来，否则 LLM 只看到数值不知道是哪项指标。
            # 有些页表格跨 chunk 断开（00191 的首行是"无形资产"），表头
            # 本身是数据行 —— 这时列名不可信，只报本年第一个数值。
            usable = any(h and re.search(r"[一-鿿A-Za-z]", _norm_cell(h))
                         for h in header[1:])
            if usable:
                # 报表常见合并单元格：数值在第 3/5 列，列名在 4/6 列，
                # 直接 zip 会把数值和"附注"错配、甚至全部落空。
                hdr = [_norm_cell(h) for h in header]
                pairs = []
                for j, v in enumerate(r["row"][1:], 1):
                    v = _norm_cell(v)
                    if not v:
                        continue
                    k = next((i for i in range(j, len(hdr)) if hdr[i]), None)
                    if k is None:
                        k = next((i for i in range(j - 1, -1, -1) if hdr[i]), None)
                    col = hdr[k] if k is not None else f"第{j}列"
                    if _SKIP_COL.search(col):
                        continue
                    pairs.append(f"{col}: {v}")
            else:
                vals = [_norm_cell(v) for v in r["row"][1:] if _norm_cell(v)]
                pairs = [f"{r['year']}年: {vals[0]}"] if vals else []
            if not pairs:
                continue
            content = "【表格精确取数】"
            if r["stock_code"] or r["year"]:
                content += f"{r['stock_code']} {r['year']}年报 · "
            content += (r["section"] or "") + "\n"
            content += f"{r['row'][0]}：{'；'.join(pairs)}"
            out.append({
                "chunk_id": r["chunk_id"],
                "stock_code": r["stock_code"],
                "year": r["year"],
                "page_num": r["page_num"],
                "section": r["section"],
                "content": content,
            })
        return out

    def query(
        self,
        question: str,
        filter_meta: Optional[dict] = None,
        verbose: bool = False,
    ) -> dict:
        r = self.retrieve(question, filter_meta, verbose)
        final = r["final"]

        # ⑤ 相关性阈值检查
        if not final:
            return {
                "answer": "未找到相关内容，无法回答此问题。",
                "citations": [], "retrieved": [],
            }
        # 阈值始终用 vec_score（余弦相似度，0~1 可解释）；rrf_score 量纲不同不适合做阈值
        top_score = final[0].get("vec_score", final[0].get("rerank_score", 1.0))
        if top_score < SCORE_THRESHOLD and filter_meta is None:
            return {
                "answer": "根据年报知识库未能找到与该问题相关的内容，建议直接查阅原始年报。",
                "citations": [], "retrieved": final,
            }

        # ⑤.5 表格精确行补位：追加而不是替换。替换会丢掉 RRF 已召回的证据
        # （多跳题需要多年份上下文），补一行只多约 200 字符。
        table_items = self._table_context_items(question, filter_meta)
        if table_items:
            final = final + table_items

        # ⑥ LLM 生成
        context, citations = build_context(final, self.parents)
        answer = call_llm(question, context, self.client)

        return {"answer": answer, "citations": citations, "retrieved": final}


# ── 入口 ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="年报 RAG 问答（原生版）")
    parser.add_argument("--query",         type=str,  default=None)
    parser.add_argument("--stock",         type=str,  default=None, help="股票代码，如 600519")
    parser.add_argument("--year",          type=str,  default=None, help="年份，如 2023")
    parser.add_argument("--query-rewrite", action="store_true", help="开启查询改写（增加一次 LLM 调用）")
    parser.add_argument("--no-bm25",       action="store_true", help="关闭 BM25（消融实验用）")
    parser.add_argument("--no-rerank",     action="store_true", help="关闭 Rerank（消融实验用）")
    parser.add_argument("--rerank-min-score", type=float, default=None,
                        help="精排分数下限；默认不过滤。启用前先用 measure_rerank.py 重定")
    parser.add_argument("--index-dir",     type=str, default=None,
                        help="索引目录；默认 vectorstore/，可指向 vectorstore/structured")
    parser.add_argument("--table-row",     type=str, default=None,
                        help="不生成答案，按首列标签从检索候选的 table_json 精确取整行")
    args = parser.parse_args()

    pipeline = RAGPipeline(
        use_bm25         = not args.no_bm25,
        use_rerank       = not args.no_rerank,
        use_query_rewrite= args.query_rewrite,
        rerank_min_score = args.rerank_min_score,
        index_dir        = args.index_dir,
    )

    filter_meta = {}
    if args.stock: filter_meta["stock_code"] = args.stock
    if args.year:  filter_meta["year"]        = args.year
    if not filter_meta: filter_meta = None

    def print_result(q: str, result: dict):
        print(f"\n{'='*60}")
        print(f"问题：{q}")
        print(f"{'='*60}")
        print(f"\n{result['answer']}")
        if result["citations"]:
            print("\n── 来源 ──")
            for c in result["citations"]:
                print(f"  {c['source']}")

    if args.table_row:
        if not args.query:
            parser.error("--table-row 需要同时提供 --query（用于定位相关表格）")
        r = pipeline.retrieve(args.query, filter_meta=filter_meta, verbose=True)
        rows = lookup_table_value(
            r["candidates"], args.table_row,
            fallback_pool=pipeline.table_pool(filter_meta, args.query))
        print(f"\n按行标签 {args.table_row!r} 命中 {len(rows)} 行")
        for row in rows:
            label = f"{row['stock_code']} {row['year']}年报 · {row['section']} · 第{row['page_num']}页"
            print(f"  [{' | '.join(row['row'])}]  ← {label}")
        return

    if args.query:
        result = pipeline.query(args.query, filter_meta=filter_meta, verbose=True)
        print_result(args.query, result)
    else:
        print("年报 RAG 问答系统")
        print(f"模型：{LLM_MODEL}  |  向量库：{INDEX_PATH}")
        print("输入 'exit' 退出，'mode' 查看当前配置\n")
        while True:
            try:
                q = input("问题：").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if not q:
                continue
            if q.lower() == "exit":
                break
            if q.lower() == "mode":
                print(f"BM25={'on' if pipeline.use_bm25 else 'off'}  "
                      f"Rerank={'on' if pipeline.use_rerank else 'off'}  "
                      f"QueryRewrite={'on' if pipeline.use_qr else 'off'}")
                continue
            result = pipeline.query(q, filter_meta=filter_meta, verbose=True)
            print_result(q, result)


if __name__ == "__main__":
    main()
