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
# 表格计量单位的来源（解析产物，不在版本库里；缺失时单位标注自动降级为"未知"）
PARSED_DIR      = Path(os.getenv("RAG_PARSED_DIR") or (BASE_DIR / "data" / "parsed"))

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
5. 回答简洁，重点突出，避免无关废话
6. 只使用与问题所述公司、年份一致的资料；公司或年份不符的资料不得用来替代作答，宁可说明资料不足"""


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


def infer_companies_from_question(question: str, meta_list: list[dict]) -> list[str]:
    """问题里点名了哪些公司 → 升序 stock_code 列表（可能 0/1/多个）。

    `infer_filter_from_question` 在同时命中多家公司时**故意**不锁 stock_code
    （锁哪家都会把另一家滤没），那是单公司检索的正确行为；但跨公司比较题需要
    「每家都单独召回一次」，所以这里把命中的公司全部列出来给多公司分支用。
    匹配口径与 infer_filter 一致：完整代码出现在问题里，或 source_file 第三段的
    公司简称出现在问题里。推断不出来就是空列表，绝不猜。
    """
    found = {
        str(item.get("stock_code", ""))
        for item in meta_list
        if item.get("stock_code") and str(item["stock_code"]) in question
    }
    for code, name in company_name_map(meta_list).items():
        if name and name in question:
            found.add(code)
    found.discard("")
    return sorted(found)


def missing_companies(codes: list[str], items: list[dict]) -> list[str]:
    """覆盖校验：被点名的公司里，哪些在上下文中一条证据都没有。

    跨公司表格补位会把多家公司的数值合并进**一条** content，所以除了 stock_code
    字段还要看正文里是否出现该公司代码 —— 只看字段会把已覆盖的公司误判成缺失。
    """
    missing = []
    for code in codes:
        covered = any(
            str(item.get("stock_code", "")) == code
            or code in str(item.get("content", ""))
            for item in items
        )
        if not covered:
            missing.append(code)
    return missing


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

# 一个单元格"像金额"吗 —— 四条分支各钉一类真实写法：
#   千分位  `272,699,660,092.25`（元单位摘要表/合并报表）
#   小数    `3220.55`（万元单位表，常不带分组）
#   长整数  `8252000000`（无分组无小数的整数金额）
#   百分数  `91.60%`（毛利率类问题的答案本身）
# 刻意排除：`2022`（4 位整数 = 年份/序号）、`主要: 水污染物`（纯文字）。
_AMOUNT_THOUSANDS = re.compile(r"^-?\d+(?:[,，]\d{3})+(?:\.\d+)?$")
_AMOUNT_DECIMAL   = re.compile(r"^-?\d+\.\d+$")
_AMOUNT_BARE_INT  = re.compile(r"^-?\d{5,}$")
_AMOUNT_PERCENT   = re.compile(r"^-?\d+(?:\.\d+)?\s*(?:%|％)$")


def looks_like_amount(value) -> bool:
    """这格是不是金额/比率。补位取错行时的唯一兜底判据。

    实测失效模式（v1 Q11「比较贵州茅台和五粮液2022年的销售毛利率，哪家**公司**更高」）：
    两家的年报块里都没有"销售毛利率"这一行，旧代码退到 2 字碎片标签"公司"，
    把第五节环保表里「主要污染物及特征污染物」那行当成指标取了出来。
    """
    v = _norm_cell(value)
    return bool(_AMOUNT_THOUSANDS.match(v) or _AMOUNT_DECIMAL.match(v)
                or _AMOUNT_BARE_INT.match(v) or _AMOUNT_PERCENT.match(v))


# ── 表格计量单位（缺陷 2 修复）────────────────────────────────────────────────
# 为什么要回到 data/parsed：单位声明行**没有进索引**。实测 002415 2023 全部索引块
# 里只有 3 处单位声明（页 92/97/117），而取到研发费用行的附注表在第 164-165 页；
# 300750 2023 合并利润表（第 120 页）的「单位：万元」只存在于解析产物里。
# 于是同一条补位上下文里会并排出现 11,392,948,404.47（元）与 1,835,610.84（万元），
# 模型直接比大小必然得出方向相反的结论。
# data/parsed 不在版本库里（.gitignore 忽略 data/），所以这里是 best-effort：
# 解析不到就不标注，绝不猜单位。
_UNIT_DECL_RE = re.compile(r"单位[:：]\s*(百万元|亿元|万元|千元|元)")
# 行标签/表头自带的单位后缀，如「归属于上市公司股东的净利润（元）」「资产减值（万元）」
_UNIT_SUFFIX_RE = re.compile(r"[（(]\s*(百万元|亿元|万元|千元|元)\s*[)）]")
UNIT_UNKNOWN_MARK = "（单位未标注）"
# 页级声明只在这么宽的页窗内采信。实测 002415 2023 的「万元」声明最后一次出现在
# 第 136 页，而取到研发费用的附注表在第 164 页 —— 无限carry-over 会把（元）表
# 标成万元，标错比不标更糟。
UNIT_PAGE_WINDOW = 2


def unit_from_labels(*texts) -> str:
    """从行标签与表头里抓单位后缀 —— 这个信号就在索引里，不依赖解析产物。

    必须先归一化空白：PDF 断行会把「（万元）」拆成「（万\\n元）」，
    直接对原文匹配会把明明写着单位的行漏成"单位未标注"。
    """
    for t in texts:
        m = _UNIT_SUFFIX_RE.search(_norm_cell(t))
        if m:
            return m.group(1)
    return ""



def parse_unit_declarations(blocks: list[dict]) -> list[tuple[int, str]]:
    """按文档顺序抽出 [(页码, 单位)]，一页内有多个声明时取**最后一个**。

    年报排版是"单位：万元"紧贴在它下面的表格前，同页反复出现（每张子表各写一次），
    取最后一个才对应最靠近的那张表。
    """
    decls: list[tuple[int, str]] = []
    for block in blocks or []:
        found = _UNIT_DECL_RE.findall(str(block.get("content") or ""))
        if not found:
            continue
        try:
            page = int(block.get("page_num"))
        except (TypeError, ValueError):
            continue
        decls.append((page, found[-1]))
    return decls


def unit_at(decls: list[tuple[int, str]], page: int,
            window: int = UNIT_PAGE_WINDOW) -> str:
    """第 page 页的单位 = 窗口内（同页或往前 ≤ window 页）最后一条声明；否则未知。"""
    unit = ""
    for d_page, d_unit in decls:
        if d_page > page or page - d_page > window:
            continue
        unit = d_unit
    return unit


def load_unit_declarations(source_file: str, parsed_dir: Path) -> list[tuple[int, str]]:
    """读解析产物拿单位声明序列；文件缺失/损坏一律返回空列表（不抛异常）。"""
    if not source_file:
        return []
    path = Path(parsed_dir) / str(source_file)
    key = str(path)
    if key in _UNIT_CACHE:
        return _UNIT_CACHE[key]
    decls: list[tuple[int, str]] = []
    try:
        with open(path, encoding="utf-8") as f:
            decls = parse_unit_declarations(json.load(f).get("blocks") or [])
    except (OSError, ValueError):
        decls = []
    _UNIT_CACHE[key] = decls
    return decls


_UNIT_CACHE: dict[str, list[tuple[int, str]]] = {}

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


# ── 表格兜底取数的「行选择」规则 ──────────────────────────────────────────────
# 为什么需要这一档：`lookup_table_value` 会把同一个指标在年报里出现过的行**全部**
# 返回（摘要表、合并报表、母公司报表、附注/会计政策各写一遍）。旧实现直接取
# `found[:1]`，等于按块在语料里的先后顺序碰运气，多年度/多公司题的证据会被整段
# 丢掉。2026-11-17 离线诊断实测：multihop 34 题里，兜底通道**一条都没有**进到上下文
# （time_trend 被「无 year 就 return」的闸挡住；跨公司题则被 table_pool 的单公司
# 硬闸挡住），最终 33/34 题 answer_relevancy 恰好为 0。
# 现在的规则：同一 (公司, 年度) 内按表来源优先级取一行；不同年度各留一行。
#
# 摘要表与合并报表并为第一档，是刻意的：两者数值本应一致，而现状（numeric 36 题
# 里 32 题的答案数值就是这条通道送进去的）取到的正是「主要会计数据和财务指标」
# 摘要表；把摘要单独降档会无端改掉一批本来答对的题。
MAX_BACKFILL_ROWS = 3                  # 一次补位最多合并几个 (公司, 年度) 的行
_YEARS_RE = re.compile(r"(?<!\d)(20\d{2})(?!\d)")   # 与 infer_filter_from_question 同形

_CONSOLIDATED_RE = re.compile(r"合并(?:资产负债表|利润表|现金流量表|所有者权益变动表|资产负债表续)")
_PARENT_RE       = re.compile(r"母公司(?:资产负债表|利润表|现金流量表|所有者权益变动表)")
_SUMMARY_RE      = re.compile(r"主要会计数据|会计数据和财务指标")
_NOTES_RE        = re.compile(r"重要会计政策|会计估计|项目附注|财务报表附注|附注")


def table_source_rank(item: dict) -> int:
    """这张表来自哪一档：0 合并/摘要，1 母公司，2 附注与会计政策，3 其他。"""
    blob = _norm_cell(str(item.get("section") or "") +
                      str(item.get("content") or "")[:400])
    if _PARENT_RE.search(blob) and not _CONSOLIDATED_RE.search(blob):
        return 1
    if _CONSOLIDATED_RE.search(blob) or _SUMMARY_RE.search(blob):
        return 0
    if _NOTES_RE.search(blob):
        return 2
    return 3


def select_backfill_rows(found: list[dict], pool_by_chunk: dict[str, dict],
                         max_rows: int = MAX_BACKFILL_ROWS) -> list[dict]:
    """把同一指标的候选行按 (公司, 年度) 分组，组内取表来源优先级最高的一行。

    同分时保持**池子原顺序**（即年报里的出现顺序），这样单年题选到的行与修复前
    逐字一致 —— 改动的只是「多年度/多公司时不再只留一行」。
    """
    best: dict[tuple[str, str], tuple[int, int, dict]] = {}
    for pos, r in enumerate(found):
        key = (str(r.get("stock_code") or ""), str(r.get("year") or ""))
        rank = (table_source_rank(pool_by_chunk.get(r.get("chunk_id")) or {}), pos)
        cur = best.get(key)
        if cur is None or rank < cur[:2]:
            best[key] = (*rank, r)
    # 年度升序、公司代码升序：趋势题读起来就是 2021 → 2022 → 2023
    ordered = sorted(best.values(), key=lambda t: (t[2]["year"], t[2]["stock_code"]))
    return [t[2] for t in ordered[:max_rows]]


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
        # 单位声明的查找目录。只在这里赋值：用 object.__new__ 构造的离线守卫实例
        # 没有这个属性，`_row_unit` 会直接返回"未知"，守卫测试因此不依赖 data/parsed。
        self.unit_dir      = PARSED_DIR
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
        stock_codes: Optional[list[str]] = None,
    ) -> list[dict]:
        """表格精确取数的兜底池：收窄到 stock_code/year 后的全部表格块。

        必须有高置信的公司代码才返回 —— 15 份年报都有“营业收入”，只按年份或全库
        扫会跨公司串数，宁可返回空也不返回错公司的数。
        `stock_codes`：跨公司比较题显式给出**全部**被点名公司（来自
        `infer_companies_from_question`），每家都进池，比较题才可能两边都有证据。
        不传时沿用 filter_meta / 问题推断出的单公司口径，行为与修复前一致。
        """
        filt = dict(filter_meta or {})
        for k, v in infer_filter_from_question(
                question, self.vec_store.meta_list).items():
            filt.setdefault(k, v)
        codes = [str(c) for c in (stock_codes or [])]
        if not codes and filt.get("stock_code"):
            codes = [str(filt["stock_code"])]
        if not codes:
            return []
        code_set = set(codes)
        rest = {k: v for k, v in filt.items() if k != "stock_code"}
        return [item for item in self.vec_store.meta_list
                if item.get("table_json")
                and str(item.get("stock_code", "")) in code_set
                and match_filter(item, rest)]

    def _search_candidates(
        self,
        retrieval_query: str,
        filter_meta: Optional[dict],
    ) -> tuple[list[dict], list[dict], list[dict]]:
        """向量 + BM25 召回并 RRF 融合。单公司与多公司两条路共用，口径不分叉。"""
        vec_results = self.vec_store.search(retrieval_query, TOP_K_RETRIEVE, filter_meta)
        bm25_results = []
        if self.use_bm25 and self.bm25_store:
            bm25_results = self.bm25_store.search(retrieval_query, TOP_K_RETRIEVE, filter_meta)
            candidates   = reciprocal_rank_fusion(vec_results, bm25_results)
        else:
            candidates = vec_results
        return vec_results, bm25_results, candidates

    def _retrieve_multi_company(
        self,
        retrieval_query: str,
        expanded_question: str,
        companies: list[str],
    ) -> dict:
        """跨公司比较题：每家公司**单独**过滤召回一次，再按公司配额合并。

        不做 per-company 召回时，top-4 会被语料里更相似的那一家占满 —— 实测 v1 的
        4 道 cross_doc_compare，被点名的两家公司总是只进一家，模型只剩「资料不足」
        可说。槽位总量不变：仍是 TOP_K_RERANK 条，均分给每家公司。
        """
        per = max(1, TOP_K_RERANK // len(companies))
        all_vec, all_bm25, final = [], [], []
        for code in companies:
            vec, bm25, candidates = self._search_candidates(retrieval_query, {"stock_code": code})
            all_vec  += vec
            all_bm25 += bm25
            if self.use_rerank:
                final += rerank(expanded_question, candidates, per)
            else:
                final += candidates[:per]

        return {
            "query":        retrieval_query,
            "vec_results":  all_vec,
            "bm25_results": all_bm25,
            "candidates":   final,
            "final":        final,
        }

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

        # ①.5 跨公司比较题走按公司配额的分支（调用方显式给了 filter_meta 时不接管，
        #      那是人工指定过滤，语义优先）。
        if filter_meta is None:
            companies = infer_companies_from_question(question, self.vec_store.meta_list)
            if len(companies) >= 2:
                if verbose:
                    logger.info(f"跨公司检索：{'、'.join(companies)} 每家公司各取配额")
                return self._retrieve_multi_company(
                    retrieval_query, expanded_question, companies)

        # ② 向量检索 + ③ BM25 + RRF 融合
        vec_results, bm25_results, candidates = self._search_candidates(retrieval_query, filter_meta)
        if verbose:
            logger.info(f"向量召回: {len(vec_results)} 条，最高分={vec_results[0]['vec_score']:.3f}" if vec_results else "向量召回: 0 条")
            if self.use_bm25 and self.bm25_store:
                logger.info(f"BM25 召回: {len(bm25_results)} 条，RRF 后: {len(candidates)} 条")

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

    def _row_unit(self, item: dict, row: Optional[list] = None) -> str:
        """这行数字的计量单位。两层来源，都不成立就返回空串（不猜）：
          1. 行标签 / 表头自带的「（元）（万元）」后缀 —— 就在索引里，最可靠；
          2. data/parsed 里窗口内的页级「单位：X」声明。
        """
        unit = unit_from_labels(*(row or []), *((item.get("table_json") or [[]])[0]))
        if unit:
            return unit
        parsed_dir = getattr(self, "unit_dir", None)
        if not parsed_dir:
            return ""
        try:
            page = int(item.get("page_num"))
        except (TypeError, ValueError):
            return ""
        return unit_at(load_unit_declarations(item.get("source_file", ""), parsed_dir), page)

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

        公司必须高置信（`table_pool` 的单公司硬闸不变）；年份**允许多个** ——
        趋势题按年度各取一行，再合并成**一条**上下文，所以送进 LLM 的上下文条数
        与修复前完全相同（4 条检索 + 至多 1 条补位），不占额外槽位。
        """
        pool = self.table_pool(
            filter_meta, question,
            stock_codes=infer_companies_from_question(question, self.vec_store.meta_list))
        if not pool:
            return []
        filt = dict(filter_meta or {})
        for k, v in infer_filter_from_question(
                question, self.vec_store.meta_list).items():
            filt.setdefault(k, v)
        # 旧闸是 `if not filt.get("year"): return []`。而 infer 层遇到「2021到2023」
        # 这种多个年份时**故意**不锁年份（tests/test_table_lookup.py 里
        # test_infer_filter_skips_ambiguous_years 钉着这条），于是这道闸把全部跨年
        # 趋势题在取数之前就拒了 —— 实测 multihop 34 题没有一条拿到补位上下文。
        # 原注释的理由（补一行会挤占逐年对比证据、Context Precision 下滑）成立的前提
        # 正是「只能补一行」；现在按年度各补一行并合并成一条，前提不再成立。
        # 保留的约束：问题里一个年份都没有时仍然不补（没有年份就无从判断取哪年）。
        if not filt.get("year") and not _YEARS_RE.findall(_norm_cell(question)):
            return []
        qn = _norm_cell(question)
        names_map = company_name_map(self.vec_store.meta_list)
        # 公司名碎片（"宁德"/"时代"）不能当指标名。跨公司题池子里有多家，
        # 按**全部被点名公司**过滤，而不只看池子第一块那一家。
        company_names = {names_map[c] for c in
                         infer_companies_from_question(question, self.vec_store.meta_list)
                         if names_map.get(c)}
        if not company_names and pool:
            company_names = {names_map.get(pool[0].get("stock_code", ""), "")}
        labels: set[str] = set()
        aliases: set[str] = set()
        cell_labels: set[str] = set()
        for item in pool:
            for row in item.get("table_json") or []:
                if not row:
                    continue
                label = _LABEL_FILLER.sub("", _norm_label(row[0]))
                cell_labels.add(label)
                if any(cn and (label in cn or cn in label) for cn in company_names):
                    continue
                # 表格首列常混着 "2021年" "1" 这类序号/年份，它们在问题里
                # 也出现且往往比指标名更长，必须排除，否则会用年份去取行。
                # 中文指标名至少 3 字：2 字以内必然是碎片 —— 实测"哪家**公司**更高"
                # 会匹配到环保表的"公司"行，把污染物名称当成财务指标取出来。
                # 代价（已知）："商誉""存货"这类 2 字指标不再补位，退回正常检索。
                if (not label[0].isdigit()
                        and len(_CJK_RE.findall(label)) >= 3
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

        found_all: list[dict] = []
        for label in candidates:
            for name in (label, *_TABLE_LABEL_ALIASES.get(label, ())):
                # 数值列必须至少有一格像金额，否则这行是文字说明（污染物名称、
                # 审计意见里的"营业收入"段落），取出来只会污染上下文。
                found = [r for r in lookup_table_value(pool, name)
                         if _norm_label(r["row"][0]) == _norm_label(name)
                         and any(looks_like_amount(v) for v in r["row"][1:])]
                if found:
                    found_all = found          # 不在这里截断，交给 select_backfill_rows 分组
                    break
            if found_all:
                break
        if not found_all:
            return []

        by_chunk = {item.get("chunk_id"): item for item in pool}
        # 旧实现是 `rows = found[:1]`：同一指标在一份年报里被摘要表、合并表、母公司表、
        # 附注各写一遍时，只留块顺序最靠前的那一条，跨年趋势题因此永远只拿到一个年度。
        rows = select_backfill_rows(found_all, by_chunk)

        out: list[str] = []
        units: list[str] = []
        first_row: Optional[dict] = None
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
            out.append(content)
            units.append(self._row_unit(src, r["row"]))
            if first_row is None:
                first_row = r
        if not out:
            return []

        # 单位标注：只在**至少解析出一个单位**时才动这些行 —— 全未知就保持修复前的
        # 逐字输出（tests/test_backfill_multiyear 钉着它），不凭空加噪声。
        # 混合场景（跨公司/跨表补位）里查不到单位的那几行显式标「未标注」，
        # 让模型看得见"这两行的量纲可能对不上"，而不是拿 11,392,948,404.47 去和
        # 1,835,610.84 直接比大小（缺陷 2 的实测形状）。
        if any(units):
            out = [c + (f"（单位：{u}）" if u else UNIT_UNKNOWN_MARK)
                   for c, u in zip(out, units)]

        # 多个年度/多家公司的行**合并成一条**上下文：条数与修复前一致（不额外占槽位），
        # 但每个年度/公司的数值都带着自己的出处写在这一条里。
        # 引用的 chunk_id/page_num/section 仍取第一条来源，与修复前单行时逐字段一致。
        return [{
            "chunk_id": first_row["chunk_id"],
            "stock_code": first_row["stock_code"],
            "year": first_row["year"],
            "page_num": first_row["page_num"],
            "section": first_row["section"],
            "content": "\n".join(out),
        }]

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

        # ⑤.6 跨公司覆盖校验：点名的公司只要有一家连一条证据（含补位行）都没进
        #      上下文，就不让 LLM 在半边证据上作答 —— 实测它会拿手里那一家加上
        #      常识编出比较结论（v1 Q14 把 2023 年报的 AI 描述当成 2022 年作答）。
        companies = infer_companies_from_question(question, self.vec_store.meta_list)
        if len(companies) >= 2:
            missing = missing_companies(companies, final)
            if missing:
                names = company_name_map(self.vec_store.meta_list)
                who = "、".join(f"{names.get(m, m)}（{m}）" for m in missing)
                return {
                    "answer": f"参考资料中缺少 {who} 的年报内容，无法完成跨公司比较。",
                    "citations": [], "retrieved": final,
                }

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
