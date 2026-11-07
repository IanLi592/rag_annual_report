"""
RAG 问答 HTTP 服务（FastAPI）

提供两类接口：
  /query        — 标准问答，返回答案 + 引用
  /query/debug  — 调试接口，逐步返回每个检索阶段的中间结果
  /             — 可视化 Web 页面

启动：
  cd src
  uvicorn serve:app --host 0.0.0.0 --port 8000

依赖：
  pip install fastapi uvicorn
"""

import importlib.util
import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

_PIPELINE_PATH = Path(__file__).parent / "rag_pipeline.py"
_STATIC_DIR    = Path(__file__).parent / "static"


# ── 动态加载 rag_pipeline（importlib 方式，保持与启动目录无关）────────────────

def _load_pipeline_module():
    spec   = importlib.util.spec_from_file_location("rag_pipeline", _PIPELINE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# ── 全局状态 ──────────────────────────────────────────────────────────────────

_module  = None   # 持有模块引用，供 debug 端点访问模块级函数
pipeline = None   # RAGPipeline 实例


def _env_float(name: str, default: float | None = None) -> float | None:
    """读浮点环境变量；空串/未设置时回 default。"""
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    return float(raw)


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _module, pipeline
    logger.info("服务启动，初始化 RAG Pipeline...")
    _module  = _load_pipeline_module()
    index_dir = os.getenv("RAG_INDEX_DIR") or None
    pipeline = _module.RAGPipeline(
        use_bm25          = True,
        # 精排打开，但分数下限默认关闭：
        #   100 题实测（scripts/ci/measure_rerank.py --sample 100）
        #   基线 RRF top4      Hit@4 0.560 / MRR@4 0.312
        #   重排不过滤          Hit@4 0.550 / MRR@4 0.330
        #   重排 + 下限 0.70    Hit@4 0.410 / MRR@4 0.261  ← 旧默认，已回退
        # 0.7 是在 25 题子集上选的，100 题上会误杀大量正确文档。
        # 要重新启用下限，先用 measure_rerank.py 在 100 题上确定新阈值。
        use_rerank        = True,
        rerank_min_score  = _env_float("RERANK_MIN_SCORE", None),
        use_query_rewrite = False,
        index_dir         = index_dir,
    )
    logger.info(
        f"Pipeline 初始化完成：索引={pipeline.index_dir} 策略={pipeline.strategy} "
        f"rerank_min_score={pipeline.rerank_min_score}")
    yield
    logger.info("服务关闭")


# ── FastAPI 应用 ──────────────────────────────────────────────────────────────

app = FastAPI(
    title       = "年报 RAG 问答服务",
    description = "FAISS + BM25 混合检索 + DashScope qwen-plus，含检索链路调试接口",
    version     = "1.0.0",
    lifespan    = lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
)


# ── 数据模型 ──────────────────────────────────────────────────────────────────

class QueryRequest(BaseModel):
    # example 走 json_schema_extra：Field(example=...) 是 Pydantic V2 的废弃用法，
    # V3.0 会移除（pytest 里以 PydanticDeprecatedSince20 告警形式出现过 3 条）。
    question:   str           = Field(...,  json_schema_extra={"example": "贵州茅台2023年营业收入是多少"})
    stock_code: Optional[str] = Field(None, json_schema_extra={"example": "600519"})
    year:       Optional[str] = Field(None, json_schema_extra={"example": "2023"})

class Citation(BaseModel):
    index:    int
    source:   str
    chunk_id: str

class QueryResponse(BaseModel):
    answer:    str
    citations: list[Citation]

class TableLookupRequest(BaseModel):
    question:  str           = Field(..., json_schema_extra={"example": "贵州茅台2023年财务数据"})
    row_label: str           = Field(..., json_schema_extra={"example": "营业收入"})
    stock_code: Optional[str] = Field(None, json_schema_extra={"example": "600519"})
    year:       Optional[str] = Field(None, json_schema_extra={"example": "2023"})

class TableRow(BaseModel):
    chunk_id:   str
    stock_code: str
    year:       str
    page_num:   int
    section:    str
    row:        list[str]

class TableLookupResponse(BaseModel):
    question:  str
    row_label: str
    matched:   int
    rows:      list[TableRow]

# ── Debug 专用模型 ────────────────────────────────────────────────────────────

class VecChunk(BaseModel):
    rank:            int
    vec_score:       float
    source:          str
    content_preview: str

class BM25Chunk(BaseModel):
    rank:            int
    bm25_score:      float
    source:          str
    content_preview: str

class RRFChunk(BaseModel):
    rank:            int
    rrf_score:       float
    vec_rank:        Optional[int]
    bm25_rank:       Optional[int]
    source:          str
    content_preview: str

class ContextChunk(BaseModel):
    index:   int
    source:  str
    content: str          # 完整内容，用于展示"输入 LLM 的原文"

class DebugResponse(BaseModel):
    question:       str
    vec_results:    list[VecChunk]
    bm25_results:   list[BM25Chunk]
    rrf_results:    list[RRFChunk]
    context_chunks: list[ContextChunk]
    answer:         str
    citations:      list[Citation]


# ── 工具函数 ──────────────────────────────────────────────────────────────────

def _build_source(item: dict) -> str:
    """把 chunk metadata 格式化为可读来源字符串。"""
    s = f"{item.get('stock_code','')} {item.get('year','')}年报"
    section = item.get("section", "")
    if section:
        # 只保留最后两级，避免过长
        parts = section.split(" > ")
        s += " · " + " > ".join(parts[-2:])
    page = item.get("page_num", -1)
    if page and page != -1:
        s += f" · 第{page}页"
    return s

def _preview(text: str, n: int = 150) -> str:
    text = text.strip()
    return text[:n] + "…" if len(text) > n else text

def _filter_meta(req: QueryRequest) -> Optional[dict]:
    fm = {}
    if req.stock_code: fm["stock_code"] = req.stock_code
    if req.year:       fm["year"]        = req.year
    return fm or None


# ── 接口 ──────────────────────────────────────────────────────────────────────

@app.get("/", include_in_schema=False)
def index():
    """返回可视化页面。"""
    return FileResponse(_STATIC_DIR / "index.html")


@app.get("/health", summary="健康检查")
def health():
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline 尚未初始化")
    return {
        "status": "ok",
        "pipeline_ready": True,
        "index_dir": str(pipeline.index_dir),
        "strategy": pipeline.strategy,
        "rerank_min_score": pipeline.rerank_min_score,
    }


@app.post("/query", response_model=QueryResponse, summary="标准问答")
def query(req: QueryRequest):
    """标准问答接口，返回最终答案和来源引用。"""
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline 尚未初始化")
    try:
        result = pipeline.query(req.question, filter_meta=_filter_meta(req), verbose=True)
    except Exception as e:
        logger.error(f"Pipeline 异常: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
    return QueryResponse(
        answer    = result["answer"],
        citations = [Citation(**c) for c in result["citations"]],
    )


@app.post("/table/lookup", response_model=TableLookupResponse, summary="按行标签精确取表格行")
def table_lookup(req: TableLookupRequest):
    """在检索候选的 `table_json` 中按首列标签取整行。

    与 /query 共用检索入口，但搜索的是 RRF 后的全部候选（而非精排 top-4），
    避免精排把正确表格挤掉；命中后直接读原始二维结构，不经过 LLM。
    """
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline 尚未初始化")
    try:
        r = pipeline.retrieve(req.question, _filter_meta(req), verbose=False)
        rows = _module.lookup_table_value(r["candidates"], req.row_label)
    except Exception as e:
        logger.error(f"Table lookup 异常: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
    return TableLookupResponse(
        question=req.question,
        row_label=req.row_label,
        matched=len(rows),
        rows=[TableRow(**row) for row in rows],
    )


@app.post("/query/debug", response_model=DebugResponse, summary="调试接口：逐步返回中间结果")
def query_debug(req: QueryRequest):
    """
    调试接口，逐步执行 RAG 流水线并返回每步的中间结果：

    1. 向量检索（vec_results）：FAISS 召回 Top-10，含余弦相似度分数
    2. BM25 检索（bm25_results）：关键词召回 Top-10，含 BM25 分数
    3. RRF 融合（rrf_results）：两路合并后 Top-5，标注每条来自哪一路
    4. 输入 LLM 的上下文（context_chunks）：最终 Top-4 的完整文本
    5. 生成答案（answer + citations）
    """
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline 尚未初始化")

    fm = _filter_meta(req)

    try:
        # 与 /query 共用同一个检索入口（改写→向量/BM25→RRF→精排同阈值），
        # 调试接口不再自己复制一份逻辑。
        r = pipeline.retrieve(req.question, fm, verbose=False)
        vec_results  = r["vec_results"]
        bm25_results = r["bm25_results"]
        candidates   = r["candidates"]
        final        = r["final"]

        vec_rank_map  = {item["chunk_id"]: rank for rank, item in enumerate(vec_results, 1)}
        bm25_rank_map = {item["chunk_id"]: rank for rank, item in enumerate(bm25_results, 1)}

        # 与 /query 完全同口径的父块拼装
        context, cits = _module.build_context(final, pipeline.parents)

        # ⑤ LLM 生成
        answer = _module.call_llm(req.question, context, pipeline.client)

    except Exception as e:
        logger.error(f"Debug Pipeline 异常: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

    return DebugResponse(
        question = req.question,
        vec_results = [
            VecChunk(
                rank=i+1,
                vec_score=item.get("vec_score", 0.0),
                source=_build_source(item),
                content_preview=_preview(item["content"]),
            ) for i, item in enumerate(vec_results[:5])
        ],
        bm25_results = [
            BM25Chunk(
                rank=i+1,
                bm25_score=item.get("bm25_score", 0.0),
                source=_build_source(item),
                content_preview=_preview(item["content"]),
            ) for i, item in enumerate(bm25_results[:5])
        ],
        rrf_results = [
            RRFChunk(
                rank=i+1,
                rrf_score=item.get("rrf_score", 0.0),
                vec_rank=vec_rank_map.get(item["chunk_id"]),
                bm25_rank=bm25_rank_map.get(item["chunk_id"]),
                source=_build_source(item),
                content_preview=_preview(item["content"]),
            ) for i, item in enumerate(candidates[:5])
        ],
        context_chunks = [
            ContextChunk(
                index=i+1,
                source=_build_source(item),
                content=_module.resolve_chunk_content(item, pipeline.parents),
            ) for i, item in enumerate(final)
        ],
        answer    = answer,
        citations = [Citation(**c) for c in cits],
    )
