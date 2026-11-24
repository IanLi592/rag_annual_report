"""
年报 RAG · Streamlit 简易问答页

用最少的代码把 src/rag_pipeline.py 包成一个可点选的网页：
  - 问答模式：调 RAGPipeline.query()，返回答案 + 引用来源（与 HTTP /query 同口径）
  - 检索调试模式：调 RAGPipeline.retrieve()，看向量 / BM25 / RRF 三路召回明细（与 /query/debug 同口径）

两种模式各只做一次检索，不会同时再额外调用一次 embedding（不重复计费）。

启动：
  pip install streamlit            # 若尚未安装
  streamlit run app.py            # 默认 http://localhost:8501

依赖 DASHSCOPE_API_KEY：本文件会按项目既有约定自动从根目录 .env 读取
（已存在的环境变量优先，不覆盖命令行显式传入的值）。
"""

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
# 让 `from rag_pipeline import ...` 生效；rag_pipeline 内部还会 import jieba_domain，
# 同样依赖 src 在 sys.path 上，所以这里把 src 目录整体加进来。
sys.path.insert(0, str(ROOT / "src"))


def _load_dotenv_if_present() -> None:
    """裸跑本文件时也能拿到 key，语义与 scripts/ci/check_judge.py 的 loader 一致：
    KEY=VALUE、跳过空行与 # 注释、剥成对引号、已存在的环境变量优先（命令行 > .env）。
    """
    envf = ROOT / ".env"
    if not envf.exists():
        return
    for line in envf.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        k, v = k.strip(), v.strip()
        if len(v) >= 2 and v[0] == v[-1] and v[0] in "\"'":
            v = v[1:-1]
        os.environ.setdefault(k, v)


_load_dotenv_if_present()

import streamlit as st
from rag_pipeline import RAGPipeline, company_name_map, resolve_chunk_content

# ── 索引选项：默认 semantic，另给一套 structured（带表格父块）用于对照 ──────────
INDEX_CHOICES = {
    "semantic（默认）": str(ROOT / "vectorstore"),
    "structured（表格父块）": str(ROOT / "vectorstore" / "structured"),
}

EXAMPLES = [
    "贵州茅台2023年营业收入是多少",
    "五粮液2022年的净利润",
    "比较贵州茅台和五粮液2022年的研发费用",
    "宁德时代2021年到2023年的总资产趋势",
]


@st.cache_resource(show_spinner="加载 FAISS 索引 + 构建 BM25（约 10 秒，只跑一次）...")
def get_pipeline(index_dir: str, use_bm25: bool, use_rerank: bool) -> RAGPipeline:
    """按 (索引目录, BM25, Rerank) 缓存一个 Pipeline 实例。

    这三个是构造期开关，改了才会重建；公司/年份是查询期参数，改它们不会重建。
    Rerank 首次启用会懒加载 BAAI/bge-reranker-base（约 1.1 GB，见 get_reranker）。
    """
    return RAGPipeline(
        use_bm25=use_bm25,
        use_rerank=use_rerank,
        use_query_rewrite=False,
        index_dir=index_dir,
    )


def _src_label(item: dict) -> str:
    """把 chunk metadata 格式化成简短来源串（与 serve._build_source 同口径）。"""
    s = f"{item.get('stock_code', '')} {item.get('year', '')}年报"
    section = item.get("section", "")
    if section:
        parts = section.split(" > ")
        s += " · " + " > ".join(parts[-2:])
    page = item.get("page_num", -1)
    if page and page != -1:
        s += f" · 第{page}页"
    return s


def _preview(text: str, n: int = 120) -> str:
    text = (text or "").strip().replace("\n", " ")
    return text[:n] + "…" if len(text) > n else text


# ── 页面 ────────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="年报 RAG 问答", page_icon="📊", layout="centered")
st.title("📊 上市公司年报 RAG 问答")
st.caption("FAISS + BM25 混合检索 · DashScope qwen-plus 生成 · 引用可溯源")

if not os.getenv("DASHSCOPE_API_KEY"):
    st.error("未检测到 DASHSCOPE_API_KEY。请在项目根目录的 .env 里写一行 "
             "`DASHSCOPE_API_KEY=sk-xxx`，或先在终端 `export` 后重启本页。")
    st.stop()

with st.sidebar:
    st.header("检索配置")
    index_label = st.selectbox("向量索引", list(INDEX_CHOICES.keys()))
    index_dir = INDEX_CHOICES[index_label]
    use_bm25 = st.toggle("BM25 关键词召回", value=True,
                         help="关掉就只剩纯向量检索（消融对照）。")
    use_rerank = st.toggle("CrossEncoder 精排", value=False,
                           help="首次启用会下载 bge-reranker-base（约 1.1 GB）；"
                                "模型不可用时自动降级为直接截断。")
    st.divider()
    st.header("按公司 / 年份过滤")
    st.caption("过滤是查询期参数，改它不会重建索引。")

# 构建（或复用缓存的）Pipeline
pipeline = get_pipeline(index_dir, use_bm25, use_rerank)

# 从真实索引元数据里取公司清单与年份清单——不另写一张公司表，换数据集不会与现实脱节。
names = company_name_map(pipeline.vec_store.meta_list)
code_to_label = {
    c: (f"{n}（{c}）" if n else c) for c, n in sorted(names.items())
}
years = sorted({str(m.get("year", "")) for m in pipeline.vec_store.meta_list if m.get("year")})

with st.sidebar:
    sel_company = st.selectbox("公司", ["不限"] + list(code_to_label.values()))
    sel_year = st.selectbox("年份", ["不限"] + years)

    if code_to_label:
        st.caption(f"语料含 {len(code_to_label)} 家公司 · 年份 {years[0]}–{years[-1]}"
                   if years else f"语料含 {len(code_to_label)} 家公司")

# 组装查询期 filter_meta
filter_meta = {}
if sel_company != "不限":
    filter_meta["stock_code"] = next(
        c for c, lbl in code_to_label.items() if lbl == sel_company)
if sel_year != "不限":
    filter_meta["year"] = sel_year
filter_meta = filter_meta or None

st.success(f"索引就绪：{index_label} · {pipeline.vec_store.index.ntotal} 条向量 "
           f"· 策略={pipeline.strategy or '—'}")

mode = st.radio("模式", ["问答", "检索调试"], horizontal=True,
                help="问答：调 query()，返回答案+引用。检索调试：调 retrieve()，"
                     "看三路召回明细，不调用 LLM、不生成答案。")

def _fill_example(text: str) -> None:
    """示例按钮回调。回调在脚本重跑之前执行，把问题写进带 key 的 session_state，
    st.text_input 下一帧就会显示它。直接 `question = ex` 只改当前帧的变量、不会回填控件。"""
    st.session_state.q_input = text

question = st.text_input("问题", key="q_input",
                         placeholder="例如：贵州茅台2023年营业收入是多少")

col1, col2, col3, col4 = st.columns(4)
for c, ex in zip((col1, col2, col3, col4), EXAMPLES):
    c.button(ex, use_container_width=True, on_click=_fill_example, args=(ex,))

if not question:
    st.info("输入一个问题，或点上面的示例。")
    st.stop()

if filter_meta:
    shown = " + ".join(f"{k}={v}" for k, v in filter_meta.items())
    st.caption(f"已应用过滤：{shown}（侧栏可清除）")

run = st.button("运行", type="primary")
if not run:
    st.stop()

try:
    if mode == "问答":
        with st.spinner("检索 + 生成中…"):
            result = pipeline.query(question, filter_meta=filter_meta, verbose=True)

        st.subheader("答案")
        st.markdown(result["answer"])

        if result["citations"]:
            st.subheader("引用来源")
            for c in result["citations"]:
                st.markdown(f"- `[{c['index']}]` {c['source']}")

        with st.expander(f"送进 LLM 的上下文（{len(result['retrieved'])} 段）"):
            for i, item in enumerate(result["retrieved"], 1):
                content = resolve_chunk_content(item, pipeline.parents)
                st.markdown(f"**[{i}] {_src_label(item)}**")
                st.text(_preview(content, 600))
    else:
        with st.spinner("检索中…（不调用 LLM）"):
            r = pipeline.retrieve(question, filter_meta=filter_meta, verbose=True)

        st.caption(f"实际检索 query：`{r['query']}`")

        st.subheader("① 向量召回（FAISS Top-5）")
        st.dataframe([
            {"排名": i + 1, "vec_score": round(it.get("vec_score", 0.0), 4),
             "来源": _src_label(it), "内容预览": _preview(it["content"])}
            for i, it in enumerate(r["vec_results"][:5])
        ], use_container_width=True, hide_index=True)

        st.subheader("② BM25 关键词召回（Top-5）")
        if r["bm25_results"]:
            st.dataframe([
                {"排名": i + 1, "bm25_score": round(it.get("bm25_score", 0.0), 3),
                 "来源": _src_label(it), "内容预览": _preview(it["content"])}
                for i, it in enumerate(r["bm25_results"][:5])
            ], use_container_width=True, hide_index=True)
        else:
            st.info("BM25 未召回结果（可在侧栏确认「BM25 关键词召回」已开启）。")

        st.subheader("③ 融合后最终送模型的上下文（Top-4）")
        st.dataframe([
            {"排名": i + 1, "来源": _src_label(it),
             "内容预览": _preview(it["content"])}
            for i, it in enumerate(r["final"][:4])
        ], use_container_width=True, hide_index=True)
except Exception as e:
    st.error(f"运行出错：{type(e).__name__}: {e}")
