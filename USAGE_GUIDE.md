# 代码调用与测试指南

> 上市公司年报 RAG 问答系统

---

## 📖 本文档导读

**文档定位**：全项目的"操作手册"——所有命令、参数、报错处理都在这里，边跑代码边查。

**使用方式**：
1. 按"一、环境准备"一次性配好依赖与 API Key（`DASHSCOPE_API_KEY`，评估还需 `DEEPSEEK_API_KEY`）；
2. 按"二、整体流程"从步骤 1 顺序跑到步骤 5；
4. "三、在代码中直接调用"教你把 RAG 当模块嵌入自己的程序；
5. "四、HTTP 服务模式"启动 FastAPI 后可打开浏览器访问 `http://localhost:8000` 查看可视化页面；
6. "五、评估流程"跑 RAGAS 与消融实验；
7. 遇到报错先查"六、调试与常见问题"。

**命令 ↔ 源码对照**：每个"步骤"标题下的命令分别对应 `src/download_reports.py` → `src/parse_pdf.py` → `src/chunk_documents.py` → `src/build_index.py` → `src/rag_pipeline.py`（服务为 `src/serve.py`，评估为 `evaluation/` 两个脚本），命令行参数的含义都在对应源码的 argparse 定义处（源码中已加注释）。

**关联文档**：想先理解原理再动手，读 `ARCHITECTURE.md`。

---

## 一、环境准备

### 1.1 安装依赖

```bash
pip install -r requirements.txt
```

主要依赖清单：

```
pdfplumber / pymupdf          # PDF 解析
faiss-cpu                     # 向量索引
rank_bm25 / jieba             # 关键词检索
openai                        # DashScope / OpenAI 兼容接口
sentence-transformers          # 本地模型
ragas / datasets              # 评估框架
```

### 1.2 配置 API Key

```bash
# Windows
set DASHSCOPE_API_KEY=sk-xxxxxxxx

# PowerShell
$env:DASHSCOPE_API_KEY = "sk-xxxxxxxx"

# Linux / macOS
export DASHSCOPE_API_KEY=sk-xxxxxxxx
```

DashScope 控制台：https://dashscope.console.aliyun.com/

跑评估（`evaluation/evaluate.py`）时，RAGAS 的 LLM 打分走 DeepSeek，还需额外配置：

```bash
# PowerShell
$env:DEEPSEEK_API_KEY = "sk-xxxxxxxx"

# Linux / macOS
export DEEPSEEK_API_KEY=sk-xxxxxxxx
```

Embedding 仍走 DashScope（`text-embedding-v3`），只有 LLM 打分用 DeepSeek。

---

## 二、整体流程

### 步骤 1：下载年报 PDF

```bash
python src/download_reports.py
```

**内部流程**：
1. 从巨潮资讯网 API 搜索 5 家公司 × 3 年 = 15 份年报
2. 搜索策略：`searchkey = "公司名+年份+年度报告"`
3. 五粮液特殊处理：fallback 为 `"五 粮 液"` 字间加空格（cninfo 数据库存储格式）
4. 下载到 `data/raw_pdf/`，生成 `data/manifest.json`

**预期输出**：
```
data/raw_pdf/
  600519_2021_贵州茅台_贵州茅台2021年年度报告.pdf   3.2 MB
  600519_2022_贵州茅台_贵州茅台2022年年度报告.pdf   3.2 MB
  600519_2023_贵州茅台_贵州茅台2023年年度报告.pdf   3.4 MB
  000858_2021_五粮液_2021年年度报告.pdf             3.4 MB
  ... （共 15 个文件，约 85 MB）
```

---

### 步骤 2：解析 PDF

```bash
python src/parse_pdf.py
```

**内部流程**：
1. 遍历 `manifest.json` 中每个 PDF
2. 每页同时用 pdfplumber（表格）和 PyMuPDF（文字+字体）处理
3. 判断扫描页：文字少于 50 字且含图片 → 调用 OCR（未安装 tesseract 时跳过并留占位符）
4. 表格转为 Markdown 格式，标题按字体大小/加粗识别，维护章节路径栈
5. 输出每个 PDF 对应的 JSON 文件

**预期输出**：
```
data/parsed/
  600519_2023_贵州茅台_贵州茅台2023年年度报告.json   # 约 1,660 个 blocks
  601318_2023_中国平安_中国平安2023年年度报告.json   # 约 7,550 个 blocks（平安表格最多）
  ... （共 15 个 JSON）
```

输出 JSON 结构：
```json
{
  "meta": {"filename": "600519_2023...", "stock_code": "600519", "year": "2023"},
  "blocks": [
    {
      "block_type": "title",
      "content": "第三章 管理层讨论与分析",
      "page_num": 21,
      "section_path": ["第三章 管理层讨论与分析"],
      "is_ocr": false
    },
    ...
  ]
}
```

---

### 步骤 3：文档分块

```bash
python src/chunk_documents.py
```

**控制参数**（修改脚本顶部变量）：

```python
STRATEGY = "semantic"   # 可选: "fixed" | "semantic" | "hierarchical"
```

**三种策略的区别**：

| 策略 | 切割依据 | chunk 大小 | 适合场景 |
|------|---------|-----------|---------|
| `fixed` | 每 500 字符截断，overlap=50 | 均匀 | Baseline 对比 |
| `semantic` | 遇标题强制切，段落合并不超 800 字 | 不均匀 | 默认，语义完整 |
| `hierarchical` | 父块（完整章节）+ 子块（细粒度）| 双层 | 长文档精确召回 |

**预期输出**：
```
data/chunks/all_semantic.json   # 10,352 个 chunks，文件 14.96 MiB（实测 15,687,726 字节）
```

每个 chunk 的格式：
```json
{
  "chunk_id": "600519_2023_0423",
  "content": "报告期内，公司实现营业总收入1,476.94亿元...",
  "metadata": {
    "stock_code": "600519",
    "year": "2023",
    "page_num": 56,
    "section": "第十节 > 二、财务报表 > 利润表",
    "block_types": ["text"],
    "is_ocr": false,
    "strategy": "semantic",
    "source_file": "600519_2023_贵州茅台_贵州茅台2023年年度报告.pdf"
  }
}
```

---

### 步骤 4：构建向量索引

```bash
python src/build_index.py
```

**内部流程**：
1. 加载 `data/chunks/all_semantic.json`（10,352 条）
2. 按批次（10 条/批，共 ~1036 批）调用 DashScope text-embedding-v3
3. 对所有向量做 L2 归一化（使内积等价于余弦相似度）
4. 构建 FAISS IndexFlatIP，批量 add 归一化向量
5. 分别保存索引文件和元数据

**耗时与费用**：约 5~10 分钟；10,352 条 × 平均 50 token ≈ 50 万 token ≈ **约 0.35 元**

**预期输出**：
```
vectorstore/faiss_index.bin   # 40.44 MiB（10352 × 1024 × 4 bytes）
vectorstore/faiss_meta.json   # 14.98 MiB（每条 chunk 的文本 + 溯源元数据）
```

**进度日志**（每 100 批打印一次）：
```
2026-04-16 [INFO] Embedding 进度: 100/1036 批
2026-04-16 [INFO] Embedding 进度: 200/1036 批
...
2026-04-16 [INFO] 索引构建完成，共 10352 条向量
2026-04-16 [INFO] FAISS 索引已保存 → vectorstore/faiss_index.bin  (41408 KB)
```

---

### 步骤 5：原生版问答

```bash
# 交互式
python src/rag_pipeline.py

# 单次查询
python src/rag_pipeline.py --query "贵州茅台2023年营业收入是多少"

# 加过滤条件（只检索特定公司/年份）
python src/rag_pipeline.py --query "营业收入" --stock 600519 --year 2023

# 开启查询改写
python src/rag_pipeline.py --query "茅台最近怎么样" --query-rewrite

# 消融测试：关闭 BM25 或 Rerank
python src/rag_pipeline.py --query "..." --no-bm25
python src/rag_pipeline.py --query "..." --no-rerank
```

**内部流程**（以默认配置为例）：

```
用户输入 "贵州茅台2023年营业收入是多少"
    │
    ├─ [可选] qwen-turbo 查询改写
    │
    ├─ DashScope API 生成查询向量（1 次 API 调用）
    │
    ├─ FAISS 内积搜索 → top-10（<1 ms）
    │
    ├─ jieba 分词 + BM25Okapi 打分 → top-10（CPU，约 0.1 s）
    │
    ├─ RRF 融合 → top-19（去重后合并）
    │
    ├─ [可选] CrossEncoder Rerank → top-4
    │   （模型未下载时自动降级为直接截断）
    │
    ├─ 相关性阈值检查（vec_score < 0.25 时拒绝回答）
    │
    └─ qwen-plus LLM 生成（1 次 API 调用）
```

**实际输出示例**：

```
问题：贵州茅台2023年营业收入是多少

贵州茅台2023年营业收入为人民币14,769,360.50万元[2]。

── 来源 ──
  [1] 600519 2022年报 · 第十节 > 三、关键审计事项 · 第52页
  [2] 600519 2023年报 · 第十节 > 三、关键审计事项 · 第56页
  [3] 600519 2023年报 · 第十节 > 二、财务报表 > 母公司资产负债表 · 第61页
  [4] 600519 2023年报 · 第十节 > 二、非持续的公允 · 第133页
```

**交互式模式特殊命令**：
- `exit`：退出
- `mode`：查看当前 BM25 / Rerank / QueryRewrite 开关状态

---

## 三、在代码中直接调用（作为模块使用）

### 调用原生版 Pipeline

```python
import sys
sys.path.insert(0, "src")
from rag_pipeline import RAGPipeline

# 初始化（首次约需 10 秒：加载索引 + 构建 BM25）
pipeline = RAGPipeline(
    use_bm25=True,
    use_rerank=False,         # Rerank 需要本地模型
    use_query_rewrite=False,
)

# 单次查询
result = pipeline.query("贵州茅台2023年营业收入是多少")
print(result["answer"])
print(result["citations"])     # 来源列表
print(len(result["retrieved"])) # 实际使用的 chunk 数

# 带过滤条件
result = pipeline.query(
    "营业收入是多少",
    filter_meta={"stock_code": "600519", "year": "2023"},
)
```

`result` 的数据结构：
```python
{
    "answer":    "贵州茅台2023年营业收入为...",
    "citations": [
        {"index": 1, "source": "600519 2023年报 · ...", "chunk_id": "600519_2023_0423"},
        ...
    ],
    "retrieved": [  # 完整的 chunk 列表（含 metadata）
        {"content": "...", "stock_code": "600519", "year": "2023", "page_num": 56, ...},
        ...
    ]
}
```

---

## 四、HTTP 服务模式

### 5.1 启动服务

```bash
pip install fastapi uvicorn   # 若尚未安装

cd src
uvicorn serve:app --host 0.0.0.0 --port 8000
```

启动日志：
```
服务启动，初始化 RAG Pipeline...
FAISS 索引加载完成，共 10352 条向量
构建 BM25 索引（分词中，请稍候）...
BM25 索引完成
Pipeline 初始化完成，开始接受请求
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:8000
```

FAISS 和 BM25 只在启动时加载一次，之后每次请求只需 2 次 DashScope API 调用。

开发模式（修改代码后自动重载）：
```bash
uvicorn serve:app --host 0.0.0.0 --port 8000 --reload
```

---

### 5.2 接口说明

#### `POST /query` — 问答（主接口）

请求体：
```json
{
  "question":   "贵州茅台2023年营业收入是多少",
  "stock_code": "600519",   // 可选，限定股票范围
  "year":       "2023"      // 可选，限定年份范围
}
```

响应体：
```json
{
  "answer": "贵州茅台2023年营业收入为人民币14,769,360.50万元[2]。",
  "citations": [
    {"index": 1, "source": "600519 2022年报 · ...", "chunk_id": "..."},
    {"index": 2, "source": "600519 2023年报 · 第十节 · 第56页", "chunk_id": "..."}
  ]
}
```

#### `POST /query/debug` — 调试接口（返回检索中间结果）

请求体与 `/query` 相同，响应额外给出三路线索，可视化页面用它展示"某条结果是被哪一路召回的"：

```json
{
  "question": "贵州茅台2023年营业收入是多少",
  "vec_results":    [{"rank": 1, "vec_score": 0.71,  "source": "...", "content_preview": "..."}],
  "bm25_results":   [{"rank": 1, "bm25_score": 12.3, "source": "...", "content_preview": "..."}],
  "rrf_results":    [{"rank": 1, "rrf_score": 0.032, "vec_rank": 3, "bm25_rank": 1, "source": "...", "content_preview": "..."}],
  "context_chunks": [{"index": 1, "source": "...", "content": "..."}],
  "answer": "...",
  "citations": [{"index": 1, "source": "...", "chunk_id": "..."}]
}
```

向量、BM25、RRF 三路各返回 top-5；`context_chunks` 是最终送进 LLM 的上下文（默认 top-4，hierarchical 策略下为父块全文）。字段定义见 `src/serve.py:98`。

#### `GET /health` — 健康检查

```json
{"status": "ok", "pipeline_ready": true}
```

#### `GET /docs` — Swagger 交互文档

浏览器访问 `http://localhost:8000/docs`，可直接在页面上填写参数发起请求，无需额外工具。

---

### 5.3 调用示例

**curl**：
```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "贵州茅台2023年营业收入是多少"}'

# 带过滤条件
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "营业收入是多少", "stock_code": "600519", "year": "2023"}'
```

**Python requests**：
```python
import requests

resp = requests.post(
    "http://localhost:8000/query",
    json={"question": "宁德时代2021年研发费用是多少"},
)
data = resp.json()
print(data["answer"])
for c in data["citations"]:
    print(f"  [{c['index']}] {c['source']}")
```

**JavaScript fetch**（前端对接）：
```javascript
const resp = await fetch("http://localhost:8000/query", {
  method: "POST",
  headers: {"Content-Type": "application/json"},
  body: JSON.stringify({question: "贵州茅台2023年营业收入是多少"}),
});
const data = await resp.json();
console.log(data.answer);
```

---

## 五、评估流程

### 5.1 RAGAS 自动评估

```bash
# 评估原生版（约需 30 分钟，每题多次 LLM 调用）
python evaluation/evaluate.py

# 只跑几道题（调试用）
python evaluation/evaluate.py --question-ids 1,2,3

# 跳过 RAGAS 打分，只看答案
python evaluation/evaluate.py --skip-ragas
```

**输出示例**：
```
RAGAS 评估结果 - native
==================================================
  Faithfulness（忠实度）:       0.8921
  Answer Relevancy（答案相关性）: 0.8734
  Context Precision（上下文精确率）:0.7812
  Context Recall（上下文召回率）:  0.8103
```

结果保存到 `evaluation/results/native_20260416_210000.json` 和对应 CSV。

### 5.2 消融实验

需要先分别构建三套向量索引（修改 `chunk_documents.py` 中的 `STRATEGY` 变量分别运行三次 chunk_documents + build_index）：

```
data/chunks/all_fixed.json          → vectorstore/faiss_fixed/
data/chunks/all_semantic.json       → vectorstore/faiss_semantic/
data/chunks/all_hierarchical.json   → vectorstore/faiss_hierarchical/
```

然后运行：

```bash
# 默认：只测 semantic 策略，两种检索方式
python evaluation/compare_strategies.py

# 指定策略和检索方式
python evaluation/compare_strategies.py \
    --strategies semantic,hierarchical \
    --modes vector_only,bm25_only,hybrid

# 修改 top-k
python evaluation/compare_strategies.py --top-k 4
```

**输出示例**：
```
======================================================================
消融实验结果汇总（Top-4）
======================================================================
分块策略          检索方式          Hit Rate        MRR   题数
----------------------------------------------------------------------
semantic         hybrid           0.900       0.833      9
hierarchical     hybrid           0.889       0.822      9
semantic         vector_only      0.778       0.722      9
semantic         bm25_only        0.667       0.611      9
======================================================================
```

---

## 六、调试与常见问题

### Q: 答案返回"根据年报知识库未能找到与该问题相关的内容"

**排查步骤**：
1. 确认向量索引已构建（`vectorstore/faiss_index.bin` 存在且大于 1 MB）
2. 检查 `vec_score` 是否正常：在 `rag_pipeline.py` 中临时添加打印：
   ```python
   logger.info(f"top vec_score = {vec_results[0]['vec_score']:.3f}")
   ```
   正常值应在 0.5～0.85 之间
3. 阈值过高：当前 `SCORE_THRESHOLD = 0.25`，可临时改为 0.1 测试

### Q: DashScope API 报错 `batch size is invalid`

text-embedding-v3 单批上限为 **10 条**（不是文档宣传的 25）。确认 `build_index.py` 中 `BATCH_SIZE = 10`。

### Q: 原生版 Rerank 触发了 bge-reranker-base 下载

Rerank 需要约 1.1 GB 的本地模型。如不需要可：
- 命令行加 `--no-rerank`
- 或将模型下载到 `models/bge-reranker-base/` 后本地加载（下次不再下载）

### Q: 如何切换 LLM 模型

修改 `src/rag_pipeline.py` 顶部：
```python
LLM_MODEL = "qwen-plus"    # 可换 "qwen-max"（更强）/ "qwen-turbo"（更快更便宜）
```
接口兼容 OpenAI 格式，也可换为 DeepSeek：
```python
DASHSCOPE_URL = "https://api.deepseek.com"
LLM_MODEL = "deepseek-chat"
```

---

## 七、运行时资源消耗参考

| 步骤 | 耗时 | 费用（DashScope） | 本地资源 |
|------|------|-----------------|---------|
| 01 下载 PDF | 2~5 min | 免费 | 85 MB 磁盘 |
| 02 解析 PDF | 3~5 min | 免费 | CPU，约 500 MB 内存 |
| 03 分块 | < 1 min | 免费 | — |
| 04 建向量索引| 5~10 min | **~0.35 元** | 约 2 GB 内存（批量向量） |
| 05 单次问答（CLI）| 2~5 sec | ~0.002 元/次 | — |
| serve.py 启动 | ~10 sec（一次性）| 免费 | 约 1.5 GB 内存（常驻）|
| HTTP 单次请求 | 2~5 sec | ~0.002 元/次 | — |
| RAGAS 评估（20题）| 30~60 min | ~5~10 元 | — |
