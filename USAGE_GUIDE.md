# 代码调用与测试指南

> 上市公司年报 RAG 问答系统

---

## 📖 本文档导读

**文档定位**：全项目的"操作手册"——所有命令、参数、报错处理都在这里，边跑代码边查。

**使用方式**：
1. 按"一、环境准备"一次性配好依赖与 API Key（`DASHSCOPE_API_KEY`，评估还需 `MIMO_API_KEY`）；
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

跑评估（`evaluation/evaluate.py`）时，RAGAS 的 LLM 裁判默认走**小米 MiMo Token Plan**，还需额外配置：

```bash
# PowerShell
$env:MIMO_API_KEY = "tp-xxxxxxxx"

# Linux / macOS
export MIMO_API_KEY=tp-xxxxxxxx
```

MiMo 开放平台：https://platform.xiaomimimo.com/ —— 在「订阅管理 / API Keys」里建密钥，
**Key 只完整显示一次**，当场存好。Token Plan 的 OpenAI 兼容端点是
`https://token-plan-cn.xiaomimimo.com/v1`（中国集群；新加坡集群是 `token-plan-sgp.…`，
Key 与 Base URL 必须同平台，交叉搭配是最常见的 401 来源）。

Embedding 仍走 DashScope（`text-embedding-v3`），只有 LLM 裁判用 MiMo ——
MiMo 的公开资料里没有 embeddings 端点，而 `answer_relevancy` 等指标要算向量相似度。

想复现 ARCHITECTURE.md 里的旧结论（那批数字是在 DeepSeek 裁判上跑的），切回旧尺子：

```bash
$env:RAGAS_LLM_PROVIDER = "deepseek"     # 并配 DEEPSEEK_API_KEY
```

换集群或试别的模型 ID 不用改源码：`MIMO_BASE_URL` / `MIMO_MODEL` 可覆盖
（预设见 `evaluation/evaluate.py` 的 `JUDGE_PRESETS`）。

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
4. 表格转为 Markdown 格式，标题按编号规律/字体大小识别 H1/H2/H3，维护章节路径栈
5. 输出每个 PDF 对应的 JSON 文件

**可选：Qwen-VL 复杂页解析**（表格密集页 / 扫描页，默认关闭）：
```bash
python src/parse_pdf.py --vlm-mode complex          # 只把复杂页交给 qwen3-vl-plus
python src/parse_pdf.py --vlm-pages 21,22,23        # 只解析指定页（1-based）
python src/parse_pdf.py --vlm-sample 5              # 每份 PDF 抽样 5 张复杂页做验证
python src/parse_pdf.py --vlm-mode all --vlm-max-pages 10   # 全页模式 + 硬预算
```
- 需要 `DASHSCOPE_API_KEY`；没有 key 或调用失败时**逐页回退规则解析**，不会中断整份年报
- 结果缓存在 `data/vlm_cache/<pdf名>/page_XXXX.json`，重跑不重复计费
- 环境变量：`VLM_MODE=off|complex|all`、`VLM_MAX_PAGES=60`、`VLM_DPI=150`、`VLM_MODEL=qwen3-vl-plus`

实测（2026-11-07，茅台 2023 第 63/65 页）：`qwen3-vl-plus` 每页返回 6 个块，
两张利润表分别还原为 31×4、27×4 的 `raw_table`。默认模型已从不可用的
`qwen-vl-max` 改为账号实际可用的 `qwen3-vl-plus`；若你的账号只有旧模型，
用 `VLM_MODEL=` 覆盖即可。

**预期输出**：
```
data/parsed/
  600519_2023_贵州茅台_贵州茅台2023年年度报告.json   # 约 1,308 个 blocks
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
python src/chunk_documents.py                       # 默认 semantic
python src/chunk_documents.py --strategy structured # 表格摘要 + 章节树 Parent-Child
```

**控制参数**：`--strategy` 优先，未传时取脚本顶部 `STRATEGY` 变量。

**四种策略的区别**：

| 策略 | 切割依据 | chunk 大小 | 适合场景 |
|------|---------|-----------|---------|
| `fixed` | 每 500 字符截断，overlap=50 | 均匀 | Baseline 对比 |
| `semantic` | 遇标题强制切，段落合并不超 800 字 | 不均匀 | 默认，语义完整 |
| `hierarchical` | 父块（完整章节）+ 子块（细粒度）| 双层 | 长文档精确召回 |
| `structured` | 章节树 Parent-Child；表格子块=摘要+原始结构，父块单独存 | 子块≤800 / 父块≤2000 | 表格取数、上下文补全 |

**预期输出**：
```
data/chunks/all_semantic.json   # 7,931 个 chunks，文件 11.42 MiB（实测 11,973,715 字节）
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
    "heading_level": 3,
    "strategy": "semantic",
    "source_file": "600519_2023_贵州茅台_贵州茅台2023年年度报告.pdf"
  }
}
```

`structured` 策略的子块额外带 `parent_id`、`table_json`（表格原始二维结构），
父块写入 `data/chunks/*_structured_parents.json` 并在建索引时合并到
`vectorstore/structured/parents.json`。

---

### 步骤 4：构建向量索引

```bash
python src/build_index.py                                   # 主索引（semantic）
python src/build_index.py --strategy structured --out-dir vectorstore/structured
```

`--out-dir` 用于给新策略单独落盘，验证完成前不覆盖线上索引。structured 索引
会额外写 `parents.json`（父块全文），检索时按子块的 `parent_id` 查表拼装。

**内部流程**：
1. 加载 `data/chunks/all_semantic.json`（7,931 条）
2. 按批次（10 条/批，共 ~794 批）调用 DashScope text-embedding-v3
3. 对所有向量做 L2 归一化（使内积等价于余弦相似度）
4. 构建 FAISS IndexFlatIP，批量 add 归一化向量
5. 分别保存索引文件和元数据

**耗时与费用**：约 5~10 分钟；7,931 条 × 平均 50 token ≈ 40 万 token ≈ **约 0.35 元**

**预期输出**：
```
vectorstore/faiss_index.bin   # 30.98 MiB（7931 × 1024 × 4 bytes）
vectorstore/faiss_meta.json   # 11.61 MiB（每条 chunk 的文本 + 溯源元数据）
```

**进度日志**（每 100 批打印一次）：
```
2026-11-07 [INFO] Embedding 进度: 100/794 批
2026-11-07 [INFO] Embedding 进度: 200/794 批
...
2026-11-07 [INFO] 索引构建完成，共 7931 条向量
2026-11-07 [INFO] FAISS 索引已保存 → vectorstore/faiss_index.bin  (31724 KB)
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

# 切到 structured 索引 / 启用精排分数下限
python src/rag_pipeline.py --query "..." --index-dir vectorstore/structured
python src/rag_pipeline.py --query "..." --rerank-min-score 0.80

# 表格精确取数：先检索相关表格，再按首列行标签取原始二维行（不经过 LLM）
python src/rag_pipeline.py --query "贵州茅台2023年利润表" \
    --index-dir vectorstore/structured --table-row "营业收入"
```

`--table-row` 在默认 semantic 索引和 structured 索引上都可用：两种分块都会把
表格原始二维结构写进 `metadata.table_json`。它复用 `retrieve()` 的候选集
（RRF 之后、精排之前），避免表格被精排挤掉；匹配时先归一化空白与
`其中：`/`（元）` 等前后缀，并且**只匹配首列**，命中后直接打印整行及公司、
年份、章节、页码。

候选集里一行都没命中时，会按问题推断公司与年份，在**收窄后的全部表格块**里再扫
一遍（`rag_pipeline.table_pool`）。报表页以数字为主，向量相似度经常输给审计意见
这类叙述页——实测 `贵州茅台2023年利润表营业收入` 的 Top-10 里没有合并利润表，
但兜底扫描能精确命中。推断不出公司（如问题同时提到多家公司）时不扫全库，
避免跨公司串数；此时与旧行为一致，只返回候选集里的命中。

**查询实体归一**：用户输入 `000858` 时，检索前会自动扩成 `五粮液000858`
（`expand_query_entities`），因为 chunk 正文只有简称、没有代码。扩展只影响
向量/BM25/重排的 query，送 LLM 的仍是原问题。

**生成前的表格行补位**：`query()` 会检查问题里是否点名了某个财务指标
（总资产、营业收入、归母净资产等）。如果按公司/年份收窄后的表格池里有
精确同名行，就把它追加为一条上下文（最多 1 条），避免“数值就在库里、
但 RRF top-4 没排进来”导致误拒。表格行内容带行标签和列名，不会只给一串数字。

`--rerank-min-score` 默认不设（只排序+截断，不丢上下文）。旧默认 0.70 是在
25 题子集上选的，100 题实测 Hit@4 从 0.560 降到 0.410，已回退；启用前先用
`python scripts/ci/measure_rerank.py --sample 100` 重定阈值。

**内部流程**（以默认配置为例）：

```
用户输入 "贵州茅台2023年营业收入是多少"
    │
    ├─ 查询实体归一（股票代码 → 简称+代码，仅影响检索 query）
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
    ├─ 表格行补位（问题点名的指标有精确行时，追加 1 条上下文）
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
FAISS 索引加载完成，共 7931 条向量
构建 BM25 索引（分词中，请稍候）...
BM25 索引完成
Pipeline 初始化完成，开始接受请求
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:8000
```

FAISS 和 BM25 只在启动时加载一次，之后每次请求只需 2 次 DashScope API 调用。

服务启动时的可切换项（环境变量）：
```bash
RAG_INDEX_DIR=../vectorstore/structured   # 切索引目录；不设则用 vectorstore/
RERANK_MIN_SCORE=0.80                     # 精排分数下限；不设 = 不过滤（默认）
```
`/health` 会回显 `index_dir`、`strategy`、`rerank_min_score`，用于确认线上实际加载的是哪套索引。

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

#### `POST /table/lookup` — 表格精确取数

请求体：
```json
{
  "question":  "贵州茅台2023年利润表营业收入是多少",  // 用于检索相关表格
  "row_label": "营业收入",                          // 首列行标签，支持模糊包含
  "stock_code": "600519",                           // 可选
  "year":       "2023"                              // 可选
}
```

响应体：
```json
{
  "question": "贵州茅台2023年利润表营业收入是多少",
  "row_label": "营业收入",
  "matched": 1,
  "rows": [
    {
      "chunk_id": "...", "stock_code": "600519", "year": "2023",
      "page_num": 63, "section": "财务报告 > 合并利润表",
      "row": ["其中：营业收入", "147,693,604,994.14", "124,099,843,771.99"]
    }
  ]
}
```

与 `/query` 走同一个 `pipeline.retrieve()`，但取的是 **RRF 候选集**（精排前），
因为精排按语义相关性排序，可能把"表格行对但标题不显眼"的块挤掉。默认
semantic 索引与 structured 索引都保存了 `table_json`，两种索引都能取到数据；
没有表格块的索引返回 `matched: 0`。这条路径不调用 LLM，费用只有一次 embedding 检索。

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

### 5.2 RAGAS 基线对比门（CI）

`scripts/ci/check_eval_compare.sh` 把“这次跑分”和 `scripts/ci/ragas_baseline.json`
按同一批题号对比，输出四项指标 diff 与逐题回退：

```bash
# 默认：按 v2 类别分层抽 2 题/类（共 6 题），没有基线时自动建立
sh scripts/ci/check_eval_compare.sh

# 重建基线（例如主动改了 Prompt/分块，认为新结果才是新基线）
sh scripts/ci/check_eval_compare.sh --update-baseline --per-category 2

# 指标回退超过 3pp 时退出码 1（默认只报告，不拦人）
sh scripts/ci/check_eval_compare.sh --strict

# 全量 125 题（30~60 分钟，花钱档）
python evaluation/evaluate.py --question-set v2
```

pre-push 钩子只在本次推送改到 `parse_pdf.py` / `chunk_documents.py` /
`rag_pipeline.py` / 题集时触发 6 题冒烟对比；`CI_EVAL=0 git push` 可关闭，
`CI_EVAL_STRICT=1 git push` 把指标回退当失败。基线只保存题号与逐题 RAGAS 分数，
不存答案和上下文。

### 5.3 消融实验

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
| RAGAS 评估（v1 20 题）| 30~60 min | ~5~10 元 | — |
| RAGAS 评估（v2 125 题）| 按题数约为 v1 的 6 倍（未完整跑过，仅线性估算）| 同左比例 | — |
