# 技术架构说明

> 上市公司年报 RAG 问答系统 — 整体方案、选型决策与设计原理

---

## 📖 本文档导读

**文档定位**：全项目的"总设计图"

## 一、项目定位

本项目以"上市公司年度报告智能问答"为场景，构建一套企业级落地标准的 RAG（检索增强生成）系统。
数据来自巨潮资讯网（证监会指定披露平台），包含 5 家公司 × 3 年 = 15 份年报 PDF，总量约 85 MB。

## 二、整体流水线

```
原始 PDF
    │
    ▼ download_reports.py
数据获取（巨潮资讯 API）
    │
    ▼ parse_pdf.py
PDF 解析（文字 + 表格 + OCR + 章节结构）
    │
    ▼ chunk_documents.py
文档分块（四种策略可切换）
    │
    ▼ build_index.py
向量化 + 索引构建（DashScope API / 本地 BGE）
    │
    ▼ rag_pipeline.py
问答流水线：查询 → 检索 → 重排 → 生成
    │
    ▼ evaluation/
评估（RAGAS 四项指标 + 消融实验）
```

---

## 三、各环节技术选型

### 3.1 数据获取

**巨潮资讯网 API**（`cninfo.com.cn`）是证监会指定的上市公司信息披露平台，数据公开合法。

关键参数：`announceType=report`，按 `searchkey`（公司名+年份）全文搜索。

选型原因：
- 原始 PDF 未经任何清洗，包含扫描件、嵌套表格、页眉页脚噪声，具有真实的工程挑战性
- 同一公司多年数据可测试**时效性冲突处理**
- 不同行业（消费品、金融、新能源、科技）覆盖多种 PDF 复杂度

---

### 3.2 PDF 解析

**组合策略**：pdfplumber（表格）+ PyMuPDF（文字+字体信息）+ pytesseract（OCR，可选）

| 库 | 职责 | 选型原因 |
|----|------|---------|
| `pdfplumber` | 表格提取 | 基于规则的表格算法，对财务报表的行列识别更准确 |
| `PyMuPDF`（`import pymupdf`） | 文字+字体元数据 | 提供每个 span 的字体大小和加粗信息，用于识别标题层级 |
| `pytesseract` | 扫描页 OCR | 部分年报附件（审计报告原件）为扫描件，需 OCR 降级处理 |

解析输出的三种块类型：
- `title`：识别依据是字体大于 14pt 或加粗且行长小于 50 字
- `table`：直接转为 Markdown 格式（方便 LLM 读取）
- `text`：正常文本段落

每个块保留完整元数据：`page_num`、`section_path`（章节路径栈）、`is_ocr`、
`heading_level`（H1/H2/H3，0=非标题）、`parser`（rule/vlm/ocr）。

**Qwen-VL 分支（可选，默认关闭）**：只处理规则解析容易坏的页——表格密集页
（≥2 张表或表格面积占比 ≥20%）与扫描页。走 DashScope OpenAI 兼容接口，
输出同样是 `title/text/table` 块，表格保留 `raw_table` 二维结构。开关由
`VLM_MODE=off|complex|all` 或 `--vlm-mode/--vlm-pages` 控制；有
`VLM_MAX_PAGES` 硬预算、按页缓存（`data/vlm_cache/`）、任何失败逐页回退规则解析。

---

### 3.3 文档分块（Chunking）

提供四种策略，通过 `--strategy` 或脚本顶部 `STRATEGY` 变量切换：

#### 策略 A：固定大小分块（`fixed`）
```
chunk_size=500字符，overlap=50字符
```
- 优点：简单可预测
- 缺点：无视句子/段落边界，表格被截断
- 用途：作为 baseline，体现"不好的分块"的代价

#### 策略 B：语义分块（`semantic`）—— 默认
```
遇标题强制切块，段落合并不超过 800 字符
```
- 遇 `title` 块：先 flush 缓冲区，标题单独成块
- 遇 `table` 块：单独成块（不与文字混合），同时写入 `metadata.table_json`
  （原始二维结构）；因此默认 semantic 索引也能做表格精确取数
- 遇 `text` 块：累积到 800 字符再切
- 效果：保留语义完整性，章节边界清晰

#### 策略 C：层级分块（`hierarchical`）
```
父块：完整章节（~1500字符）
子块：父块内的细粒度分割（~300字符）
```
- 子块用于精确向量匹配，父块内容在检索时附加给 LLM 作为扩展上下文
- 即"Small-to-Big"检索策略，缓解长文档召回粒度不匹配问题

#### 策略 D：结构化章节树（`structured`）
```
子块：段落/表格（≤800 字符），只带 parent_id
父块：整章拼接（≤2000 字符），单独写入 parents.json
```
- 表格子块 = 表格摘要 + 原始 Markdown，原始二维结构进 `metadata.table_json`
- structured 的摘要前缀带公司名/年份（`【表格摘要】五粮液2021年报 · 章节`），
  同一指标在多份年报里重复出现时更容易被问题命中；BM25 A/B 显示摘要文本会
  稀释关键词分数（Hit@4 0.72→0.68），所以该前缀**只用于 structured 子块**，
  semantic/BM25 不受影响（见 `docs/resume_gap_plan.md` §15）
- `heading_level` 优先取解析阶段显式的 H1/H2/H3，缺失时退回 `section_path` 深度
- 检索时"子块召回、父块拼装"：`rag_pipeline.resolve_chunk_content` 按
  `parent_id` 查 `parents.json`，查不到再退回内联 `parent_content` / 子块自身
- 落盘到 `vectorstore/structured/`，通过 `RAG_INDEX_DIR` 或 `--index-dir` 切换；
  **不设为默认**——100 题实测混合检索上弱于 semantic（见 `docs/resume_gap_plan.md` §11）
- 章节栈修复后（2026-11-07）重新分块：6,838 个 chunks / 4,195 个 parents；
  章节根被“目录/公司名/日期”污染的比例从 57% 降到 0.6%（指标待重测）

**实际产出**（语义分块）：7,931 个 chunks，平均 526 字符（2026-11-07 解析去重后；去重前 10,352）

---

### 3.4 Embedding 模型

#### DashScope text-embedding-v3（API）

| 参数 | 值 |
|------|----|
| 模型 | `text-embedding-v3` |
| 维度 | 1024（可选 768/512） |
| 批次上限 | **10 条**（API 硬限制） |
| 接口 | OpenAI 兼容接口 |
| 费用 | ~0.0007 元/千 token |

调用方式：
```python
client = OpenAI(
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)
resp = client.embeddings.create(
    model="text-embedding-v3",
    input=batch,          # 最多 10 条
    dimensions=1024,
)
```

---

### 3.5 向量库

**FAISS IndexFlatIP**（精确内积检索）

```
检索时间：7,931 条向量 × 1024 维 → 单次查询 < 1ms
```

选型原因：
- `IndexFlatIP`：暴力内积，精确不近似，数据量 < 10 万时速度足够
- 向量已 L2 归一化，内积等价于余弦相似度（`cos(q, d) = q·d`）
- 持久化为单文件（`faiss_index.bin`），无需服务进程

扩展路径：数据量 > 50 万时可换 `IndexIVFFlat`（需 `train`）或 `IndexHNSWFlat`（无需 train）。

**文件产出**：
- `vectorstore/faiss_index.bin`：索引，30.98 MiB（7,931 × 1024 × 4 bytes = 32,530,432，实测文件 32,485,421 字节）
- `vectorstore/faiss_meta.json`：元数据，11.52 MiB（含每条 chunk 的文本和溯源信息）

---

### 3.6 检索策略

实现三路检索并融合：

#### 向量检索（Vector）
- 查询文本 → DashScope API → 查询向量 → FAISS 内积搜索
- 优势：语义相似，理解同义词和近义表达

#### BM25 关键词检索
- 分词：`jieba` + A 股领域词表 `src/domain_dict.txt`（157 词，修正“扣非净利润”“商誉减值”等切词）
- 注意：领域词表只作用于分词/BM25，**不影响 DashScope embedding**（模型内部自行 tokenize）
- 打分：`BM25Okapi`（`rank_bm25` 库）
- 优势：精确匹配数字、股票代码、专有名词（"营业收入"、"600519"）

#### RRF 混合融合（Reciprocal Rank Fusion）
```
score(d) = Σ 1 / (k + rank_i(d))    k = 60（经验值）
```
- 将向量排名和 BM25 排名合并为统一分数
- 互补两路检索的盲区：向量检索对精确数字弱，BM25 对语义理解弱

#### CrossEncoder Rerank（可选）
- 模型：`BAAI/bge-reranker-base`（本地约 1.1 GB）
- 对 RRF 融合后的候选集（top-10→top-20）精排，输出 top-4 给 LLM
- 原理：双向 attention 对 (query, doc) 对打分，比 bi-encoder 更准确但不可向量化
- 若未下载模型则自动降级为直接截断
- **分数下限默认关闭**（`RERANK_MIN_SCORE` 未设）：100 题实测基线 RRF
  Hit@4 0.960 / MRR 0.831，重排后 0.890 / 0.735；下限 0.70 为 0.850 / 0.708，
  没有任何阈值能同时提升 Hit 和削减上下文。数字题重排 MRR 有提升
  （0.884→0.926），但多跳/幻觉题变差，因此不设为默认（§15）

#### 查询实体归一
- 题集和真实用户都可能输入 6 位股票代码，而 chunk 正文只有公司简称
  （“五粮液”），代码在正文里不出现，会人为压低召回。
- `expand_query_entities()` 在检索前把已知代码替换成 `简称+代码`
  （如 `000858` → `五粮液000858`），向量/BM25/重排都用扩展后的 query，
  送 LLM 的仍是原问题。36 道数字题实测：semantic Hit@4 0.500→1.000、
  MRR 0.317→0.847；structured 0.306→0.972、MRR 0.157→0.810。
- 题集中的 100 道代码题干已改写成自然公司名（`normalize_question_entities.py`），
  代码扩展保留为线上兜底。

#### 表格精确取数（默认 semantic + structured 索引均可用）
- 表格块在 `metadata.table_json` 保存原始二维结构
- 检索命中表格后，`lookup_table_value()` 按首列行标签直接取整行，
  不经过 LLM，避免从 Markdown 里重新数格子
- 候选集全落空时，按问题推断的 `stock_code`/`year` 在全部表格块里兜底扫描
  （报表页数字密集，向量相似度常输给审计意见等叙述页）；推断不出公司则不扫，
  防止跨公司串数
- `query()` 生成前还会做**表格行补位**：问题里点名的指标（总资产、营业收入、
  归母净资产等）若在收窄后的表格池里有精确行，就追加一条上下文补进去。
  实测 36 道数字题里，不做补位只有 12 道正常回答，补位后 36 道全部答对
  （同一批题上调过标签映射，存在过拟合风险，见 `docs/resume_gap_plan.md` §15）
- 消费端：CLI `--table-row`、HTTP `POST /table/lookup`

---

### 3.7 LLM 生成

**DashScope qwen-plus**（两套实现统一）

| 参数 | 值 |
|------|----|
| 模型 | `qwen-plus` |
| 接口 | OpenAI 兼容，`/v1/chat/completions` |
| Temperature | 0.1（低随机性，保证数字准确性） |
| 上下文窗口 | 放入 top-4 个 chunk（平均约 2000 字） |

系统提示核心约束：
1. 只从参考资料中回答，不编造数据
2. 引用具体数字时标注来源编号（如 `[1]`）
3. 参考资料不足时主动拒绝回答

**可选查询改写**（`--query-rewrite`）：
用 `qwen-turbo` 将模糊问题改写为检索友好形式，例如：
```
原始：茅台最近怎么样
改写：贵州茅台2023年营业收入净利润同比增长率经营情况
```

---

## 四、评估体系

### 4.1 评测题集（v2 125 题；v1 20 题保留旧口径）

v2 四类：`numeric` 36 / `multihop` 34 / `hallucination` 30 / `refusal` 25，
由 `data/parsed` 派生并经 `verify_question_set.py` 逐字校验。题干统一用公司
简称（100 道原代码题干已改写），避免“用户输入股票代码”这种非自然问法把
召回指标压低；代码兜底仍由 `expand_query_entities()` 提供。

v1 旧口径（ARCHITECTURE 里早期数字的来源）：

| 类型 | 题数 | 考察能力 | 示例 |
|------|------|---------|------|
| `simple_fact` | 5 | 基础检索 | 茅台2023年营业收入？ |
| `precise_number` | 5 | BM25数字召回 | 宁德时代2021研发费用占比？ |
| `cross_doc_compare` | 4 | 多文档综合 | 茅台vs五粮液2022毛利率对比 |
| `time_trend` | 3 | 跨年版本整合 | 茅台2021-2023营收趋势 |
| `should_refuse` | 3 | 幻觉控制 | 茅台股价是多少（应拒绝）|

### 4.2 RAGAS 四项指标

| 指标 | 计算方式 | 含义 |
|------|---------|------|
| **Faithfulness** | LLM 判断 | 答案是否完全来自检索内容（无幻觉） |
| **Answer Relevancy** | Embedding 相似度 | 答案是否切题 |
| **Context Precision** | LLM 判断 | 检索内容中有用比例 |
| **Context Recall** | LLM 判断 | 需要的内容被检索到了多少 |

评估中 LLM 裁判默认使用**小米 MiMo Token Plan**（`mimo-v2.6-flash` @ `https://token-plan-cn.xiaomimimo.com/v1`，需 `MIMO_API_KEY`；预设见 `evaluation/evaluate.py` 的 `JUDGE_PRESETS`），Embedding 使用 DashScope `text-embedding-v3`，两者都走 OpenAI 兼容接口，无需 OpenAI key。设 `RAGAS_LLM_PROVIDER=deepseek` 可切回原先的 `deepseek-v4-flash` 裁判 —— 本文下列指标数字最初是在 DeepSeek 裁判上得到的，换裁判等于换尺子，归因时要先确认档位。

**RAGAS 基线对比门**：`scripts/ci/check_eval_compare.sh` 用固定题号重跑评估，与
`scripts/ci/ragas_baseline.json` 比四项指标并输出逐题回退；pre-push 在改动
`parse_pdf.py` / `chunk_documents.py` / `rag_pipeline.py` / 题集时自动触发 6 题冒烟。
默认只报告不拦人（`CI_EVAL_STRICT=1` 才以回退为失败），因为 6 题上的 RAGAS 抖动大；
完整口径仍用 `python evaluation/evaluate.py --question-set v2`。

**拒答控制组**：`scripts/ci/measure_refusal.py` 把回答分成“真拒答 / 空检索 /
正常回答”三类，并额外跑一组应答题（numeric + multihop）。只测 25 道拒答题
会把“全拒”误判成 100%：拒答题 25/25 真拒答、0 空检索的同时，控制组 36 题
在表格行补位前只有 12 题正常回答，补位后 36/36 答对（开发集，存在过拟合风险）。

### 4.3 消融实验矩阵

```
分块策略（3）× 检索方式（3）= 9 种组合
  策略: fixed / semantic / hierarchical
  检索: vector_only / bm25_only / hybrid
```
> structured 走独立的数值题 A/B（`scripts/ci/measure_index_ab.py`），
> 目录布局与 `compare_strategies.py` 的 `vectorstore/faiss_{strategy}/` 约定不同，未并入此矩阵。

指标：`Hit Rate @4`（命中率）和 `MRR`（平均倒数排名），不依赖 LLM，运行快。

---

## 五、关键工程决策与踩坑

| 问题 | 根因 | 解法 |
|------|------|------|
| 巨潮 API `stock` 参数返回 0 结果 | API 已不支持股票代码过滤 | 改用 `searchkey` 全文搜索 |
| 五粮液搜索无结果 | cninfo 存储为 "五 粮 液"（字间有空格）| 拆字加空格作为 fallback 关键词 |
| DashScope embedding 报错 batch too large | text-embedding-v3 上限为 10 条，非文档宣传的 25 | `BATCH_SIZE = 10` |
| 开启 BM25 时答案始终"无法找到" | 阈值 0.25 对余弦相似度设计，RRF 分数 ~0.016 量纲完全不同 | 阈值检查始终读 `vec_score` |
| 表格内容在 table 和 text 块中重复 | `parse_pdf.py` 收集了 `table_bboxes` 但未用于过滤文字区域 | 已修：文字块按表格 bbox 过滤（43,048→39,499 blocks，−8.2%） |

---

## 六、目录结构

```
rag_annual_report/
├── src/                         # 原生版（DashScope API）
│   ├── download_reports.py      # 巨潮 API 下载 PDF
│   ├── parse_pdf.py             # PDF → 结构化 JSON blocks
│   ├── chunk_documents.py       # blocks → chunks（四种策略）
│   ├── build_index.py           # chunks → FAISS 向量索引
│   ├── rag_pipeline.py          # 问答流水线（BM25+向量+RRF+Rerank+LLM）
│   ├── serve.py                 # FastAPI HTTP 服务
│   └── static/index.html        # 检索链路可视化 Web 页面
│
├── evaluation/
│   ├── questions.json           # v1：20 题（旧口径）
│   ├── questions_v2.json        # v2：125 题四类
│   ├── evaluate.py              # RAGAS 四项指标自动评估
│   └── compare_strategies.py    # 消融实验（策略 × 检索方式）
│
├── scripts/ci/                  # 本地 CI（静态/索引/单测/服务/RAGAS 对比）
│   └── ragas_baseline.json      # RAGAS 6 题冒烟基线（题号 + 逐题分数）
│
├── requirements.txt
├── ARCHITECTURE.md              # 本文档
├── USAGE_GUIDE.md               # 代码调用与测试指南
```
