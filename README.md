# 上市公司年报 RAG 问答系统

> 以巨潮资讯网的 15 份 A 股年报 PDF 为语料，跑通「下载 → 解析 → 分块 → 向量化 → 检索 → 重排 → 生成 → 评估」全链路，并配套 125 题四类评测集（另保留 20 题旧口径）与消融实验。

这是一个**按企业级落地标准实现的正式 RAG 项目**，语料刻意保持在最小验证规模：四种分块策略、三路检索、可开关的 Rerank 与查询改写都能用一行命令现场对比，评估脚本直接产出 RAGAS 四项指标与消融结果。小规模只为了分钟级可复现（建索引约 0.35 元、5~10 分钟），各环节的实现与取舍仍按生产标准处理。

## 实测规模

下表数字来自当前仓库产物实测，不是设计估算：

| 环节 | 产物 | 实测 |
|------|------|------|
| 语料 | `data/raw_pdf/` | 15 份 PDF（5 家公司 × 2021–2023），84.3 MiB |
| 解析 | `data/parsed/` | 15 个 JSON，共 39,032 个 blocks（表格去重 + 页脚/落款噪声过滤后，2026-11-10 重解析） |
| 分块 | `data/chunks/all_semantic.json` | 7,874 个 chunks，平均 528 字符，18.23 MiB（2026-11-16 重解析后重分块） |
| 索引 | `vectorstore/faiss_index.bin` | 7,874 × 1024 维，30.76 MiB（`faiss_meta.json` 17.53 MiB） |
| 评测 | `evaluation/questions.json` / `questions_v2.json` | v1 20 题（旧口径）；v2 125 题：numeric 36 / multihop 34 / hallucination 30 / refusal 25 |

## 快速开始

```bash
# 1. 依赖
pip install -r requirements.txt

# 2. API Key（PowerShell；或写进项目根 .env，本地 CI 会自动载入）
$env:DASHSCOPE_API_KEY = "sk-xxxxxxxx"     # 必需：Embedding + LLM 生成
$env:MIMO_API_KEY      = "tp-xxxxxxxx"     # 仅评估时需要：RAGAS 的 LLM 裁判（小米 MiMo Token Plan）
                                           # 想复现旧结论：$env:RAGAS_LLM_PROVIDER = "deepseek" 并配 DEEPSEEK_API_KEY

# 3. 依次跑五步（本机已产出 data/ 与 vectorstore/，本地可直接从第 5 步开始）
python src/download_reports.py      # 巨潮 API 下载 PDF
python src/parse_pdf.py             # PDF → 结构化 blocks（默认规则解析；--vlm-mode complex 启用 Qwen-VL）
python src/chunk_documents.py       # blocks → chunks（默认 semantic 策略）
python src/build_index.py           # chunks → FAISS 索引（约 0.35 元）
python src/rag_pipeline.py --query "贵州茅台2023年营业收入是多少"

# 可选：构建 structured 索引（表格摘要 + 章节树 Parent-Child），单独落盘不覆盖主索引
python src/chunk_documents.py --strategy structured && \
python src/build_index.py --strategy structured --out-dir vectorstore/structured

# 4. 起 Web 服务（自带检索链路可视化页面）
cd src && uvicorn serve:app --host 0.0.0.0 --port 8000
# 浏览器打开 http://localhost:8000
# 切到 structured 索引：RAG_INDEX_DIR=../vectorstore/structured
# 开启重排分数下限（默认关闭，需先按 100 题重定阈值）：RERANK_MIN_SCORE=0.xx
```

常用开关：`--stock 600519 --year 2023` 限定检索范围，`--query-rewrite` 开启查询改写，`--no-bm25` / `--no-rerank` 做消融对比，`--index-dir vectorstore/structured` 切索引，`--table-row "营业收入"` 按行标签精确取表格原始行（默认 semantic 与 structured 索引都支持）。参数含义见 `src/rag_pipeline.py:834`。

## 本地 CI

```bash
sh scripts/ci/run_all.sh --tier p0p1   # 静态 + 索引 + 单测 + 服务门
sh scripts/ci/run_all.sh --tier eval   # RAGAS 基线对比（默认 6 题冒烟，花钱档）
```

pre-commit 跑静态门；pre-push 跑索引门与单测门。若本次推送改到
`parse_pdf.py` / `chunk_documents.py` / `rag_pipeline.py` / 题集，会自动追加一次
RAGAS 6 题对比，输出四项指标 diff 与逐题回退（`CI_EVAL=0` 可关闭）。
基线在 `scripts/ci/ragas_baseline.json`，只存题号与逐题分数。

## 技术选型一览

| 环节 | 选型 |
|------|------|
| PDF 解析 | pdfplumber（表格）+ PyMuPDF（文字与字体层级）+ pytesseract（扫描件 OCR，可选）+ Qwen-VL 版式解析（可选，只走复杂页） |
| Embedding | DashScope `text-embedding-v3`，1024 维，单批上限 10 条 |
| 向量检索 | FAISS `IndexFlatIP`（向量已 L2 归一化，内积 = 余弦） |
| 关键词检索 | jieba 分词 + A 股领域词表（`src/domain_dict.txt`，只影响 BM25，不影响 DashScope embedding） + `BM25Okapi` |
| 混合融合 | RRF（k=60） |
| 精排 | CrossEncoder `BAAI/bge-reranker-base`（本地 1.1 GB，缺模型自动降级为截断） |
| 生成 | DashScope `qwen-plus`，temperature 0.1 |
| 服务 | FastAPI + uvicorn |
| 评估 | RAGAS 四项指标（LLM 裁判走小米 MiMo Token Plan，可切回 DeepSeek）+ 消融实验 Hit Rate@4 / MRR |

## 项目结构

```
rag_annual_report/
├── src/
│   ├── download_reports.py     # 巨潮 API 下载（含"五 粮 液"拆字 fallback）
│   ├── parse_pdf.py            # PDF → title/table/text 三类块 + 章节路径（H1/H2/H3）
│   ├── parse_pdf_vlm.py        # Qwen-VL 复杂页解析分支（默认关闭，按页缓存）
│   ├── chunk_documents.py      # 四种分块策略（fixed/semantic/hierarchical/structured）
│   ├── build_index.py          # 批量 Embedding + FAISS 索引落盘（可 --out-dir 分策略存）
│   ├── rag_pipeline.py         # 检索→重排→生成，可 import 亦可 CLI
│   ├── refusal.py              # 拒答判定：真拒答 / 空检索 / 正常回答
│   ├── serve.py                # FastAPI：/query、/table/lookup、/query/debug、/health
│   └── static/index.html       # 展示检索中间结果的可视化页面
├── evaluation/
│   ├── questions.json          # v1：20 题（旧口径退路）
│   ├── questions_v2.json       # v2：125 题四类（numeric/multihop/hallucination/refusal）
│   ├── evaluate.py             # RAGAS 评估
│   └── compare_strategies.py   # 分块策略 × 检索方式消融
├── data/          # PDF / parsed / chunks —— .gitignore，需自建
├── vectorstore/   # FAISS 索引        —— .gitignore，需自建
├── models/        # bge-reranker-base —— .gitignore，可选
├── requirements.txt
├── ARCHITECTURE.md
└── USAGE_GUIDE.md
```

## 该看哪份文档

想先弄明白「为什么这么设计」，读 **[ARCHITECTURE.md](ARCHITECTURE.md)**：整体流水线、各环节选型理由、五类关键工程踩坑（含已知 Bug）。

想直接上手「怎么跑、参数怎么调、报错怎么办」，读 **[USAGE_GUIDE.md](USAGE_GUIDE.md)**：步骤级命令、CLI 与 HTTP 接口、代码内调用方式、常见问题、资源消耗参考。

## 上手前值得知道的三件事

1. `data/`、`vectorstore/`、`models/` 都被 `.gitignore` 排除。新克隆需自己跑步骤 1–4，建索引约 5–10 分钟、约 0.35 元。
2. Rerank 默认开启，但**分数下限默认关闭**（`RERANK_MIN_SCORE` 未设置时不丢上下文），本地没有 1.1 GB 的 `models/bge-reranker-base/` 时会自动降级为直接截断，不影响其他环节。OCR 依赖系统级 `tesseract-ocr`，未安装则扫描页留占位符。Qwen-VL 分支默认 `VLM_MODE=off`，只在 `--vlm-mode complex` 或 `--vlm-pages` 时调用，且有 `VLM_MAX_PAGES` 硬预算与按页缓存。
3. ~~表格内容在 `table` 与 `text` 块重复~~ **已修（2026-11-07）**：`parse_pdf.py` 原先收集了 `table_bboxes` 却从未在文字提取里使用，摘要页 77% 的文字行被重复发射；现在表格区域只保留 table 块那一份，块数 43,048 → 39,893；同日再过滤 `62 / 143` 这类页脚页码，最终 39,499。**2026-11-10 又补了两处漏**：页脚尾缀「…2021 年度报告全文」与带空格的落款「2022 年4 月29 日」原先同时绕过正文噪声过滤（`NOISE_PATTERNS`）和章节栈过滤（`is_non_section_title`），重解析后块数 39,499 → 39,032，chunk 里脏 `section` 320 → 0、正文含页脚行的 chunk 383 → 0。**11-16 重解析后复核**：`（续）`跑马灯已进噪声过滤（全库实测 72 块全部整行只有「（续）」二字，加过滤零误伤）；`2021年` 年度表头单元格**仍不处理**——92,932 处命中全部在表格单元格内部，过滤等于动所有财报表的数值，风险远大于收益。
