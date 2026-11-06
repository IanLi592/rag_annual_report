"""
RAGAS 自动化评估脚本

对原生 RAG 进行标准化评测，
输出 4 项 RAGAS 核心指标：
  - Faithfulness（忠实度）：答案是否完全来自检索内容，无幻觉
  - Answer Relevancy（答案相关性）：答案是否回答了问题
  - Context Precision（上下文精确率）：检索到的内容有多少是有用的
  - Context Recall（上下文召回率）：有用内容有多少被检索到了

使用方式：
  python evaluate.py

依赖：
  pip install ragas datasets openai
  export DASHSCOPE_API_KEY="sk-xxx"

注意：
  RAGAS 本身也需要调用 LLM 来打分（Faithfulness 和 Answer Relevancy）
  默认使用 OpenAI GPT-4，这里改为 DashScope qwen-plus
  RAGAS 评估一次约消耗 20 道题 × 每题多次 LLM 调用，注意 token 成本
"""

import os
import sys
import json
import argparse
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

BASE_DIR   = Path(__file__).parent.parent
EVAL_DIR   = Path(__file__).parent
RESULT_DIR = EVAL_DIR / "results"
RESULT_DIR.mkdir(exist_ok=True)

# ── RAGAS 裁判 LLM 的可切换配置 ───────────────────────────────────────────────
# 默认走小米 MiMo Token Plan（OpenAI 兼容端点）。Embedding 不受这里影响，
# 恒走 DashScope text-embedding-v3 —— MiMo 公开资料里没有 embeddings 端点，
# 而 answer_relevancy / context_* 有相当一部分要靠 embeddings 算相似度。
#
# 切换：set RAGAS_LLM_PROVIDER=deepseek（大小写不敏感）
# 保留 DeepSeek 的意义：ARCHITECTURE.md 里那批旧结论是在 DeepSeek 裁判上跑的，
# 换裁判等于换尺子，得留一条能一键回到旧口径的路径来归因差异。
JUDGE_PRESETS = {
    "mimo": {
        "base_url": "https://token-plan-cn.xiaomimimo.com/v1",
        "model":    "mimo-v2.6-flash",
        "key_env":  "MIMO_API_KEY",
    },
    "deepseek": {
        "base_url": "https://api.deepseek.com",
        "model":    "deepseek-v4-flash",
        "key_env":  "DEEPSEEK_API_KEY",
    },
}


def resolve_judge() -> dict:
    """取裁判配置。环境变量可逐项覆盖，便于换集群/试模型 ID 而不改代码。"""
    name = (os.getenv("RAGAS_LLM_PROVIDER") or "mimo").strip().lower()
    if name not in JUDGE_PRESETS:
        raise ValueError(
            f"未知的 RAGAS_LLM_PROVIDER={name!r}，可选：{', '.join(sorted(JUDGE_PRESETS))}"
        )
    preset = JUDGE_PRESETS[name]
    cfg = {
        "name":     name,
        "key_env":  preset["key_env"],
        "base_url": os.getenv(f"{name.upper()}_BASE_URL") or preset["base_url"],
        "model":    os.getenv(f"{name.upper()}_MODEL")    or preset["model"],
    }
    key = os.getenv(cfg["key_env"], "")
    if not key or key in ("sk-xxxxxxxx", "tp-xxxxxxxx"):
        raise EnvironmentError(
            f"裁判 LLM 走 {name}，但环境变量 {cfg['key_env']} 没填（或还是占位值）。\n"
            f"  .env 里加一行 {cfg['key_env']}=你的key（该文件已被 .gitignore 忽略）\n"
            f"  或临时切回旧尺子：set RAGAS_LLM_PROVIDER=deepseek"
        )
    return cfg


DASHSCOPE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
EMBED_MODEL   = "text-embedding-v3"


# ── 加载评测题集 ──────────────────────────────────────────────────────────────

def load_questions(question_ids: list[int] = None) -> list[dict]:
    qpath = EVAL_DIR / "questions.json"
    with open(qpath, encoding="utf-8") as f:
        data = json.load(f)
    questions = data["questions"]
    if question_ids:
        questions = [q for q in questions if q["id"] in question_ids]
    return questions


# ── 原生版 RAG 调用 ──────────────────────────────────────────────────────────

def run_native_rag(questions: list[dict]) -> list[dict]:
    """
    调用 src/rag_pipeline.py 中的 RAGPipeline 跑所有测试题。
    返回 RAGAS 所需格式：[{question, answer, contexts, ground_truth}]
    """
    sys.path.insert(0, str(BASE_DIR / "src"))
    from rag_pipeline import RAGPipeline   # 直接 import 模块

    logger.info("初始化原生 RAG Pipeline...")
    pipeline = RAGPipeline(use_bm25=True, use_rerank=False)  # 评估时关闭 Rerank 加快速度

    results = []
    for q in questions:
        logger.info(f"[{q['id']:02d}] {q['question'][:40]}...")
        try:
            result = pipeline.query(q["question"], verbose=False)
            results.append({
                "question":     q["question"],
                "answer":       result["answer"],
                "contexts":     [r["content"] for r in result["retrieved"]],
                "ground_truth": q["ground_truth"],
                "question_id":  q["id"],
                "question_type":q["type"],
            })
        except Exception as e:
            logger.error(f"  题目 {q['id']} 失败: {e}")
            results.append({
                "question":     q["question"],
                "answer":       f"ERROR: {e}",
                "contexts":     [],
                "ground_truth": q["ground_truth"],
                "question_id":  q["id"],
                "question_type":q["type"],
            })

    return results


# ── RAGAS 评估 ────────────────────────────────────────────────────────────────

def run_ragas_eval(results: list[dict], pipeline_name: str) -> dict:
    """
    用 RAGAS 框架打分。
    RAGAS 会用 LLM 评判 Faithfulness 和 Answer Relevancy，
    用 Embedding 评判 Context Precision 和 Context Recall。
    这里将两者都改为 DashScope，避免需要 OpenAI key。
    """
    from ragas import evaluate
    from ragas.metrics import (
        faithfulness,
        answer_relevancy,
        context_precision,
        context_recall,
    )
    from ragas.llms import LangchainLLMWrapper
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from langchain_openai import ChatOpenAI, OpenAIEmbeddings
    from datasets import Dataset

    logger.info(f"开始 RAGAS 评估（{pipeline_name}）...")

    # 裁判 LLM 可切换（默认 MiMo Token Plan）；Embedding 恒走 DashScope
    judge = resolve_judge()
    logger.info(f"RAGAS 裁判 LLM：{judge['name']} / {judge['model']} @ {judge['base_url']}")

    ragas_llm = LangchainLLMWrapper(ChatOpenAI(
        model=judge["model"],
        openai_api_key=os.getenv(judge["key_env"]),
        openai_api_base=judge["base_url"],
        temperature=0,
    ))
    ragas_embeddings = LangchainEmbeddingsWrapper(OpenAIEmbeddings(
        model=EMBED_MODEL,
        openai_api_key=os.getenv("DASHSCOPE_API_KEY"),
        openai_api_base=DASHSCOPE_URL,
        dimensions=1024,
        # DashScope 的 OpenAI 兼容接口只接受字符串，不接受 token id 列表；
        # 关闭长度检查后才会把原始字符串传给接口，否则报 400
        # "contents is neither str nor list of str"
        check_embedding_ctx_length=False,
    ))

    for metric in [faithfulness, answer_relevancy, context_precision, context_recall]:
        metric.llm       = ragas_llm
        metric.embeddings = ragas_embeddings

    # 过滤掉 should_refuse 类型（这类题的 ground_truth 是描述性的，不适合指标评估）
    eval_results = [r for r in results if r.get("question_type") != "should_refuse"]

    dataset = Dataset.from_list([
        {
            "question":     r["question"],
            "answer":       r["answer"],
            "contexts":     r["contexts"],
            "ground_truth": r["ground_truth"],
        }
        for r in eval_results
    ])

    score = evaluate(
        dataset,
        metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
    )

    # ragas 0.4.x 的 evaluate 返回 EvaluationResult，
    # score[metric] 是「每题一个分」的 list，需要取均值才是整体指标
    import numpy as np
    def _mean(metric_name: str) -> float:
        vals = [v for v in score[metric_name] if v is not None and not (isinstance(v, float) and np.isnan(v))]
        return float(np.mean(vals)) if vals else float("nan")

    faithfulness_v     = _mean("faithfulness")
    answer_relevancy_v = _mean("answer_relevancy")
    context_precision_v = _mean("context_precision")
    context_recall_v   = _mean("context_recall")

    logger.info(f"\n{'='*50}")
    logger.info(f"RAGAS 评估结果 - {pipeline_name}")
    logger.info(f"{'='*50}")
    logger.info(f"  Faithfulness（忠实度）:       {faithfulness_v:.4f}")
    logger.info(f"  Answer Relevancy（答案相关性）: {answer_relevancy_v:.4f}")
    logger.info(f"  Context Precision（上下文精确率）:{context_precision_v:.4f}")
    logger.info(f"  Context Recall（上下文召回率）:  {context_recall_v:.4f}")

    # ── 逐题分数落盘 ────────────────────────────────────────────────────────
    # 只有均值就没法回答"precision 被哪几题拖低的"。按 dataset 行序 zip 回原 dict，
    # 依赖 eval_results 与 results 共享同一批对象（它是 results 的过滤视图，不是拷贝）。
    metric_names = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]
    for r in results:
        r.setdefault("ragas", {})          # 被排除的 should_refuse 也留个空位，便于下游判断
    for m in metric_names:
        for r, v in zip(eval_results, score[m]):
            r["ragas"][m] = None if (v is None or (isinstance(v, float) and np.isnan(v))) else float(v)

    return {
        "pipeline":           pipeline_name,
        "faithfulness":       faithfulness_v,
        "answer_relevancy":   answer_relevancy_v,
        "context_precision":  context_precision_v,
        "context_recall":     context_recall_v,
    }


# ── 按题型分析 ────────────────────────────────────────────────────────────────

def analyze_by_type(results: list[dict]):
    """统计不同题型的基本表现（简单版：看有多少题拒绝回答了 should_refuse）。"""
    from collections import defaultdict

    type_stats = defaultdict(list)
    for r in results:
        qtype  = r.get("question_type", "unknown")
        answer = r.get("answer", "")
        # 简单判断：是否含有拒绝关键词
        refused = any(kw in answer for kw in ["无法回答", "无法提供", "不在", "不包含", "超出"])
        type_stats[qtype].append({"refused": refused, "answer_len": len(answer)})

    print("\n── 按题型统计 ──")
    for qtype, stats in type_stats.items():
        refuse_rate = sum(1 for s in stats if s["refused"]) / len(stats)
        avg_len     = sum(s["answer_len"] for s in stats) / len(stats)
        print(f"  {qtype:<25} 题数={len(stats)}  拒绝率={refuse_rate:.0%}  平均回答长度={avg_len:.0f}字")


# ── 保存结果 ──────────────────────────────────────────────────────────────────

def save_results(raw_results: list[dict], scores: dict, pipeline_name: str):
    import pandas as pd
    from datetime import datetime

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path  = RESULT_DIR / f"{pipeline_name}_{timestamp}.json"

    output = {
        "pipeline":    pipeline_name,
        "timestamp":   timestamp,
        "scores":      scores,
        "raw_results": raw_results,
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    logger.info(f"详细结果已保存 → {out_path}")

    # 也生成 CSV 方便查看
    df = pd.DataFrame([{
        "id":     r["question_id"],
        "type":   r["question_type"],
        "question": r["question"][:30] + "...",
        "answer_len": len(r["answer"]),
        "context_count": len(r["contexts"]),
        # 逐题指标直接进 CSV，用 Excel 排一遍就能看出拖后腿的是哪类题
        **{m: r.get("ragas", {}).get(m) for m in
           ("faithfulness", "answer_relevancy", "context_precision", "context_recall")},
    } for r in raw_results])
    csv_path = RESULT_DIR / f"{pipeline_name}_{timestamp}.csv"
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    logger.info(f"CSV 摘要已保存 → {csv_path}")


# ── 入口 ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="RAG 系统 RAGAS 评估")
    parser.add_argument("--question-ids", type=str, default=None, help="指定题号，逗号分隔，如 1,2,3")
    parser.add_argument("--skip-ragas",   action="store_true", help="跳过 RAGAS 打分，只打印答案")
    args = parser.parse_args()

    q_ids = [int(x) for x in args.question_ids.split(",")] if args.question_ids else None
    questions = load_questions(q_ids)
    logger.info(f"加载 {len(questions)} 道测试题")

    native_scores  = None

    # 预检裁判配置：先验 key/供应商，别等 20 题的答案全生成完（那一步才花钱）
    # 之后，才在打分环节报"缺 key"。实测全量一次 292 s，白花不起。
    if not args.skip_ragas:
        try:
            _judge = resolve_judge()
        except (ValueError, OSError) as e:
            logger.error(f"裁判预检失败：{e}")
            sys.exit(2)
        logger.info(f"裁判预检通过：{_judge['name']} / {_judge['model']} @ {_judge['base_url']}")

    native_results = run_native_rag(questions)
    analyze_by_type(native_results)
    if not args.skip_ragas:
        native_scores = run_ragas_eval(native_results, "native")
        save_results(native_results, native_scores, "native")


if __name__ == "__main__":
    main()
