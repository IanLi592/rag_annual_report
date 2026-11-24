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

QUESTION_SETS = {
    # v1：20 题，ARCHITECTURE.md 里那批数字的口径
    # v2：125 题，四类（numeric/multihop/hallucination/refusal），由 build_question_set.py
    #      派生并经 verify_question_set.py 逐字校验；换裁判/换题集都会改变尺子
    "v1": "questions.json",
    "v2": "questions_v2.json",
}


def resolve_question_set(which: str | None) -> tuple[str, Path]:
    name = (which or os.getenv("RAGAS_QUESTION_SET") or "v2").strip().lower()
    if name not in QUESTION_SETS:
        raise ValueError(f"未知题集 {name!r}，可选：{', '.join(QUESTION_SETS)}")
    return name, EVAL_DIR / QUESTION_SETS[name]


def load_questions(question_ids: list[int] = None, which: str | None = None) -> list[dict]:
    name, qpath = resolve_question_set(which)
    if not qpath.exists():
        raise FileNotFoundError(
            f"题集 {name} 不存在: {qpath.name}\n"
            f"  先生成：python evaluation/build_question_set.py")
    with open(qpath, encoding="utf-8") as f:
        data = json.load(f)
    questions = data["questions"]
    logger.info(f"题集 {name}（{qpath.name}）共 {len(questions)} 题")
    if question_ids:
        questions = [q for q in questions if q["id"] in question_ids]
    return questions


# ── 原生版 RAG 调用 ──────────────────────────────────────────────────────────

def ragas_contexts(retrieved: list[dict], parents: dict) -> list[str]:
    """把每条检索结果拼成「来源标签 + 正文」，与 build_context 喂给作答模型的口径一致。

    旧实现直接取裸正文（[r["content"] ...]），丢掉了 chunk 的 stock_code / 年份 /
    章节这些只存在于来源标签里的信息 —— 于是当参考答案（ground_truth）要求模型
    指出「这个数属于哪家公司」时，RAGAS 裁判看到的上下文里根本没有那家公司的标识，
    忠实度(Faithfulness)与上下文召回率(Context Recall)被系统性低估。这里复用同一条
    build_context 链路（label 取自其 citations 的 source，正文取自 resolve_chunk_content），
    使「模型看到的」与「裁判看到的」严格一致。逐条独立，不改动检索到的块集合。
    """
    from rag_pipeline import build_context, resolve_chunk_content
    _, cits = build_context(retrieved, parents)
    return [f"{c['source']}\n{resolve_chunk_content(item, parents)}"
            for item, c in zip(retrieved, cits)]


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
                "contexts":     ragas_contexts(result["retrieved"], pipeline.parents),
                "ground_truth": q["ground_truth"],
                "question_id":  q["id"],
                "question_type":q["type"],
                "gen_error":    None,          # 显式标记：生成成功
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
                # 失败必须是个结构化的事实，不能只藏在答案字符串里：
                # 旧实现只有 'ERROR: …' 这个约定，下游照常拿去打分（见下面的守卫）。
                "gen_error":    f"{type(e).__name__}: {e}",
            })

    return results


# ── 生成阶段完整性守卫 ────────────────────────────────────────────────────────
# 实测踩坑（2026-11-16）：DashScope 账户欠费后每题生成都抛异常，旧实现把
# 'ERROR: …' 当答案继续往下走 —— RAGAS 照样给这些空答案打分（白烧 1,464 次裁判
# 调用），产物照常写盘、退出码 0，调用方完全看不出这 96 题是废的；全批失败时
# ragas 内部还抛 IndexError 崩在最后。
# 规矩：生成失败的题不进打分（那是花钱的一步），失败率超阈值直接中止。
# 拒答题「正常检索为空」不算失败 —— 只认 gen_error 这个显式标记。

DEFAULT_MAX_GEN_FAILURE_RATE = 0.1      # 125 题里容忍 12 题偶发；25 题批次容忍 2 题


class GenerationIntegrityError(RuntimeError):
    """生成阶段失败率超过容忍上限。"""


def generation_failures(results: list[dict]) -> list[int]:
    """生成阶段就失败的题号（这些行的分数不可信）。"""
    return [r["question_id"] for r in results if r.get("gen_error")]


def enforce_generation_integrity(
        results: list[dict],
        max_failure_rate: float = DEFAULT_MAX_GEN_FAILURE_RATE) -> list[dict]:
    """返回可以送去打分的行；失败率超阈值时抛 GenerationIntegrityError。

    `max_failure_rate=1.0` 是逃生阀（明知配额有问题还想看部分结果时用），
    `0.0` 是零容忍（canary 单题探测就该用它）。
    """
    if not results:
        raise GenerationIntegrityError("生成阶段一条结果都没有，没有任何东西可以打分")

    failed = generation_failures(results)
    rate   = len(failed) / len(results)
    if rate > max_failure_rate:
        shown = ", ".join(map(str, failed[:10])) + ("…" if len(failed) > 10 else "")
        raise GenerationIntegrityError(
            f"生成失败 {len(failed)}/{len(results)} = {rate:.0%}，超过容忍上限 "
            f"{max_failure_rate:.0%}（题号 {shown}）。已中止：不进 RAGAS 打分、不落盘。"
            f"先查 API 配额/账户状态，再重跑。")
    if failed:
        logger.warning(f"  生成失败 {len(failed)} 题，已从打分中排除：{failed}")
    return [r for r in results if not r.get("gen_error")]


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
    """按题型统计；拒答单独区分“真拒答 / 空检索 / 正常回答”。

    旧实现只用 `无法回答` 等关键词，且不区分 n_ctx=0 —— 检索失败会被算成
    模型拒答。这里复用 src/refusal.py 的同一套判定，避免两份口径。
    """
    from collections import defaultdict

    try:
        from refusal import classify_refusal
    except ImportError:  # src 还没进 sys.path（直接跑本函数时）
        sys.path.insert(0, str(BASE_DIR / "src"))
        from refusal import classify_refusal

    type_stats = defaultdict(list)
    for r in results:
        qtype  = r.get("question_type", "unknown")
        answer = r.get("answer", "")
        n_ctx  = len(r.get("contexts") or [])
        type_stats[qtype].append({
            "category": classify_refusal(answer, n_ctx),
            "answer_len": len(answer),
        })

    print("\n── 按题型统计 ──")
    for qtype, stats in type_stats.items():
        n = len(stats)
        refused = sum(1 for s in stats if s["category"] == "refused_with_context")
        empty = sum(1 for s in stats if s["category"] == "empty_context")
        answered = sum(1 for s in stats if s["category"] == "answered")
        judgeable = n - empty
        avg_len = sum(s["answer_len"] for s in stats) / max(n, 1)
        print(
            f"  {qtype:<25} 题数={n}  "
            f"真拒答={refused} 空检索={empty} 回答={answered}  "
            f"可判定拒答率={(refused/judgeable if judgeable else 0):.0%}  "
            f"平均长度={avg_len:.0f}字")


# ── 保存结果 ──────────────────────────────────────────────────────────────────

def save_results(raw_results: list[dict], scores: dict, pipeline_name: str,
                 failed_ids: list[int] | None = None):
    import pandas as pd
    from datetime import datetime

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path  = RESULT_DIR / f"{pipeline_name}_{timestamp}.json"

    output = {
        "pipeline":    pipeline_name,
        "timestamp":   timestamp,
        "scores":      scores,
        # 哪些题在生成阶段就废了：没有这个字段，一份「96 题是 ERROR 串」的产物
        # 和一份正常产物长得一模一样，下游只能靠文件大小猜。
        "generation_failures": list(failed_ids or []),
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
    parser.add_argument("--question-set", default=None, choices=sorted(QUESTION_SETS),
                        help="题集：v2=125 题四类（默认，可用 RAGAS_QUESTION_SET 覆盖）；"
                             "v1=20 题（ARCHITECTURE.md 旧结论口径）")
    parser.add_argument("--max-gen-failure-rate", type=float,
                        default=DEFAULT_MAX_GEN_FAILURE_RATE,
                        help="生成阶段失败率上限，超过就中止（不打分、不落盘）。"
                             "默认 %(default)s；0=零容忍（canary 用），1=逃生阀")
    args = parser.parse_args()
    if not 0.0 <= args.max_gen_failure_rate <= 1.0:
        parser.error("--max-gen-failure-rate 必须在 0~1 之间")

    q_ids = [int(x) for x in args.question_ids.split(",")] if args.question_ids else None
    try:
        questions = load_questions(q_ids, args.question_set)
    except (FileNotFoundError, ValueError) as e:
        logger.error(str(e))
        sys.exit(2)
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

    # 守卫必须卡在「花钱打分」之前，而且 --skip-ragas 的 canary 也要过一遍
    # （canary 的用法就是：跑 1 题 + --skip-ragas + --max-gen-failure-rate 0，
    #   答案无效就以退出码 2 停在这里，一毛钱裁判调用都不会发生）。
    try:
        scorable = enforce_generation_integrity(native_results,
                                                args.max_gen_failure_rate)
    except GenerationIntegrityError as e:
        logger.error(str(e))
        sys.exit(2)
    failed_ids = generation_failures(native_results)

    if not args.skip_ragas:
        native_scores = run_ragas_eval(scorable, "native")
        save_results(native_results, native_scores, "native", failed_ids)


if __name__ == "__main__":
    main()
