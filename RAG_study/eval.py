from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Any, Dict, List

from rag import DeepSeekClient, SimpleRAG, load_text, split_text


@dataclass
class EvalItem:
    """单条评测样本。"""

    qid: str
    query: str
    expected_answer: str
    answer_keywords: List[str]
    evidence_chunk_ids: List[int]


def normalize_text(text: str) -> str:
    """轻量归一化：小写 + 去除空白。"""
    return "".join(text.lower().split())


def load_eval_set(path: str) -> List[EvalItem]:
    """加载 jsonl 评测集。"""
    items: List[EvalItem] = []
    for idx, raw_line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        line = raw_line.strip()
        if not line:
            continue

        data: Dict[str, Any] = json.loads(line)
        items.append(
            EvalItem(
                qid=str(data.get("id", f"q{idx}")),
                query=str(data["query"]),
                expected_answer=str(data.get("expected_answer", "")),
                answer_keywords=[str(x) for x in data.get("answer_keywords", [])],
                evidence_chunk_ids=[int(x) for x in data.get("evidence_chunk_ids", [])],
            )
        )
    return items


def first_relevant_rank(ranked_ids: List[int], evidence_ids: List[int]) -> int | None:
    """返回第一个相关片段在结果中的名次（1-based），找不到则 None。"""
    evidence_set = set(evidence_ids)
    for idx, cid in enumerate(ranked_ids, 1):
        if cid in evidence_set:
            return idx
    return None


def keyword_coverage(answer: str, keywords: List[str]) -> float:
    """关键词覆盖率：命中关键词数 / 关键词总数。"""
    if not keywords:
        return 0.0
    norm_answer = normalize_text(answer)
    hits = 0
    for kw in keywords:
        if normalize_text(kw) in norm_answer:
            hits += 1
    return hits / len(keywords)


def evaluate(
    rag: SimpleRAG,
    eval_items: List[EvalItem],
    top_k: int,
    candidate_k: int,
    with_llm: bool,
    verbose: bool,
) -> Dict[str, Any]:
    retrieval_hits: List[float] = []
    mrr_scores: List[float] = []
    keyword_coverages: List[float] = []
    keyword_full_hits: List[float] = []
    exact_contains_hits: List[float] = []
    per_item: List[Dict[str, Any]] = []

    for item in eval_items:
        # 1) 召回 + 2) 重排
        candidates = rag.retrieve(item.query, top_k=max(candidate_k, top_k))
        reranked = rag.rerank(item.query, candidates, top_k=top_k)
        ranked_ids = [chunk.chunk_id for chunk, _, _, _ in reranked]

        # 检索指标：Recall@k + MRR
        rank = first_relevant_rank(ranked_ids, item.evidence_chunk_ids)
        hit = rank is not None
        mrr = 1.0 / rank if rank is not None else 0.0

        retrieval_hits.append(1.0 if hit else 0.0)
        mrr_scores.append(mrr)

        result: Dict[str, Any] = {
            "id": item.qid,
            "query": item.query,
            "ranked_chunk_ids": ranked_ids,
            "evidence_chunk_ids": item.evidence_chunk_ids,
            "retrieval_hit_at_k": hit,
            "mrr": mrr,
        }

        # 生成指标（可选）
        if with_llm:
            generated_answer = rag._generate_with_llm(item.query, reranked)
            coverage = keyword_coverage(generated_answer, item.answer_keywords)
            full_hit = coverage == 1.0 and len(item.answer_keywords) > 0

            if item.expected_answer:
                exact_contains = (
                    normalize_text(item.expected_answer) in normalize_text(generated_answer)
                )
            else:
                exact_contains = False

            keyword_coverages.append(coverage)
            keyword_full_hits.append(1.0 if full_hit else 0.0)
            exact_contains_hits.append(1.0 if exact_contains else 0.0)

            result.update(
                {
                    "generated_answer": generated_answer,
                    "keyword_coverage": coverage,
                    "keyword_full_hit": full_hit,
                    "expected_answer_included": exact_contains,
                }
            )

        per_item.append(result)

        if verbose:
            print(f"[{item.qid}] {item.query}")
            print(f"  ranked_chunk_ids={ranked_ids}")
            print(f"  evidence_chunk_ids={item.evidence_chunk_ids}")
            print(f"  retrieval_hit_at_k={hit}, mrr={mrr:.4f}")
            if with_llm:
                print(
                    "  "
                    f"keyword_coverage={result['keyword_coverage']:.4f}, "
                    f"keyword_full_hit={result['keyword_full_hit']}, "
                    f"expected_answer_included={result['expected_answer_included']}"
                )
            print("-" * 80)

    summary: Dict[str, Any] = {
        "total_questions": len(eval_items),
        "top_k": top_k,
        "candidate_k": candidate_k,
        "retrieval_recall_at_k": mean(retrieval_hits) if retrieval_hits else 0.0,
        "retrieval_mrr": mean(mrr_scores) if mrr_scores else 0.0,
    }

    if with_llm:
        summary.update(
            {
                "answer_keyword_coverage": mean(keyword_coverages) if keyword_coverages else 0.0,
                "answer_keyword_full_hit_rate": mean(keyword_full_hits)
                if keyword_full_hits
                else 0.0,
                "answer_expected_contains_rate": mean(exact_contains_hits)
                if exact_contains_hits
                else 0.0,
            }
        )

    return {"summary": summary, "details": per_item}


def print_summary(report: Dict[str, Any], with_llm: bool) -> None:
    """打印汇总结果。"""
    summary = report["summary"]
    print("\n=== 评估汇总 ===")
    print(f"样本数: {summary['total_questions']}")
    print(f"top_k: {summary['top_k']}, candidate_k: {summary['candidate_k']}")
    print(f"Recall@k: {summary['retrieval_recall_at_k']:.4f}")
    print(f"MRR: {summary['retrieval_mrr']:.4f}")
    if with_llm:
        print(f"关键词覆盖率: {summary['answer_keyword_coverage']:.4f}")
        print(f"关键词全命中率: {summary['answer_keyword_full_hit_rate']:.4f}")
        print(f"期望答案包含率: {summary['answer_expected_contains_rate']:.4f}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="评估 RAG 检索/重排效果（可选评估生成效果）。"
    )
    parser.add_argument("--eval-set", default="eval_set.jsonl", help="jsonl 评测集路径")
    parser.add_argument("--story", default="story.txt", help="原始文档路径")
    parser.add_argument("--chunk-size", type=int, default=120, help="切块大小")
    parser.add_argument("--overlap", type=int, default=30, help="切块重叠大小")
    parser.add_argument("--top-k", type=int, default=3, help="最终输出的 top-k")
    parser.add_argument("--candidate-k", type=int, default=10, help="召回候选数")
    parser.add_argument(
        "--with-llm",
        action="store_true",
        help="是否调用 DeepSeek 评估生成答案（需要 DEEPSEEK_API_KEY）",
    )
    parser.add_argument("--verbose", action="store_true", help="打印每条样本细节")
    parser.add_argument("--save-json", default="", help="可选：把评估结果保存为 json 文件")
    args = parser.parse_args()

    text = load_text(args.story)
    chunks = split_text(text, chunk_size=args.chunk_size, overlap=args.overlap)
    llm_client = DeepSeekClient() if args.with_llm else None
    rag = SimpleRAG(chunks, llm_client=llm_client)
    eval_items = load_eval_set(args.eval_set)

    report = evaluate(
        rag=rag,
        eval_items=eval_items,
        top_k=args.top_k,
        candidate_k=args.candidate_k,
        with_llm=args.with_llm,
        verbose=args.verbose,
    )
    print_summary(report, with_llm=args.with_llm)

    if args.save_json:
        Path(args.save_json).write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"\n已保存评估详情到: {args.save_json}")


if __name__ == "__main__":
    main()
