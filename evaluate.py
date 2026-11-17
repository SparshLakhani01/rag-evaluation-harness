#!/usr/bin/env python3
"""Run the RAG evaluation harness over the synthetic corpus.

Compares BM25 and TF-IDF retrieval end to end: retrieve, extract an
answer sentence from the top document, score against ground truth.
Prints a summary table and writes a JSON report. Fully deterministic:
no randomness anywhere, so repeated runs produce identical reports.

Usage:
    python evaluate.py [--k 3] [--output results/report.json]
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.answerer import extract_answer
from src.metrics import answer_correct, summarize, token_f1
from src.retrievers import BM25Retriever, TfidfRetriever, corpus_idf

BASE = os.path.dirname(os.path.abspath(__file__))


def load_json(relpath):
    with open(os.path.join(BASE, relpath), encoding="utf-8") as fh:
        return json.load(fh)


def run_eval(retriever, docs, eval_set, k, idf):
    by_id = {d["id"]: d["text"] for d in docs}
    per_question = []
    for item in eval_set:
        ranked = retriever.search(item["question"], k=k)
        ranked_ids = [doc_id for doc_id, _ in ranked]
        cited_id = ranked_ids[0] if ranked_ids else None
        sentence, _ = (
            extract_answer(item["question"], by_id[cited_id], idf)
            if cited_id
            else ("", 0.0)
        )
        relevant = set(item["relevant_docs"])
        per_question.append(
            {
                "id": item["id"],
                "question": item["question"],
                "gold_answer": item["answer"],
                "predicted_answer": sentence,
                "cited_doc": cited_id,
                "retrieved_top_k": ranked_ids,
                "retrieved_relevant": bool(relevant & set(ranked_ids)),
                "citation_correct": cited_id in relevant,
                "answer_correct": answer_correct(sentence, item["answer"]),
                "token_f1": round(token_f1(sentence, item["answer"]), 4),
                "quote_verbatim": sentence in by_id[cited_id]
                if cited_id
                else False,
            }
        )
    return {"metrics": summarize(per_question), "per_question": per_question}


def main():
    parser = argparse.ArgumentParser(description="RAG evaluation harness")
    parser.add_argument("--k", type=int, default=3,
                        help="retrieval depth for recall@k (default: 3)")
    parser.add_argument("--output", default="results/report.json",
                        help="where to write the JSON report")
    args = parser.parse_args()

    docs = load_json("data/corpus.json")
    eval_set = load_json("data/eval_set.json")
    idf = corpus_idf(docs)

    retrievers = {
        "bm25": BM25Retriever(docs),
        "tfidf": TfidfRetriever(docs),
    }
    results = {}
    for name, retriever in retrievers.items():
        results[name] = run_eval(retriever, docs, eval_set, args.k, idf)

    report = {
        "synthetic_data": True,
        "dataset": "Summitline Outdoors support corpus (synthetic, "
                   "hand-written for this project)",
        "corpus_docs": len(docs),
        "eval_questions": len(eval_set),
        "k": args.k,
        "retrievers": results,
        "comparison": {
            metric: round(
                results["bm25"]["metrics"][metric]
                - results["tfidf"]["metrics"][metric],
                4,
            )
            for metric in results["bm25"]["metrics"]
            if metric != "n_questions"
        },
    }

    out_path = os.path.join(BASE, args.output)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)

    print(f"Corpus: {len(docs)} synthetic docs | "
          f"Eval set: {len(eval_set)} questions | k={args.k}")
    print(f"{'metric':<20}{'bm25':>8}{'tfidf':>8}{'delta':>8}")
    for metric in results["bm25"]["metrics"]:
        if metric == "n_questions":
            continue
        print(
            f"{metric:<20}"
            f"{results['bm25']['metrics'][metric]:>8.3f}"
            f"{results['tfidf']['metrics'][metric]:>8.3f}"
            f"{report['comparison'][metric]:>+8.3f}"
        )
    print(f"\nReport written to {args.output}")


if __name__ == "__main__":
    main()
