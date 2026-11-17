"""Evaluation metrics. Definitions (also stated in the README):

- recall@k: share of eval questions with at least one ground-truth
  relevant document in the top-k retrieved documents.
- citation precision: share of questions whose cited (top-1) document
  is in the ground-truth relevant set.
- answer accuracy: share of questions whose extracted answer sentence
  either contains the normalized ground-truth answer as a substring,
  or reaches token-F1 >= 0.5 against the ground-truth answer.
- mean token-F1: average token-level F1 between predicted sentence
  and ground-truth answer, over all questions.
- quote fidelity: share of answers that appear verbatim in the cited
  document. 1.0 by construction for this extractive answerer; it is
  computed, not assumed, so a future generative answerer would show
  a real (lower) number here.
"""

from .text_utils import normalize, tokenize

F1_THRESHOLD = 0.5


def token_f1(predicted, gold):
    """Token-level F1 between two strings (multiset overlap)."""
    p_tokens = tokenize(predicted)
    g_tokens = tokenize(gold)
    if not p_tokens or not g_tokens:
        return 0.0
    p_counts, g_counts = {}, {}
    for t in p_tokens:
        p_counts[t] = p_counts.get(t, 0) + 1
    for t in g_tokens:
        g_counts[t] = g_counts.get(t, 0) + 1
    overlap = sum(
        min(p_counts[t], g_counts[t]) for t in set(p_counts) & set(g_counts)
    )
    if overlap == 0:
        return 0.0
    precision = overlap / len(p_tokens)
    recall = overlap / len(g_tokens)
    return 2 * precision * recall / (precision + recall)


def answer_correct(predicted_sentence, gold_answer):
    """Substring rule first, then token-F1 threshold."""
    if normalize(gold_answer) in normalize(predicted_sentence):
        return True
    return token_f1(predicted_sentence, gold_answer) >= F1_THRESHOLD


def summarize(per_question):
    """Aggregate per-question records into the headline metrics."""
    n = len(per_question)
    if n == 0:
        return {}
    return {
        "n_questions": n,
        "recall_at_k": sum(p["retrieved_relevant"] for p in per_question) / n,
        "citation_precision": sum(p["citation_correct"] for p in per_question) / n,
        "answer_accuracy": sum(p["answer_correct"] for p in per_question) / n,
        "mean_token_f1": sum(p["token_f1"] for p in per_question) / n,
        "quote_fidelity": sum(p["quote_verbatim"] for p in per_question) / n,
    }
