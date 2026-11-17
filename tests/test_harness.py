"""Unit tests for the RAG evaluation harness (stdlib unittest)."""

import json
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evaluate import load_json, run_eval
from src.answerer import extract_answer
from src.metrics import answer_correct, token_f1
from src.retrievers import BM25Retriever, TfidfRetriever, corpus_idf
from src.text_utils import normalize, split_sentences, tokenize


def load_all():
    docs = load_json("data/corpus.json")
    eval_set = load_json("data/eval_set.json")
    return docs, eval_set


class TestTextUtils(unittest.TestCase):
    def test_tokenize_strips_plurals(self):
        self.assertEqual(tokenize("Canisters and cards"), ["canister", "and", "card"])

    def test_normalize(self):
        self.assertEqual(normalize("ES-2409, Batch!"), "es 2409 batch")

    def test_split_sentences(self):
        self.assertEqual(len(split_sentences("One. Two? Three!")), 3)


class TestRetrievers(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.docs, _ = load_all()
        cls.bm25 = BM25Retriever(cls.docs)
        cls.tfidf = TfidfRetriever(cls.docs)

    def test_bm25_finds_recall_notice(self):
        top = [d for d, _ in self.bm25.search("EmberStove batch recall", k=3)]
        self.assertIn("recall-2025", top)

    def test_tfidf_finds_gift_cards(self):
        top = [d for d, _ in self.tfidf.search("do gift cards expire", k=1)]
        self.assertEqual(top, ["gift-cards"])

    def test_scores_sorted_descending(self):
        for retriever in (self.bm25, self.tfidf):
            ranked = retriever.search("warranty tent", k=5)
            scores = [s for _, s in ranked]
            self.assertEqual(scores, sorted(scores, reverse=True))
            self.assertEqual(len(ranked), 5)

    def test_unknown_query_returns_k_results(self):
        ranked = self.bm25.search("zzzzqqqq unknown terms", k=3)
        self.assertEqual(len(ranked), 3)


class TestAnswerer(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.docs, _ = load_all()
        cls.idf = corpus_idf(cls.docs)
        cls.by_id = {d["id"]: d["text"] for d in cls.docs}

    def test_extracts_weight_sentence(self):
        sent, _ = extract_answer(
            "How much does the RidgeTent 2P weigh?",
            self.by_id["tent-spec"], self.idf)
        self.assertIn("1.9 kg", sent)

    def test_answer_is_verbatim_quote(self):
        sent, _ = extract_answer(
            "What is the free shipping threshold?",
            self.by_id["shipping-policy"], self.idf)
        self.assertIn(sent, self.by_id["shipping-policy"])


class TestMetrics(unittest.TestCase):
    def test_token_f1_identical(self):
        self.assertAlmostEqual(token_f1("a b c", "a b c"), 1.0)

    def test_token_f1_partial(self):
        self.assertAlmostEqual(token_f1("a b c d", "a b"), 2 / 3, places=4)

    def test_answer_correct_substring(self):
        self.assertTrue(answer_correct(
            "The tent weighs 1.9 kg in total.", "1.9 kg"))

    def test_answer_correct_rejects_unrelated(self):
        self.assertFalse(answer_correct(
            "The jacket is blue and waterproof.", "1.9 kg"))


class TestEvalSetIntegrity(unittest.TestCase):
    def test_every_relevant_doc_exists(self):
        docs, eval_set = load_all()
        ids = {d["id"] for d in docs}
        for item in eval_set:
            self.assertTrue(item["relevant_docs"])
            for doc_id in item["relevant_docs"]:
                self.assertIn(doc_id, ids)
            self.assertTrue(item["question"].strip())
            self.assertTrue(item["answer"].strip())

    def test_corpus_ids_unique(self):
        docs, _ = load_all()
        ids = [d["id"] for d in docs]
        self.assertEqual(len(ids), len(set(ids)))


class TestEndToEnd(unittest.TestCase):
    def test_eval_is_deterministic(self):
        docs, eval_set = load_all()
        idf = corpus_idf(docs)
        first = run_eval(BM25Retriever(docs), docs, eval_set, 3, idf)
        second = run_eval(BM25Retriever(docs), docs, eval_set, 3, idf)
        self.assertEqual(first, second)

    def test_eval_metrics_in_range(self):
        docs, eval_set = load_all()
        idf = corpus_idf(docs)
        for retriever in (BM25Retriever(docs), TfidfRetriever(docs)):
            result = run_eval(retriever, docs, eval_set, 3, idf)
            for name, value in result["metrics"].items():
                if name != "n_questions":
                    self.assertGreaterEqual(value, 0.0)
                    self.assertLessEqual(value, 1.0)
            self.assertEqual(result["metrics"]["n_questions"], len(eval_set))


if __name__ == "__main__":
    unittest.main()
