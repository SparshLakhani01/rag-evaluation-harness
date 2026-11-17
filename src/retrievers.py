"""Retrievers: BM25 (implemented from scratch) and TF-IDF (scikit-learn).

Both share one interface: ``search(query, k)`` returns a list of
``(doc_id, score)`` pairs sorted by score descending. Neither uses
embeddings: there is no transformer/embedding model in this project,
by design, so the comparison is lexical vs lexical with different
weighting. See README for what that means for the results.
"""

import math
from collections import Counter, defaultdict

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from .text_utils import tokenize


class BM25Retriever:
    """Classic BM25 (Robertson et al.), k1=1.5, b=0.75."""

    def __init__(self, docs, k1=1.5, b=0.75):
        """docs: list of dicts with 'id' and 'text' (synthetic corpus)."""
        self.k1 = k1
        self.b = b
        self.doc_ids = [d["id"] for d in docs]
        self._doc_tokens = [tokenize(d["text"]) for d in docs]
        self._doc_len = [len(t) for t in self._doc_tokens]
        self._avg_len = sum(self._doc_len) / max(1, len(self._doc_len))
        df = Counter()
        for toks in self._doc_tokens:
            for term in set(toks):
                df[term] += 1
        n = len(docs)
        self.idf = {
            term: math.log(1 + (n - freq + 0.5) / (freq + 0.5))
            for term, freq in df.items()
        }
        self._term_freqs = [Counter(t) for t in self._doc_tokens]

    def search(self, query, k=3):
        qterms = tokenize(query)
        scores = []
        for idx, tf in enumerate(self._term_freqs):
            dl = self._doc_len[idx]
            norm = 1 - self.b + self.b * dl / self._avg_len
            score = 0.0
            for term in set(qterms):
                if term not in tf:
                    continue
                f = tf[term]
                score += self.idf.get(term, 0.0) * (
                    f * (self.k1 + 1) / (f + self.k1 * norm)
                )
            scores.append((self.doc_ids[idx], score))
        scores.sort(key=lambda x: (-x[1], x[0]))
        return scores[:k]


class TfidfRetriever:
    """TF-IDF cosine retrieval via scikit-learn."""

    def __init__(self, docs):
        self.doc_ids = [d["id"] for d in docs]
        self._vectorizer = TfidfVectorizer(
            tokenizer=tokenize, lowercase=False, preprocessor=None,
            token_pattern=None,
        )
        self._matrix = self._vectorizer.fit_transform(
            [d["text"] for d in docs]
        )
        norms = np.sqrt(
            np.asarray(self._matrix.multiply(self._matrix).sum(axis=1)).ravel()
        )
        self._norms = np.where(norms == 0, 1.0, norms)

    def search(self, query, k=3):
        qvec = self._vectorizer.transform([query])
        qnorm = np.sqrt(qvec.multiply(qvec).sum())
        if qnorm == 0:
            scored = [(doc_id, 0.0) for doc_id in self.doc_ids]
        else:
            sims = np.asarray(
                (self._matrix @ qvec.T).todense()
            ).ravel() / (self._norms * qnorm)
            scored = list(zip(self.doc_ids, sims.tolist()))
        scored.sort(key=lambda x: (-x[1], x[0]))
        return scored[:k]


def corpus_idf(docs):
    """IDF per term over the corpus; shared by sentence scoring."""
    df = defaultdict(int)
    for doc in docs:
        for term in set(tokenize(doc["text"])):
            df[term] += 1
    n = len(docs)
    return {
        term: math.log(1 + (n - freq + 0.5) / (freq + 0.5))
        for term, freq in df.items()
    }
