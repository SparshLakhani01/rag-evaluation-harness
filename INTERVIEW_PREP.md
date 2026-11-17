# Interview Prep — RAG Evaluation Harness

## 2-minute explanation

"I built a small evaluation harness that measures RAG quality end to
end instead of eyeballing answers. It runs a fixed 18-question eval
set over a 24-document synthetic support corpus, twice: once with
BM25 retrieval that I implemented from scratch, and once with TF-IDF
from scikit-learn. For each question it retrieves the top documents,
extracts one sentence from the top document, and scores recall at 3,
citation precision, and answer accuracy against ground-truth labels.
Everything is deterministic, so two runs produce byte-identical
reports, and the comparison between retrievers is apples to apples.
On this synthetic set TF-IDF edges out BM25 on recall, 18 out of 18
versus 17 out of 18, and both land at 14 out of 18 on answer
accuracy. The interesting part is the failure analysis: the same
four questions fail under both retrievers, and three of them are
morphological gaps, the question says 'weigh' and the document says
'weighing', which lexical retrieval cannot bridge. That is the kind
of failure an eval harness exists to surface."

## Architecture walkthrough

- `data/corpus.json` / `data/eval_set.json`: the synthetic fixtures.
  Corpus docs have ids, titles, and text. Eval items carry the
  question, a short gold answer, and the ids of relevant docs.
- `src/text_utils.py`: one tokenizer used everywhere (lowercase,
  alphanumeric runs, trailing-"s" stripping), plus normalization for
  substring matching and sentence splitting. Sharing the tokenizer
  between retrieval, extraction, and scoring avoids mismatched
  normalization bugs.
- `src/retrievers.py`: `BM25Retriever` computes IDF and BM25 scores
  directly (k1=1.5, b=0.75, standard Robertson form). `TfidfRetriever`
  wraps scikit-learn's `TfidfVectorizer` with the shared tokenizer
  and cosine similarity. Both expose `search(query, k)`.
- `src/answerer.py`: extractive sentence selection. Sentences in the
  top document are scored by IDF-weighted overlap of content tokens
  with the question; the best sentence is returned verbatim with its
  document id as the citation.
- `src/metrics.py`: recall@k, citation precision, answer accuracy
  (substring OR token-F1 >= 0.5), mean token-F1, and quote fidelity
  (is the answer verbatim in the cited doc?).
- `evaluate.py`: CLI that runs both retrievers, prints the comparison
  table, and writes `results/report.json` with per-question records.
- `tests/test_harness.py`: 17 stdlib-unittest tests covering
  tokenization, retriever ranking, extraction, metric edge cases,
  eval-set integrity, and end-to-end determinism.

## Design tradeoffs

- **Extractive, not generative.** Quoting a sentence guarantees the
  answer's wording comes from the corpus, which makes quote fidelity
  measurable and hallucination impossible by construction. The cost
  is that multi-sentence and synthesized answers are out of reach.
- **Lexical retrieval only.** No embeddings keeps the project
  dependency-light and fully offline, and makes the comparison clean.
  The cost is visible in the results: paraphrase and morphology gaps
  that embeddings would mostly close.
- **Substring-or-F1 accuracy rule.** A pure substring rule would miss
  correct answers worded differently from the gold span; pure token-F1
  with a threshold would be noisy on short gold answers. Combining
  them is a pragmatic middle ground, and both the rule and the raw
  token-F1 are reported so the choice is inspectable.
- **Shared tokenizer everywhere.** One normalization path means a
  tokenization bug affects all stages consistently rather than
  producing subtle retrieval-vs-scoring mismatches.
- **Deterministic by design.** No sampling, no randomness, fixed
  tie-breaking (score desc, then doc id). Reproducibility is a
  feature of an eval tool, not a nice-to-have.

## Limitations (say these before being asked)

- 24 docs and 18 questions is a toy scale; one question is 5.6
  accuracy points. The numbers describe this set only.
- I wrote both the corpus and the eval labels, so there is no
  independent ground truth.
- Sentence-level answers only; no multi-hop, no aggregation, no
  "I don't know" behavior when nothing relevant is retrieved.
- No latency, cost, or robustness measurement; no production path.

## Future improvements

- Stemming tokenizer option, measured against the current gap.
- A third retrieval arm using local embeddings, same eval set.
- Reranking stage; measure citation precision with/without it.
- Short-span extraction and span-F1.
- Larger, independently written eval set including paraphrases and
  unanswerable questions with an abstention metric.

## Likely interview Q&A

**Q: Why compare BM25 and TF-IDF instead of using embeddings?**
A: The point of the harness is controlled comparison, and lexical
methods run anywhere with no model download. The honest answer is
that embeddings would likely beat both on this kind of paraphrase
gap; the harness is built so adding a third arm is a small change,
and the current results document exactly where lexical retrieval
fails.

**Q: Why is mean token-F1 only 0.32 when accuracy is 0.78?**
A: Predicted answers are full sentences and gold answers are short
spans, so token-F1 is structurally depressed even when the sentence
contains the right answer. That is why accuracy uses a
substring-or-F1 rule, and why I report both numbers instead of
picking the flattering one.

**Q: How do you know the harness itself is correct?**
A: Unit tests cover the metric functions on hand-computed cases,
eval-set integrity is tested (every relevant doc exists), and the
end-to-end run is deterministic: two runs produce identical reports.
The per-question records in the JSON report make every score
auditable back to a specific question and citation.

**Q: What would you do about the morphology failures?**
A: In order: a stemming tokenizer (cheap, measurable), then
character n-gram features or an embedding retriever, each added as
a new arm of the same harness so the improvement is measured on the
same eval set rather than asserted.

**Q: Why is quote fidelity 1.0 — isn't that trivial?**
A: Yes, for this answerer it is trivially true by construction,
because it only quotes. I compute it anyway because the metric is
part of the harness's contract: if someone swaps in a generative
answerer, quote fidelity becomes a real, load-bearing measurement
instead of a silent assumption.
