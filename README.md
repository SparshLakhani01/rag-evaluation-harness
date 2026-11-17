# RAG Evaluation Harness

A small, fully deterministic harness that measures retrieval-augmented
generation quality end to end: retrieve, extract an answer, and score
it against ground truth. It compares two lexical retrievers (BM25,
implemented from scratch, and TF-IDF via scikit-learn) over the same
synthetic corpus and the same 18-question eval set, so differences in
the report are differences between retrievers, not differences in data.

The answerer is extractive only: it quotes one sentence verbatim from
the top-ranked document. It never generates text, so it cannot
hallucinate wording. The harness verifies that property (quote
fidelity) instead of assuming it.

## Data disclosure — read this first

**All data in this project is synthetic.** The corpus is 24 short
documents about a fictional company, "Summitline Outdoors", hand-written
for this project. The products (RidgeTent 2P, TrailPack 45, EmberStove,
StormShell jacket), prices, policies, and the recall notice are all
invented. There are no real customers, no real products, and no real
support tickets. The eval set is 18 hand-written questions with
ground-truth answers and relevant-document labels. Every report the
harness writes is labeled `synthetic_data: true`.

## What it does

Given a question, the pipeline:

1. Retrieves the top-k documents with the selected retriever.
2. Picks the single sentence from the top-1 document with the highest
   IDF-weighted content-token overlap with the question.
3. Cites that document and returns the sentence verbatim as the answer.

The harness scores every eval question and reports:

- **recall@k**: share of questions with at least one ground-truth
  relevant document in the top-k retrieved documents.
- **citation precision**: share of questions whose cited (top-1)
  document is in the ground-truth relevant set.
- **answer accuracy**: share of questions whose extracted sentence
  either contains the normalized ground-truth answer as a substring,
  or reaches token-F1 >= 0.5 against it.
- **mean token-F1**: average token-level F1 between the predicted
  sentence and the ground-truth answer. This is structurally low
  (see below) and is reported for completeness, not as the headline
  correctness signal.
- **quote fidelity**: share of answers appearing verbatim in the
  cited document. 1.0 by construction for this extractive answerer;
  it is computed, not assumed, so a future generative answerer would
  show a real, lower number.

## Measured results

Measured by running `./demo.sh` on a 24-document synthetic corpus and
an 18-question synthetic eval set, k=3. Two consecutive runs produced
byte-identical reports (the pipeline has no randomness).

| metric | BM25 | TF-IDF | delta (BM25 - TF-IDF) |
|---|---|---|---|
| recall@3 | 0.944 (17/18) | 1.000 (18/18) | -0.056 |
| citation precision | 0.833 (15/18) | 0.833 (15/18) | 0.000 |
| answer accuracy | 0.778 (14/18) | 0.778 (14/18) | 0.000 |
| mean token-F1 | 0.321 | 0.321 | 0.000 |
| quote fidelity | 1.000 (18/18) | 1.000 (18/18) | 0.000 |

Read these numbers as what they are: scores on one 18-question
synthetic eval set, not general claims about BM25 or TF-IDF. On this
set the two retrievers differ on exactly one question (recall@3).

### Where it fails, and why

The same 4 questions fail under both retrievers, which points at a
shared limitation rather than a retriever difference:

- **q01, q03, q12** are morphological gaps: the question uses
  "weigh", "pitch", or "washed" while the documents use "weighing",
  "pitches", or "wash". The tokenizer strips trailing "s" only, so
  these forms never match, and a same-product document with more
  overlapping terms wins instead. A stemmer or embedding retriever
  would close most of this gap; this project deliberately uses
  neither, so the failure is visible and measurable.
- **q04** is a sentence-selection error: the correct document is
  retrieved and cited, but the answerer picks the warranty's first
  sentence instead of the exclusions sentence. The extractor scores
  sentences by question-term overlap only and has no notion of what
  kind of sentence answers an "exclude" question.

Mean token-F1 (0.321) is low because predicted answers are full
sentences while gold answers are short spans ("1.9 kg"). The
substring-or-F1 accuracy rule is the headline correctness signal;
token-F1 is reported so the gap is visible, not hidden.

## How to run

```bash
./demo.sh            # creates .venv, installs deps, runs eval + tests
```

Or manually:

```bash
python3 -m venv --system-site-packages .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python evaluate.py [--k 3] [--output results/report.json]
.venv/bin/python -m unittest discover -s tests
```

`evaluate.py` prints the summary table and writes the full
per-question report to `results/report.json`.

## Project layout

```
data/corpus.json        24 synthetic support documents
data/eval_set.json      18 questions + ground-truth answers + relevant docs
src/text_utils.py       tokenizer, normalization, sentence splitting
src/retrievers.py       BM25 (from scratch) + TF-IDF (scikit-learn)
src/answerer.py         extractive sentence selection (IDF-weighted overlap)
src/metrics.py          recall@k, citation precision, accuracy, token-F1
evaluate.py             CLI: run eval, print table, write JSON report
tests/test_harness.py   17 unit tests (stdlib unittest)
```

## Limitations (honest)

- No embeddings, no language model, no reranker. Retrieval is
  lexical only, so paraphrases and morphological variants fail
  (see the failure analysis above). This is a property of the
  design, not an oversight, but it bounds what the scores mean.
- The corpus is tiny (24 docs) and the eval set is small
  (18 questions). One question is worth 5.6 points of accuracy.
  Do not extrapolate these numbers to any real corpus.
- The answerer returns whole sentences, not short spans, and can
  only answer questions whose answer sits in a single sentence of
  the top-1 document.
- Ground-truth labels were written by the same person who wrote the
  corpus, so the eval set is not an independent benchmark.
- No query logging, no latency measurement, no production serving:
  this is an offline evaluation tool.

## Future improvements

- Add a stemming tokenizer option and measure how much of the
  q01/q03/q12 gap it closes.
- Add an embedding retriever (e.g. a local sentence-transformer
  model) as a third arm of the comparison, with the same eval set.
- Add a reranking stage and measure citation precision with and
  without it.
- Add short-span answer extraction and compare span-F1 against
  sentence-F1.
- Grow the eval set with independently written questions and
  adversarial paraphrases.
