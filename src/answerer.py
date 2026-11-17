"""Extractive answerer: quote, never generate.

The answerer picks the single sentence from the top-ranked document
that best matches the question, scored by IDF-weighted overlap of
content tokens. It returns that sentence verbatim plus the cited
document id. Because the answer is always a verbatim quote from the
cited document, the system cannot hallucinate wording; the measured
"quote fidelity" metric in the report verifies this property rather
than assuming it.
"""

from .text_utils import content_tokens, split_sentences


def extract_answer(question, doc_text, idf):
    """Return (sentence, score). Empty doc text returns ('', 0.0)."""
    sentences = split_sentences(doc_text)
    if not sentences:
        return "", 0.0
    qterms = set(content_tokens(question))
    best_sentence, best_score = sentences[0], -1.0
    for sent in sentences:
        overlap = qterms & set(content_tokens(sent))
        score = sum(idf.get(t, 0.0) for t in overlap)
        if score > best_score:
            best_sentence, best_score = sent, score
    return best_sentence, best_score
