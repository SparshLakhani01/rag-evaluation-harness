"""Shared text utilities: tokenization, normalization, sentence splitting.

All data processed by this package is synthetic (see README). The
tokenizer is deliberately small and deterministic: lowercase, keep
alphanumeric runs, strip a trailing "s" from tokens longer than 3
characters so singular/plural forms match ("canister"/"canisters").
No stemming library, no external models.
"""

import re

STOPWORDS = {
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "do", "does", "did", "what", "which", "who", "whom", "whose", "when",
    "where", "why", "how", "can", "could", "should", "would", "will",
    "shall", "may", "might", "i", "you", "he", "she", "it", "we", "they",
    "my", "your", "his", "her", "its", "our", "their", "of", "to", "in",
    "on", "at", "by", "for", "with", "from", "as", "and", "or", "but",
    "if", "then", "than", "so", "such", "no", "not", "too", "very",
    "just", "about", "into", "over", "under", "up", "down", "out",
}

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def tokenize(text):
    """Lowercase alphanumeric tokens with light plural stripping."""
    tokens = _TOKEN_RE.findall(text.lower())
    out = []
    for tok in tokens:
        if len(tok) > 3 and tok.endswith("s"):
            tok = tok[:-1]
        out.append(tok)
    return out


def content_tokens(text):
    """Tokens with stopwords removed (used for sentence scoring)."""
    return [t for t in tokenize(text) if t not in STOPWORDS]


def normalize(text):
    """Canonical form used for substring answer matching."""
    return " ".join(tokenize(text))


def split_sentences(text):
    """Split on sentence-ending punctuation followed by whitespace."""
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return [p.strip() for p in parts if p.strip()]
