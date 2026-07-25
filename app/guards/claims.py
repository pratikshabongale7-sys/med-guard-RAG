"""Splitting the retrieved answer into sentences and checking for entailment for each of the sentences against the
citations where the entire text of the citation chunk is the premise and the sentences are the hypotheses"""
import re

_SENTENCE_END = re.compile(r"(?<=[.!?])\s+")
_CITATION = re.compile(r"\[(\d+)\]")
_HAS_SIGNAL = re.compile(r"\[\d+\]|\d")   # a citation or any digit

# split claims based on length and relevance signal
def split_claims(answer: str, min_len: int = 20) -> list[str]:
    parts = _SENTENCE_END.split((answer or "").strip())
    claims = []
    for part in parts:
        part = part.strip(" -*•")
        if not part:
            continue
        if len(part) >= min_len and _HAS_SIGNAL.search(part):
            claims.append(part)

    return claims

# return citations in a claim
def cited_indices(claim: str) -> list[int]:
    return [int(n) for n in _CITATION.findall(claim)]

