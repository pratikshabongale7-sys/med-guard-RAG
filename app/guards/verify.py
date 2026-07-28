"""Faithfulness check - is a claim entailed by the citation/evidence - two swappable backends
Faithfulness Layer 2: build SHORT focused premises from evidence sentences (single + top-k concat)."""
# Layer 2 (premises + NLI): verify each claim against single sentences (single-hop) and
#                    short concats (multi-hop); take the max entailment score.

import re

from app.config import settings
from app.llm import generate

_SENTENCE = re.compile(r"(?<=[.!?])\s+")

_nli = None  # lazily loaded (model, tokenizer, entailment_index)

# backend 1: LLM Judge

JUDGE_SYSTEM = (
    "You verify whether a claim is supported by the given evidence. "
    "Answer EXACTLY ONE WORD: YES if the evidence directly supports the claim, "
    "NO if it does not or only partially does. No explanation."
)


# return float values to match with NLI outputs
def _llm_judge(claim: str, evidence: str) -> float:
    message = [
        {"role": "system", "content": JUDGE_SYSTEM},
        {"role": "user", "content": f"Evidence:\n{evidence}\n\nClaim:\n{claim}\n\nSupported?"},
    ]
    text, _ = generate(message)
    print("LLM Judge LLM call")
    return 1.0 if "YES" in (text or "").strip().upper() else 0.0


# backend 2: local NLI Judge

def _load_nli():
    # import lazily so torch isn't needed when LLM Judge is being used
    global _nli
    if _nli is None:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        tok = AutoTokenizer.from_pretrained(settings.nli_model)  # noqa
        model = AutoModelForSequenceClassification.from_pretrained(settings.nli_model)
        model.eval()
        # find which id stores entailment info
        entail_ids = next(
            i for i, label in model.config.id2label.items()
            if str(label).lower().startswith("entail")
        )
        _nli = (tok, model, entail_ids, torch)

    return _nli


def _nli_judge(claim: str, evidence: str) -> float:
    # claim_for_nli = re.sub(r"\[\d+\]", "", claim)
    from transformers import PreTrainedTokenizerBase
    tok: PreTrainedTokenizerBase
    tok, model, entail_idx, torch = _load_nli()
    # NLI convention: premise = the evidence, hypothesis = the claim
    inputs = tok(evidence, claim, return_tensors="pt", truncation=True, max_lentgh=512)
    with torch.inference_mode():
        logits = model(**inputs).logits
    probabilities = torch.softmax(logits, dim=-1)[0]
    # print("id2label:", model.config.id2label, "entail_idx:", entail_idx)
    # print("probs:", [round(float(p), 3) for p in probabilities])
    # print("premise[:80]:", evidence[:80])
    # print("hypothesis:", claim[:80])
    return float(probabilities[entail_idx])  # P(evidence entails claim)


# split the entire evidence into sentences
def to_sentences(chunks: list[dict]) -> list[str]:
    sentences = []
    for c in chunks:
        for s in _SENTENCE.split(c["payload"]["text"]):
            s = s.strip()
            if len(s) > 20:
                sentences.append(s)
    return sentences


# splits the claim into words and matches it against the evidence sentences and returns the count of shared words
def _similar_sentences(claim: str, evidence_sentences: list[str], k: int) -> list[str]:
    cw = set(claim.lower().split()) # cheap lexical overlap ranking
    scored = sorted(evidence_sentences, key=lambda s: len(cw & set(s.lower().split())), reverse=True)
    return scored[:k]

# concatenate top k premises found above
def build_premises(claim: str, sentences: list[str], k: int) -> list[str]:
    """Example: top = [s1, s2, s3] (the top-3 sentences)
    premises = list(top) => premises = [s1, s2, s3] - single hop claims
    loop: n=2 → " ".join(top[:2]) = "s1 s2" - multi hop claims
    n=3 → " ".join(top[:3]) = "s1 s2 s3"
    premises = [s1, s2, s3, "s1 s2", "s1 s2 s3"] — 5 premises total
    (a fact + its qualifier spread across sentences), and both stay short enough to avoid the long-premise dilution.
    verify_claim then scores the claim against all 5 and takes the max, so whichever premise shape actually entails the claim wins"""
    top = _similar_sentences(claim, sentences, k)
    premises = list(top)                    # each single sentence (single-hop)
    for n in range(2, k + 1):
        premises.append(" ".join(top[:n]))  # top-2, top-3 concat (multi-hop, still SHORT)
    return premises

# app interface

_BACKENDS = {"llm_judge": _llm_judge, "nli_judge": _nli_judge}


# score a claim against all evidence pieces seprataely and retain the best match
def verify_claim(claim: str, sentences: list[str]) -> dict:
    """Score claim vs each candidate premise; keep the best (max)."""
    backend = _BACKENDS[settings.verifier]
    if settings.verifier == "llm_judge":
        premise = " ".join(_similar_sentences(claim, sentences, settings.premise_top_k))
        best = backend(claim, premise or "")  # ONE call
    else:
        premises = build_premises(claim, sentences, settings.premise_top_k) or [""]
        best = max((backend(claim, p) for p in premises), default=0.0)
    return {
        "claim": claim,
        "score": round(best, 3),
        "supported": best >= settings.faithfulness_threshold
    }
