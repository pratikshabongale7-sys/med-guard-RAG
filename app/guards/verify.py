"""Faithfulness check - is a claim entailed by the citation/evidence - twi swappable backends"""
from app.config import settings
from app.llm import generate

_nli = None # lazily loaded (model, tokenizer, entailment_index)

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

    return 1.0 if "YES" in (text or "").strip().upper() else 0.0

# backend 2: local NLI Judge

def _load_nli():
    # import lazily so torch isn't needed when LLM Judge is being used
    global _nli
    if _nli is None:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        tok = AutoTokenizer.from_pretrained(settings.nli_model)   # noqa
        model = AutoModelForSequenceClassification.from_pretrained(settings.nli_model)
        model.eval()
        #find which id stores entailment info
        entail_ids = next(
            i for i, label in model.config.id2label.items()
            if str(label).lower().startswith("entail")
        )
        _nli = (tok, model, entail_ids, torch)

    return _nli

def _nli_judge(claim: str, evidence: str) -> float:
    from transformers import PreTrainedTokenizerBase
    tok: PreTrainedTokenizerBase
    tok, model, entail_idx, torch = _load_nli()
    # NLI convention: premise = the evidence, hypothesis = the claim
    inputs = tok(evidence, claim, return_tensors="pt", truncation=True, max_lentgh=512)
    with torch.inference_mode():
        logits = model(**inputs).logits
    probabilities = torch.softmax(logits, dim=-1)[0]
    print("id2label:", model.config.id2label, "entail_idx:", entail_idx)
    print("probs:", [round(float(p), 3) for p in probabilities])
    print("premise[:80]:", evidence[:80])
    print("hypothesis:", claim[:80])
    return float(probabilities[entail_idx])  # P(evidence entails claim)

# app interface

_BACKENDS = {"llm_judge": _llm_judge, "nli_judge": _nli_judge}

# score a claim against each evidence piece seprataely and retain the best match
def verify_claim(claim: str, evidence_texts: list[str]) -> dict:
    backend = _BACKENDS[settings.verifier]
    scores = [backend(claim, evidence) for evidence in evidence_texts] or [0.0]
    best = max(scores)

    return {
        "claim": claim,
        "score": round(best, 3),
        "supported": best >= settings.faithfulness_threshold,
    }
