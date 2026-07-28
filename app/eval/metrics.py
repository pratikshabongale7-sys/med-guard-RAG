"""Custom metric suite for MedGuard, computed from per-item eval records.

Record shape (per gold item after a run):
  {
    "type": "answerable" | "out_of_scope",
    "abstained": bool,
    "confidence": float | None,     # supported_ratio from the gate (answerable items)
    "correct": bool | None,         # answer matched gold (answerable + answered items)
    "retrieved_pmids": [str, ...],  # for retrieval metrics
    "relevant_pmids": [str, ...],
    "claims": [{"supported": bool, "citation_ok": bool | None}, ...],
  }

Abstention is framed as SELECTIVE PREDICTION: the system trades coverage (how much it
answers) against risk (how wrong it is when it answers), plus abstention precision/recall
for the safety side. We report both costs: missing a bad answer (recall/coverage) AND
refusing a good one (precision/over-refusal).
"""


def _safe(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


# --------------------------------------------------------------------------- #
# Retrieval
# --------------------------------------------------------------------------- #
def precision_recall_at_k(retrieved_pmids: list[str], relevant_pmids: list[str], k: int):
    top = retrieved_pmids[:k]
    hit = len(set(top) & set(relevant_pmids))
    precision = _safe(hit, k)
    recall = _safe(hit, len(relevant_pmids))
    return round(precision, 3), round(recall, 3)


# --------------------------------------------------------------------------- #
# Faithfulness / citation (per-answer)
# --------------------------------------------------------------------------- #
def hallucination_rate(claims: list[dict], key: str = "supported") -> float:
    if not claims:
        return 0.0
    return round(sum(1 for c in claims if not c[key]) / len(claims), 3)


def citation_accuracy(claims: list[dict]):
    checked = [c for c in claims if c.get("citation_ok") is not None]
    return round(_safe(sum(1 for c in checked if c["citation_ok"]), len(checked)), 3) if checked else None


# --------------------------------------------------------------------------- #
# Abstention — confusion matrix + classification view (positive = should-abstain)
# --------------------------------------------------------------------------- #
def confusion(records: list[dict]) -> tuple[int, int, int, int]:
    """Return (tp, fp, fn, tn). Positive class = 'should abstain' (out_of_scope)."""
    tp = fp = fn = tn = 0
    for r in records:
        should_abstain = r["type"] == "out_of_scope"
        abstained = r["abstained"]
        if should_abstain and abstained:
            tp += 1            # correctly refused an out-of-scope question
        elif not should_abstain and abstained:
            fp += 1            # over-refusal: refused an answerable question
        elif should_abstain and not abstained:
            fn += 1            # missed refusal: answered an out-of-scope question
        else:
            tn += 1            # correctly answered an answerable question
    return tp, fp, fn, tn


def abstention_scores(records: list[dict]) -> dict:
    tp, fp, fn, tn = confusion(records)
    precision = _safe(tp, tp + fp)
    recall = _safe(tp, tp + fn)
    f1 = _safe(2 * precision * recall, precision + recall)
    specificity = _safe(tn, tn + fp)                 # correctly-answered rate on answerable
    accuracy = _safe(tp + tn, tp + fp + fn + tn)
    false_abstention_rate = _safe(fp, fp + tn)       # over-refusal among answerable questions
    return {
        "abstention_precision": round(precision, 3),
        "abstention_recall": round(recall, 3),
        "abstention_f1": round(f1, 3),
        "specificity": round(specificity, 3),
        "accuracy": round(accuracy, 3),
        "false_abstention_rate": round(false_abstention_rate, 3),
        "counts": {"tp": tp, "fp": fp, "fn": fn, "tn": tn},
    }


# --------------------------------------------------------------------------- #
# Selective prediction — coverage / risk at the operating point
# --------------------------------------------------------------------------- #
def selective_scores(records: list[dict]) -> dict:
    """Over the ANSWERABLE set at the gate's actual operating point:
    coverage = fraction answered; selective_risk = error rate among answered."""
    answerable = [r for r in records if r["type"] == "answerable"]
    answered = [r for r in answerable if not r["abstained"]]
    coverage = _safe(len(answered), len(answerable))
    correct = sum(1 for r in answered if r["correct"] is True)
    selective_risk = 1 - _safe(correct, len(answered))
    return {
        "coverage": round(coverage, 3),
        "selective_risk": round(selective_risk, 3),
        "answered": len(answered),
        "answerable": len(answerable),
    }


# --------------------------------------------------------------------------- #
# Risk-Coverage curve + AURC (threshold-independent, the gold standard)
# --------------------------------------------------------------------------- #
def risk_coverage_curve(records: list[dict]) -> dict:
    """Sweep the confidence threshold over ANSWERABLE items and trace risk vs coverage.

    Requires `confidence` and `correct` for every answerable item (answer everything,
    record correctness even for would-abstain items — i.e. run generation with the gate
    OFF, or record confidence+correctness before gating). AURC = area under the curve;
    lower is better (a good confidence score puts errors at low confidence).
    """
    items = [(r["confidence"], bool(r["correct"]))
             for r in records
             if r["type"] == "answerable"
             and r.get("confidence") is not None
             and r.get("correct") is not None]
    if not items:
        return {"points": [], "aurc": None}

    items.sort(key=lambda x: x[0], reverse=True)   # most confident answered first
    n, errors, points = len(items), 0, []
    for i, (_, correct) in enumerate(items, start=1):
        if not correct:
            errors += 1
        points.append((round(i / n, 4), round(errors / i, 4)))   # (coverage, risk)

    aurc = 0.0                                     # trapezoidal area under risk vs coverage
    for j in range(1, len(points)):
        (c0, r0), (c1, r1) = points[j - 1], points[j]
        aurc += (c1 - c0) * (r0 + r1) / 2
    return {"points": points, "aurc": round(aurc, 4)}
