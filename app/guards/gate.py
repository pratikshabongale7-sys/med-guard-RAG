"""Final aggregated answer or abstain decision over all claims against their evidences"""
from app.config import settings


def evaluate_answer(verdicts: list[dict]) -> dict:
    if not verdicts:
        return {"supported_ratio": 0.0, "passed": False}

    supported = sum(1 for v in verdicts if v["supported"])
    ratio = supported / len(verdicts)

    return {
        "supported_ratio": round(ratio, 3),
        "passed": ratio >= settings.min_supported_ratio
    }