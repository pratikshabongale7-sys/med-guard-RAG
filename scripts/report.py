"""Turn eval checkpoints into the final metric tables.

Reads eval/checkpoints/*.jsonl (from scripts.eval), flattens each nested record into the
shape metrics.py expects, and prints per-config metrics plus the headline comparisons:
  - gate ON vs OFF (hallucination + coverage + abstention)
  - verifier ablation: pure NLI vs pure LLM-judge vs two-tier
  - retrieval ablation (from eval/retrieval_ablation.json, if present)
  - risk-coverage / AURC (from an rc_measure run, if present)

Usage:
  uv run python -m scripts.report
"""

import json
from pathlib import Path

from app.eval import metrics

CKPT_DIR = Path("eval/checkpoints")


# --------------------------------------------------------------------------- #
# Loading / flattening
# --------------------------------------------------------------------------- #
def load_records(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def flatten(rec: dict) -> dict:
    """Nested {id, item, result} -> flat record for metrics.py."""
    item, result = rec["item"], rec["result"]
    retrieved: list[str] = []
    for c in result.get("citations", []):          # final cited chunks, in rank order
        pmid = c.get("pmid")
        if pmid and pmid not in retrieved:
            retrieved.append(pmid)
    return {
        "type": item["type"],
        "abstained": bool(result.get("abstained")),
        "error": result.get("error"),
        "confidence": result.get("confidence"),
        "correct": result.get("correct"),
        "claims": result.get("claims", []),
        "retrieved_pmids": retrieved,
        "relevant_pmids": item.get("relevant_pmids", []),
    }


def claims_from_records(recs: list[dict]) -> list[dict]:
    return [c for r in recs if not r["abstained"] for c in r["claims"]]


# --------------------------------------------------------------------------- #
# Per-config report
# --------------------------------------------------------------------------- #
def config_metrics(path: Path) -> dict:
    recs = [flatten(r) for r in load_records(path)]
    recs = [r for r in recs if not r["error"]]     # drop errored rows
    claims = claims_from_records(recs)

    answerable = [r for r in recs if r["type"] == "answerable" and r["relevant_pmids"]]
    rec5 = None
    if answerable:
        total = sum(metrics.precision_recall_at_k(r["retrieved_pmids"], r["relevant_pmids"], 5)[1]
                    for r in answerable)
        rec5 = round(total / len(answerable), 3)

    abst = metrics.abstention_scores(recs)
    sel = metrics.selective_scores(recs)
    return {
        "n": len(recs),
        "hallucination_rate": metrics.hallucination_rate(claims, "supported"),
        "citation_accuracy": metrics.citation_accuracy(claims),
        "coverage": sel["coverage"],
        "selective_risk": sel["selective_risk"],
        "abstention_precision": abst["abstention_precision"],
        "abstention_recall": abst["abstention_recall"],
        "abstention_f1": abst["abstention_f1"],
        "false_abstention_rate": abst["false_abstention_rate"],
        "retrieval_recall@5": rec5,
    }


def line(label: str, value) -> None:
    print(f"    {label:24} {value}")


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main() -> None:
    ckpts = sorted(CKPT_DIR.glob("*.jsonl")) if CKPT_DIR.exists() else []

    print("=" * 60)
    print("PER-CONFIG METRICS")
    print("=" * 60)
    for ck in ckpts:
        print(f"\n[{ck.stem}]")
        for k, v in config_metrics(ck).items():
            line(k, v)

    # ---- headline: gate ON vs OFF ----
    on, off = CKPT_DIR / "verify_llm.jsonl", CKPT_DIR / "guards_off.jsonl"
    if on.exists() and off.exists():
        print("\n" + "=" * 60)
        print("HEADLINE — GATE ON vs OFF")
        print("=" * 60)
        m_on, m_off = config_metrics(on), config_metrics(off)
        line("hallucination ON", m_on["hallucination_rate"])
        line("hallucination OFF", m_off["hallucination_rate"])
        line("abstention_recall ON", m_on["abstention_recall"])
        line("abstention_recall OFF", m_off["abstention_recall"])
        line("coverage ON", m_on["coverage"])
        line("coverage OFF", m_off["coverage"])

    # ---- verifier ablation: pure NLI vs pure LLM vs two-tier ----
    nli, llm = CKPT_DIR / "verify_nli.jsonl", CKPT_DIR / "verify_llm.jsonl"
    if nli.exists():
        print("\n" + "=" * 60)
        print("VERIFIER ABLATION — hallucination rate")
        print("=" * 60)
        nli_claims = claims_from_records([flatten(r) for r in load_records(nli)
                                          if not r["result"].get("error")])
        line("two-tier (production)", metrics.hallucination_rate(nli_claims, "supported"))
        line("pure NLI", metrics.hallucination_rate(nli_claims, "primary_supported"))
        if llm.exists():
            llm_claims = claims_from_records([flatten(r) for r in load_records(llm)
                                              if not r["result"].get("error")])
            line("pure LLM-judge", metrics.hallucination_rate(llm_claims, "primary_supported"))

    # ---- retrieval ablation (from retrieval_eval.py) ----
    retr = Path("eval/retrieval_ablation.json")
    if retr.exists():
        print("\n" + "=" * 60)
        print("RETRIEVAL ABLATION (recall@k / MRR)")
        print("=" * 60)
        for cfg, m in json.loads(retr.read_text()).items():
            line(cfg, m)

    # ---- risk-coverage / AURC (from an rc_measure run) ----
    rc = CKPT_DIR / "rc_measure.jsonl"
    if rc.exists():
        recs = [flatten(r) for r in load_records(rc) if not r["result"].get("error")]
        curve = metrics.risk_coverage_curve(recs)
        print("\n" + "=" * 60)
        print("RISK-COVERAGE")
        print("=" * 60)
        line("AURC (lower is better)", curve["aurc"])


if __name__ == "__main__":
    main()
