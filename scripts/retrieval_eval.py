"""Phase B: retrieval-only ablations — recall@k + MRR across retrieval configs.

NO LLM is used, so this costs ZERO API quota. It sweeps hybrid/vector/bm25 x rerank on/off,
runs retrieval over the answerable gold questions, and scores the retrieved pmids against
each question's relevant_pmids (PubMedQA gives one relevant pmid per question).

Usage:
  uv run python -m scripts.retrieval_eval --gold eval/goldset_small.jsonl
"""

import argparse
import json
from pathlib import Path

from app.config import settings
from app.retrieval import retrieve


def load_answerable(path: str) -> list[dict]:
    rows = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    return [r for r in rows if r["type"] == "answerable" and r.get("relevant_pmids")]


def ranked_pmids(question: str) -> list[str]:
    """Retrieved pmids in rank order, de-duplicated (an article can yield several chunks)."""
    out: list[str] = []
    for c in retrieve(question):
        pmid = c["payload"]["pmid"]
        if pmid not in out:
            out.append(pmid)
    return out


def score(gold: list[dict], ks=(1, 3, 5)) -> dict:
    recall = {k: 0 for k in ks}
    rr_sum = 0.0
    for r in gold:
        relevant = set(r["relevant_pmids"])
        got = ranked_pmids(r["question"])
        rank = next((i + 1 for i, p in enumerate(got) if p in relevant), 0)  # 1-based, 0 = miss
        rr_sum += 1.0 / rank if rank else 0.0
        for k in ks:
            if relevant & set(got[:k]):
                recall[k] += 1
    n = len(gold)
    return {"n": n, **{f"recall@{k}": round(recall[k] / n, 3) for k in ks},
            "mrr": round(rr_sum / n, 3)}


def main() -> None:
    ap = argparse.ArgumentParser(description="Retrieval-only ablations (no LLM)")
    ap.add_argument("--gold", default="eval/large_goldset_pubmedqa.jsonl")
    args = ap.parse_args()

    gold = load_answerable(args.gold)
    print(f"Retrieval eval on {len(gold)} answerable questions (no LLM, zero quota)\n")

    results = {}
    for mode in ("hybrid", "vector", "bm25"):
        for rerank in (True, False):
            settings.retrieval_mode = mode          # mutate the singleton; retrieve() reads it live
            settings.rerank_enabled = rerank
            tag = f"{mode}{'+rerank' if rerank else ''}"
            results[tag] = score(gold)
            print(f"{tag:16} {results[tag]}")

    out = Path(settings.eval_dir) / "retrieval_ablation.json"
    out.write_text(json.dumps(results, indent=2))
    print(f"\nsaved -> {out}")


if __name__ == "__main__":
    main()
