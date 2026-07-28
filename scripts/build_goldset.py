"""Build the Phase 4 gold set from PubMedQA, then fetch real records from NCBI.

Writes two JSONL files in eval/:
  - pubmedqa_corpus.jsonl : real NCBI records (pmid/title/abstract/journal/year/url) to INGEST.
  - goldset_pubmedqa.jsonl : questions + labels (relevant_pmids, gold_answer, decision) to EVAL.

The pmid links the two: retrieval should surface the chunks whose pmid == relevant_pmids.
We fetch the canonical NCBI record (not PubMedQA's stripped context) so the corpus has the
SAME metadata shape as the hypertension articles already in the DB.

Usage:
  uv run python -m scripts.build_goldset --topics hypertension diabetes --per-topic 50
"""

import argparse
import json
from pathlib import Path

from datasets import load_dataset

from app.config import settings
from app.ingestion.fetch import fetch_abstracts

# PQA-L = the 1,000 expert-LABELLED split. Each item: question, context, long_answer,
# final_decision (yes/no/maybe), pubid (the PubMed id).
# Synonyms per topic raise recall of the slice, since the keyword filter is literal
# (e.g. catch questions that say "high blood pressure" rather than "hypertension").
TOPICS = {
    "hypertension": ["hypertension", "blood pressure", "antihypertensive"],
    "diabetes": ["diabetes", "diabetic", "glycemic", "insulin"],
    "asthma": ["asthma", "bronchodilator", "inhaled corticosteroid"],
}

OUT_DIR = Path(settings.eval_dir)


def matches(text: str, terms: list[str]) -> bool:
    t = text.lower()
    return any(term in t for term in terms)


def pick_from_pubmedqa(topics: list[str], per_topic: int) -> list[dict]:
    """Select PubMedQA items whose question or context matches a topic's terms."""
    dataset = load_dataset("qiaojin/PubMedQA", "pqa_labeled", split="train")

    picked: list[dict] = []
    seen: set[str] = set()  # avoid duplicates - same pmid landing under two topics
    for topic in topics:
        terms = TOPICS[topic]
        count = 0
        for data in dataset:
            if count >= per_topic:
                break
            pmid = str(data["pubid"])
            if pmid in seen:
                continue
            context = " ".join(data["context"]["contexts"])
            if not (matches(data["question"], terms) or matches(context, terms)):
                continue
            picked.append({
                "pmid": pmid,
                "question": data["question"],
                "topic": topic,
                "gold_answer": data.get("long_answer", ""),
                "gold_decision": data.get("final_decision", ""),
            })
            seen.add(pmid)
            count += 1
        print(f"  {topic}: selected {count}")
    return picked


def main() -> None:
    ap = argparse.ArgumentParser(description="Build the PubMedQA gold set + corpus")
    ap.add_argument("--topics", nargs="+", help=f"one or more of: {', '.join(TOPICS)}")
    ap.add_argument("--per-topic", type=int, default=50, help="max questions per topic")
    args = ap.parse_args()

    for t in args.topics:
        if t not in TOPICS:
            raise SystemExit(f"Unknown topic '{t}'. Choose from: {', '.join(TOPICS)}")

    print(f"Selecting from PubMedQA for topics: {args.topics}")
    picked = pick_from_pubmedqa(args.topics, args.per_topic)
    pmids = [p["pmid"] for p in picked]
    print(f"Selected {len(pmids)} questions; fetching real records from NCBI...")

    # Fetch canonical NCBI records -> uniform metadata (title/journal/year/url).
    # Reuses the ingestion fetch (batches all pmids into one efetch).
    articles = fetch_abstracts(pmids=pmids, api_key=settings.ncbi_api_key, email=settings.ncbi_email)
    by_pmid = {a["pmid"]: a for a in articles}
    print(f"NCBI returned {len(by_pmid)} records with abstracts")

    corpus, rows_qa = [], []
    for p in picked:
        article = by_pmid.get(p["pmid"])
        if article is None:  # no abstract from NCBI -> drop from BOTH files so they stay aligned
            continue
        corpus.append(article)  # already has pmid/title/abstract/journal/year/url
        rows_qa.append({
            "id": f"pqa-{p['pmid']}",
            "question": p["question"],
            "type": "answerable",
            "topic": p["topic"],
            "relevant_pmids": [p["pmid"]],  # the pmid whose chunks are the correct evidence
            "gold_answer": p["gold_answer"],
            "gold_decision": p["gold_decision"],
        })

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    corpus_path = OUT_DIR / "pubmedqa_corpus.jsonl"   # fed to ingestion
    gold_path = OUT_DIR / "goldset_pubmedqa.jsonl"    # fed to the eval harness
    with open(corpus_path, "w") as f:
        for c in corpus:
            f.write(json.dumps(c) + "\n")
    with open(gold_path, "w") as f:
        for r in rows_qa:
            f.write(json.dumps(r) + "\n")

    print(f"\nWrote {len(corpus)} corpus records -> {corpus_path}")
    print(f"Wrote {len(rows_qa)} gold questions -> {gold_path}")
    print("\nNext: ingest the corpus, then merge gold with your abstention set.")
    print(f"  uv run python -m app.ingestion.run --from-file {corpus_path}")


if __name__ == "__main__":
    main()
