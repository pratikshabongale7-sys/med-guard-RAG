"""RAGAS cross-check (v0.4 collections API) over an eval checkpoint.

RAGAS is a SECOND opinion on our custom NLI faithfulness metric — not the source of truth.
v0.4 deprecated evaluate(); metrics are now async and scored per-row with `await m.ascore(...)`.

Two dependencies to satisfy for this to run:
  1. LLM wrapper — configure `llm` per the ragas v0.4 docs for your provider (Groq/OpenAI-
     compatible). See the migration guide (llm_factory / instructor_llm_factory).
  2. retrieved_contexts must be the CHUNK TEXTS the answer used. The harness result stores
     `citations` (title/pmid/url), not texts — so have answer_query ALSO return the chunk
     texts (e.g. result["contexts"] = [chunk["payload"]["text"], ...]) and store them.

If you'd rather not migrate now: `uv add "ragas<0.4"` keeps the old evaluate() API working.

Usage:
  uv run python -m scripts.ragas_eval --ckpt eval/checkpoints/gate_on.jsonl
"""

import argparse
import asyncio
import json
from pathlib import Path

from ragas.metrics.collections import AnswerRelevancy, Faithfulness

# --- 1. LLM SETUP (fill from ragas v0.4 docs for your provider) -----------------------------
# from ragas.llms import llm_factory
# llm = llm_factory("llama-3.3-70b-versatile", base_url="https://api.groq.com/openai/v1", ...)
llm = None  # TODO: replace with a configured ragas LLM before running


def rows_from_checkpoint(path: str) -> list[dict]:
    """Turn harness records into RAGAS rows for the ANSWERED items only."""
    rows = []
    for line in Path(path).read_text().splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        r = rec["result"]
        if r.get("abstained") or r.get("error"):
            continue                        # nothing to score if it refused / errored
        contexts = r.get("contexts")        # <-- chunk TEXTS (see dependency #2)
        if not contexts:                    # fallback: titles (weak; faithfulness needs texts)
            contexts = [c.get("title", "") for c in r.get("citations", [])]
        rows.append({
            "id": rec["id"],
            "user_input": rec["item"]["question"],
            "response": r.get("answer", ""),
            "retrieved_contexts": contexts,
            "reference": rec["item"].get("gold_answer", ""),
        })
    return rows


async def score(rows: list[dict]) -> list[dict]:
    faith = Faithfulness(llm=llm)
    relev = AnswerRelevancy(llm=llm)
    out = []
    for row in rows:
        f = await faith.ascore(response=row["response"], retrieved_contexts=row["retrieved_contexts"])
        a = await relev.ascore(user_input=row["user_input"], response=row["response"])
        out.append({"id": row["id"], "faithfulness": f.value, "answer_relevancy": a.value})
    return out


def summarize(scored: list[dict]) -> dict:
    if not scored:
        return {}
    n = len(scored)
    return {
        "n": n,
        "faithfulness": round(sum(s["faithfulness"] for s in scored) / n, 3),
        "answer_relevancy": round(sum(s["answer_relevancy"] for s in scored) / n, 3),
    }


async def main(ckpt: str) -> None:
    rows = rows_from_checkpoint(ckpt)
    print(f"Scoring {len(rows)} answered rows from {ckpt}")
    scored = await score(rows)
    print(json.dumps(summarize(scored), indent=2))
    out_path = Path(ckpt).with_suffix(".ragas.json")
    out_path.write_text(json.dumps(scored, indent=2))
    print(f"Per-row scores -> {out_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="RAGAS cross-check over an eval checkpoint")
    ap.add_argument("--ckpt", required=True, help="checkpoint jsonl from scripts.eval")
    args = ap.parse_args()
    asyncio.run(main(args.ckpt))