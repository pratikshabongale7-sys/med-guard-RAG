"""Resumable eval harness: run the merged gold set through MedGuard, checkpoint each item.

Each result is written and flushed immediately, so a rate limit or crash never loses work
(re-runs skip already-done ids). One checkpoint file per config/ablation tag.

Modes (set via env, all as config flips):
  Normal (real system):        uv run python -m scripts.eval --config gate_on
  Gate OFF baseline (Phase 2): ENABLE_GUARDS=false uv run python -m scripts.eval --config gate_off
  Measure-only (risk-coverage): ENABLE_GUARDS=true GATE_ACTIVE=false \
                                uv run python -m scripts.eval --config rc_measure
  Retrieval / verifier ablations: RETRIEVAL_MODE=vector / RERANK_ENABLED=false / VERIFIER=nli ...
"""

import argparse
import json
import time
from pathlib import Path

from app.config import settings
from app.eval.correctness import _cached_decision
from app.rag import answer_query


def load_gold(path: str) -> list[dict]:
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def run(config_tag: str, gold_path: str, limit: int = 0) -> Path:
    ckpt = Path(settings.eval_dir) / "checkpoints" / f"{config_tag}.jsonl"
    ckpt.parent.mkdir(parents=True, exist_ok=True)

    done: set[str] = set()
    if ckpt.exists():                      # RESUME: skip items already recorded
        done = {json.loads(line)["id"] for line in ckpt.read_text().splitlines() if line.strip()}

    gold = load_gold(gold_path)
    if limit:
        gold = gold[:limit]
    print(f"[{config_tag}] {len(gold)} items ({len(done)} already done)")

    with open(ckpt, "a") as f:             # append mode = crash-safe
        for item in gold:
            if item["id"] in done:
                continue

            result = None
            for attempt in range(5):
                try:
                    result = answer_query(item["question"])
                    break
                except Exception as e:
                    if "429" in str(e) or "rate limit" in str(e).lower():
                        wait = 30 * (attempt + 1)
                        print(f"  rate limit at {item['id']} — waiting {wait}s ({attempt + 1}/5)")
                        time.sleep(wait)
                        continue
                    result = {"error": str(e)}
                    break
            if result is None:
                result = {"error": "rate limited after retries"}

            # Answer-correctness: only for ANSWERED, ANSWERABLE items (PubMedQA yes/no/maybe).
            # Cached, so ablations that reuse the same answer don't re-call the LLM.
            if (item["type"] == "answerable"
                    and not result.get("error")
                    and not result.get("abstained")):
                pred = _cached_decision(item["question"], result.get("answer", ""))
                result["predicted_decision"] = pred
                result["correct"] = (pred == item.get("gold_decision"))
            else:
                result["correct"] = None   # abstained / out_of_scope / errored -> not scored

            f.write(json.dumps({"id": item["id"], "item": item, "result": result}) + "\n")
            f.flush()                      # persist IMMEDIATELY, not at the end
            flag = " [abstained]" if result.get("abstained") else (" [error]" if result.get("error") else "")
            print(f"  done {item['id']}{flag}")

            #TODO: 5 only for gemini-2.0-flash - 15calls/min
            time.sleep(0)   # proactive throttle: pace under 15 req/min

    print(f"[{config_tag}] wrote -> {ckpt}")
    return ckpt


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Resumable MedGuard eval harness")
    ap.add_argument("--config", required=True, help="tag for this run/ablation (e.g. gate_on)")
    ap.add_argument("--gold", default="eval/goldset_20.jsonl", help="merged gold set path")
    ap.add_argument("--limit", type=int, default=0, help="only run first N items (0 = all)")
    args = ap.parse_args()
    run(args.config, args.gold, args.limit)
