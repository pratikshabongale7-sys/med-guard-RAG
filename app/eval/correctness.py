"""Final decision from LLM's answer to given question"""
import hashlib, json

from app.llm import generate
from pathlib import Path

_DIR = Path("eval/cache/correctness")

DECISION_SYSTEM = (
    "Read the ANSWER and decide if it concludes yes, no, or maybe about the QUESTION. "
    "Reply with EXACTLY one word: yes, no, or maybe."
)

def decision_from_answer(question: str, answer: str) -> str:
    text, _ = generate([
        {"role": "system", "content": DECISION_SYSTEM},
        {"role": "user", "content": f"QUESTION: {question}\n\nANSWER: {answer}\n\nDecision:"},
    ])
    print("Correctness eval LLM call")
    t = (text or "").strip().lower()
    for d in ("maybe", "yes", "no"):   # check 'maybe' first so 'no' in 'maybe'... doesn't misfire
        if d in t:
            return d
    return "maybe"

def _cached_decision(question, answer):
    key = hashlib.sha256(f"{question}||{answer}".encode()).hexdigest()[:16]
    p = _DIR / f"{key}.json"
    if p.exists():
        return json.loads(p.read_text())
    val = decision_from_answer(question, answer)   # your uncached call
    _DIR.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(val))
    return val