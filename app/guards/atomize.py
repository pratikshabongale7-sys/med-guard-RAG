"""Faithfulness layer 1: split an answer into atomic, self-contained claims (decontextualized).
Helps in verifying multi-hop claims with NLI"""
# Layer 1 (atomize): split the answer into self-contained atomic claims (one fact each);
# #                    isolates any hallucinated claim from the good ones.

import re

from app.llm import generate

ATOMIZE_SYSTEM = (
    "Break the ANSWER into atomic claims. Each claim must:\n"
    "- assert exactly ONE fact,\n"
    "- be a complete standalone sentence with an explicit subject (never start with "
    "'it/they/this/these' or a bare verb),\n"
    "- resolve all pronouns and vague references (name 'the study' etc.),\n"
    "- carry every qualifier that affects the fact's truth (population, dose, timeframe, comparator),\n"
    "- add NO information beyond the ANSWER and KEEP the [n] citation markers from the source sentence on each claim,\n"
    "- each claim must be a DISTINCT fact - do not restate the same fact in different words or split an example into its own claim\n"
    "- extract ONLY facts the ANSWER explicitly asserts; do NOT add definitions, background, or world knowledge,\n"
    "- a short answer stating one fact should yield ONE claim, not several.\n"
    "Return ONE claim per line, no numbering."
)

_DANGLING = re.compile(r"^(it|they|this|these|those|he|she)\b", re.I)


def atomize(answer: str) -> list[str]:
    messages = [
        {"role": "system", "content": ATOMIZE_SYSTEM},
        {"role": "user", "content": f"ANSWER:\n{answer}\n\nCLAIMS:"},
    ]
    text, _ = generate(messages)
    print("Atomize LLM call")
    claims = [ln.strip(" -*•") for ln in (text or "").splitlines()]
    # guardrail 1: self-containment — drop very-short fragments of text / dangling-pronouns at the beginning
    return [c for c in claims if len(c) > 15 and not _DANGLING.match(c)]