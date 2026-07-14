from typing import Optional

from openai.types import CompletionUsage

from app.config import settings
from app.llm import generate
from app.retrieval import retrieve

SYSTEM_PROMPT = (
    "You are MedGuard, a clinical decision-support assistant. Answer ONLY from the "
    "numbered evidence excerpts provided (published medical literature). After each claim, "
    "cite the excerpt number(s) in square brackets, e.g. [1] or [2][3]. If the evidence "
    "does not contain the answer, say so plainly. This is decision-support for clinicians, "
    "not patient-facing medical advice."
)

def build_context(chunk: list[dict]) -> str:
    lines = []
    for i, c in enumerate(chunk, start=1):
        p = c["payload"]
        lines.append(f"[{i}] {p['title']} ({p['journal']}, {p['year']}). {p['text']}")

    return "\n\n".join(lines)

def answer_query(query: str) -> dict:
    top_chunks = retrieve(query)

    if not top_chunks:
        return {
            "answer": "No relevant evidence was found in the corpus for this question.",
            "citations": [],
            "evidence": {"count": 0, "items": []},
        }

    full_prompt = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Evidence:\n{build_context(top_chunks)}\n\nQuestion: {query}"},
    ]

    text, usage = generate(full_prompt, settings.llm_model)

    citations = [
        {"n": i, "title": chunk["payload"]["title"], "pmid": chunk["payload"]["pmid"], "url": chunk["payload"]["url"]}
        for i, chunk in enumerate(top_chunks, start=1)
    ]

    return {
        "answer": text,
        "citations": citations,
        "usage": {"prompt_tokens": usage.prompt_tokens, "completion_tokens": usage.completion_tokens},
    }


