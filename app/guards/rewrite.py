"""Retry on WEAK evidences till settings.max_retrieval_retries"""
from app.config import settings
from app.llm import generate

REWRITE_SYSTEM = (
    "You rewrite a clinical question to retrieve better evidence from a biomedical "
    "literature index. Expand abbreviations, add clinical synonyms and drug-class terms. "
    "Reply with ONLY the rewritten query - no quotes, no explanation."
)

def rewrite_query(original: str, previous: str) -> str:
    messages = [
        {"role": "system", "content": REWRITE_SYSTEM},
        {"role": "user", "content": (
            f"Original query: {original}\n"
            f"Previous query that retrieved weak evidence: {previous}\n"
            "Rewritten query:"
        )},
    ]

    text, _ = generate(messages, settings.llm_model)
    rewritten_query = (text or "").strip().strip('"')
    print("Query rewrite LLM call")
    return rewritten_query or original # never return empty — fall back to the original

