from app.config import settings
from app.guards.claims import split_claims, cited_indices
from app.guards.gate import evaluate_answer
from app.guards.grade import grade_evidence
from app.guards.rewrite import rewrite_query
from app.guards.verify import verify_claim
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

# old answer generator = retrieve->generate->return - no guards
# def answer_query(query: str) -> dict:
#     top_chunks = retrieve(query)
#
#     if not top_chunks:
#         return {
#             "answer": "No relevant evidence was found in the corpus for this question.",
#             "citations": [],
#             "evidence": {"count": 0, "items": []},
#         }
#
#     full_prompt = [
#         {"role": "system", "content": SYSTEM_PROMPT},
#         {"role": "user", "content": f"Evidence:\n{build_context(top_chunks)}\n\nQuestion: {query}"},
#     ]
#
#     text, usage = generate(full_prompt, settings.llm_model)
#
#     citations = [
#         {"n": i, "title": chunk["payload"]["title"], "pmid": chunk["payload"]["pmid"], "url": chunk["payload"]["url"]}
#         for i, chunk in enumerate(top_chunks, start=1)
#     ]
#
#     return {
#         "answer": text,
#         "citations": citations,
#         "usage": {"prompt_tokens": usage.prompt_tokens, "completion_tokens": usage.completion_tokens},
#     }


# new answer generator = grading + retries → generate → verify → gate
REFUSAL = (
    "I can't answer this safely: the available evidence does not sufficiently support an answer to this question."
)

# refusal helper function
def _abstain(reason: str, attempts: list[str], evidence_count: int = 0, **extra) -> dict:
    return {
        "answer": REFUSAL,
        "abstained": True,
        "reason": reason,
        "citations": [],
        "confidence": 0.0,
        "attempts": attempts,
        "evidence": {"count": evidence_count, "items": []},
        **extra,
    }


def answer_query(query: str) -> dict:
    attempts: list[str] = []
    current_query = query # current_query = possibly re-written query & query is the user's original query
    chunks: list[dict] = []
    grade = "None"

    # Guard 1: grade evidence, bounded re-retrieval
    for attempt in range(settings.max_retrieval_retries + 1):
        attempts.append(current_query)
        chunks = retrieve(current_query)
        print("CHUNKS:", [c["payload"]["title"][:60] for c in chunks])

        if not settings.enable_guards:
            break # returns the chunks retrieved above in just one attempt without any guarding

        # always grade against user's original query and not the current_query in case the retrieved evidence from the
        # current_query doesn't answer the user's original query = real abstention
        grade = grade_evidence(query, build_context(chunks)) if chunks else None
        if grade == "SUFFICIENT":
            break
        if attempt < settings.max_retrieval_retries:
            current_query = rewrite_query(query, current_query)

    if not chunks:
        return _abstain("No evidence found", attempts)
    if settings.enable_guards and grade != "SUFFICIENT":
        return _abstain(f"evidence graded {grade} after {len(attempts)} attempt(s)",
                        attempts, len(chunks))

    # successful exit of above for loop = retrieved chunks which is passed below

    # create a draft answer
    message = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Evidence:\n{build_context(chunks)}\n\nQuestion: {query}"}
    ]

    draft, usage = generate(message)

    citations = [
        {"n": i, "title": c["payload"]["title"], "pmid": c["payload"]["pmid"], "url": c["payload"]["url"]}
        for i, c in enumerate(chunks, start=1)
    ]

    # returns the old - no guard answer after first retrieval
    if not settings.enable_guards:
        return {
            "answer": draft,
            "abstained": False,
            "citations": citations,
            "confidence": None,
            "attempts": attempts,
            "guards": "Disabled"
        }

    # Guard 2: faithfulness check + gate
    claims = split_claims(draft) # draft is what the LLM outputs after generating above
    verdicts = []

    for claim in claims:
        cited_ids = cited_indices(claim)
        # verify against chunks this claim cites; if it cites none then check against all chunks
        # the chunks are stored indexing from 1 and len(chunks) is to prevent model from hallucinating into checking for a chunk that was not retrieved
        sources = [chunks[i-1] for i in cited_ids if 1 <= i <= len(chunks)] or chunks
        verdicts.append(verify_claim(claim, [source["payload"]["text"] for source in sources]))

    result = evaluate_answer(verdicts)

    if not result["passed"]:
        return _abstain(
            f"Only {result['supported_ratio']:.0%} of claims were supported by the evidence",
            attempts, len(chunks), claims=verdicts,
        )

    return {
        "answer": draft,
        "abstained": False,
        "citations": citations,
        "confidence": result["supported_ratio"],
        "claims": verdicts,
        "attempts": attempts,
        "usage": {"prompt_tokens": usage.prompt_tokens, "completion_tokens": usage.completion_tokens},  # noqa
    }
