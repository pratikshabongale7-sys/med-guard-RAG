from app.config import settings
from app.guards.atomize import atomize
from app.guards.claims import split_claims, cited_indices
from app.guards.gate import evaluate_answer
from app.guards.grade import grade_evidence
from app.guards.rewrite import rewrite_query
from app.guards.verify import verify_claim, to_sentences, _llm_judge, _similar_sentences, _BACKENDS
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
    current_query = query  # current_query = possibly re-written query & query is the user's original query
    chunks: list[dict] = []
    grade = "None"

    # Guard 1: grade evidence, bounded re-retrieval
    for attempt in range(settings.max_retrieval_retries + 1):
        attempts.append(current_query)
        chunks = retrieve(current_query)
        # print("CHUNKS:", [c["payload"]["title"][:60] for c in chunks])

        if not settings.enable_guards:
            break  # returns the chunks retrieved above in just one attempt without any guarding

        # always grade against user's original query and not the current_query in case the retrieved evidence from the
        # current_query doesn't answer the user's original query = real abstention
        grade = grade_evidence(query, build_context(chunks)) if chunks else None
        if grade == "SUFFICIENT":
            break
        if attempt < settings.max_retrieval_retries:
            current_query = rewrite_query(query, current_query)

    if not chunks:
        return _abstain("No evidence found", attempts)
    # gate_active = False is to let the system NOT abstain - computes confidence, answers everything → data for AURC
    if settings.enable_guards and settings.gate_active and grade != "SUFFICIENT":
        return _abstain(f"evidence graded {grade} after {len(attempts)} attempt(s)", attempts, len(chunks))

    # successful exit of above for loop = retrieved chunks which is passed below

    # create a draft answer
    message = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Evidence:\n{build_context(chunks)}\n\nQuestion: {query}"}
    ]

    draft, usage = generate(message)
    print("Draft message LLM call")

    citations = [
        {"citation": i, "chunk-id": c["payload"]["id"], "title": c["payload"]["title"], "pmid": c["payload"]["pmid"],
         "url": c["payload"]["url"]}
        for i, c in enumerate(chunks, start=1)
    ]

    # returns the old - no guard - answer after first retrieval
    if not settings.enable_guards:
        return {
            "answer": draft,
            "contexts": [c["payload"]["text"] for c in chunks],
            "abstained": False,
            "citations": citations,
            "confidence": None,
            "attempts": attempts,
            "guards": "Disabled"
        }

    # Guard 2: faithfulness check (layer 3 and layer 4 - faithfulness + citation) + gate
    sentences_all = to_sentences(chunks)  # faithfulness: all retrieved evidence
    verdicts = []

    if settings.verifier == "nli_judge":
        claims = atomize(draft) if settings.atomize_claims else split_claims(draft)  # draft is what the LLM outputs after generating above

        for claim in claims:
            verdict = verify_claim(claim, sentences_all) # faithfulness vs ALL evidence
            verdict["primary_supported"] = verdict["supported"] # pure NLI or LLM verdict before the following 2-tier rescue
            #TODO: uncomment for 2-tier NLI+LLM verdict
            # if not verdict["supported"] and settings.verifier != "llm_judge": # to not call llm again if llm_judge is used
            #     # Faithfulness layer 3 (two-tier): low-scoring claims get an LLM-judge second opinion; flags multi-hop NLI misses.
            #     top = _similar_sentences(claim, sentences_all, settings.premise_top_k)
            #     verdict["llm_judge"] = _llm_judge(claim, " ".join(top))
            #     verdict["supported"] = verdict["llm_judge"] >= settings.faithfulness_threshold

            # Citation accuracy: verify the claim against its CITED chunk(s) only — a separate signal
            # from faithfulness (reported, NOT used to gate). None if the claim cited nothing.
            cited_ids = cited_indices(claim)
            # the chunks are stored indexing from 1 and len(chunks) is to prevent model from hallucinating into checking for a chunk that was not retrieved
            cited_sources = [chunks[i - 1] for i in cited_ids if 1 <= i <= len(chunks)]
            verdict["citation_ok"] = (
                max((_BACKENDS[settings.verifier](claim, s["payload"]["text"]) for s in cited_sources),
                    default=0.0) >= settings.faithfulness_threshold
                if cited_sources else None  # uncited -> None, excluded from citation metric
            )
            verdicts.append(verdict)

    else:  # llm_judge verifier - citation sufficiency - cannot tell if an extra citation was there in the draft
        verdict = verify_claim(draft, sentences_all)
        verdict["primary_supported"] = verdict["supported"] # report.py needs this
        # citation accuracy, whole-draft level: is the draft supported by the chunks it cites?
        cited_ids = cited_indices(draft)  # all [n] across the answer
        cited_sources = [chunks[i - 1] for i in cited_ids if 1 <= i <= len(chunks)]
        cited_text = " ".join(s["payload"]["text"] for s in cited_sources)
        verdict["citation_ok"] = (
            _BACKENDS[settings.verifier](draft, cited_text) >= settings.faithfulness_threshold
            if cited_sources else None
        )
        verdicts.append(verdict)

    # Faithfulness layer 4 (gate): aggregate per-claim FAITHFULNESS verdicts -> supported_ratio;
    # answer if >= min_supported_ratio, else abstain. (Citation accuracy is reported, not gated.)
    result = evaluate_answer(verdicts)

    # 2-tier rescue: LLM only fires when NLI would abstain (post-grade faithfulness failure)
    if not result["passed"] and settings.verifier != "llm_judge":
        print("LLM coming to the rescue...")
        llm_ok = _llm_judge(draft, ". ".join(sentences_all)) >= settings.faithfulness_threshold  # ONE call, whole context
        if llm_ok:
            result["passed"] = True
            result["rescued_by"] = "llm"

    # gate_active = False is to let the system NOT abstain - computes confidence, answers everything → data for AURC
    if settings.gate_active and not result["passed"]:
        return _abstain(
            f"Only {result['supported_ratio']:.0%} of claims were supported by the evidence",
            attempts, len(chunks), claims=verdicts,
        )

    return {
        "answer": draft,
        "contexts": [c["payload"]["text"] for c in chunks],
        "abstained": False,
        "citations": citations,
        "confidence": result["supported_ratio"],
        "claims": verdicts,
        "attempts": attempts,
        "usage": {"prompt_tokens": usage.prompt_tokens, "completion_tokens": usage.completion_tokens},  # noqa
    }
