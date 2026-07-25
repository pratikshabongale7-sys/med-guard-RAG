"""CRAG-Style evidence grader: Does the retrieved evidence/answer actually answer the question?"""
from enum import Enum

from app.config import settings
from app.llm import generate

GRADER_SYSTEM = (
    "You grade whether retrieved medical evidence is sufficient to answer a question. "
    "Reply with EXACTLY ONE WORD, no punctuation or explanation:\n"
    "SUFFICIENT - the evidence directly answers the question\n"
    "WEAK - the evidence is related but does not clearly answer it\n"
    "NONE - the evidence is unrelated to the question"
)


class Label(Enum):
    SUFFICIENT = "SUFFICIENT"
    WEAK = "WEAK"
    NONE = "NONE"


def grade_evidence(question: str, evidence: str) -> str:
    messages = [
        {"role": "system", "content": GRADER_SYSTEM},
        {"role": "user", "content": f"Question: {question}\n\nEvidence: {evidence}\n\nGrade:"}
        # to skip the unnecessary texts from the model in the beginning "the answer should be..."
    ]

    text, _ = generate(messages, settings.llm_model)
    # parse the model output
    label = (text or "").strip().upper().strip(".")
    for valid_label in Label:
        if valid_label.value in label:
            print("RAW GRADE:", repr(text))
            return valid_label.value

    print("RAW GRADE:", repr(text))
    return Label.WEAK.value # safe default: unclear grade == treat as weak, trigger a retry

