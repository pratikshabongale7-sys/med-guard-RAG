from openai import OpenAI

from app.config import settings

_PROVIDERS = {
    "groq": {"base_url": "https://api.groq.com/openai/v1", "key": settings.groq_api_key},
    "cerebras": {"base_url": "https://api.cerebras.ai/v1", "key": settings.cerebras_api_key},
    "gemini": {"base_url": "https://generativelanguage.googleapis.com/v1beta/openai/", "key": settings.gemini_api_key},
}

_client: OpenAI | None = None # starts with None and then uses global reassigning later

def get_client() -> OpenAI:
    global _client
    if _client is None:
        cfg = _PROVIDERS[settings.llm_provider]
        _client = OpenAI(api_key=cfg["key"], base_url=cfg["base_url"])
    return _client

# returns the answer and the usage per query (answer generation)
def generate(messages: list[dict], model: str = settings.llm_model) -> tuple[str, object]:
    # for i, m in enumerate(messages):
    #     print(i, m["role"], type(m["content"]), repr(m["content"])[:120])
    response = get_client().chat.completions.create(
        model=model,
        messages=messages,
        temperature=settings.llm_temperature,
    )

    return response.choices[0].message.content, response.usage