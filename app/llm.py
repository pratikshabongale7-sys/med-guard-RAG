from openai import OpenAI

from app.config import settings

_PROVIDERS = {
    "groq": {"base_url": "https://api.groq.com/openai/v1", "key": settings.groq_api_key},
}

_client: OpenAI | None = None # starts with None and then uses global reassigning later

def get_client() -> OpenAI:
    global _client
    if _client is None:
        cfg = _PROVIDERS[settings.llm_provider]
        _client = OpenAI(api_key=cfg["key"], base_url=cfg["base_url"])
    return _client

# returns the answer and the usage per answer generation
def generate(messages: list[dict], model: str) -> tuple[str, object]:
    response = get_client().chat.completions.create(
        model=model or settings.llm_model,
        messages=messages,
        temperature=settings.llm_temperature,
    )

    return response.choices[0].message.content, response.usage