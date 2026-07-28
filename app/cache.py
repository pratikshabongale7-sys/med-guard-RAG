import hashlib
import json
from pathlib import Path

from app.config import settings
from app.corpus_version import current

_DIR = Path(settings.eval_dir) / "cache" / "retrieval"


def _key(query: str, mode: str, rerank: bool) -> str:
    raw = f"{query}|{mode}|{rerank}|{settings.qdrant_collection}|{current()}"
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def get(query, mode, rerank):
    p = _DIR / f"{_key(query, mode, rerank)}.json"
    return json.loads(p.read_text()) if p.exists() else None


def put(query, mode, rerank, value):
    _DIR.mkdir(parents=True, exist_ok=True)
    (_DIR / f"{_key(query, mode, rerank)}.json").write_text(json.dumps(value))