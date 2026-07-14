# MedGuard — Findings Log

Running log of decisions, gotchas, and answers to questions raised while building.

---

## Phase 0 — Skeleton (2026-07-09)

- FastAPI app with `/health`; config via `pydantic-settings`.
- `uv` for deps; committed `uv.lock` for reproducible CI/Docker installs.
- Dockerfile: non-root uid 1000, port 7860 (HF Spaces-ready).
- Tests: pytest + `TestClient`. Lint: ruff (`ignore = ["E501"]`).

## Phase 1 — Ingestion (2026-07-09)

- Source: PubMed abstracts via NCBI E-utilities (`esearch` → `efetch`).
- Chunking: `RecursiveCharacterTextSplitter`, size=800, overlap=100.
- Embeddings: fastembed `BAAI/bge-small-en-v1.5` (384-dim, ONNX, no torch — 8GB-friendly).
- Vector DB: Qdrant (Docker, cosine). Idempotent upserts via `uuid5` point IDs.
- Keyword: `rank-bm25` index pickled to `data/bm25.pkl`.
- Run: `uv run python -m app.ingestion.run --query "..." --max N`.

---

## Concepts & Q&A (answers to my doubts)

### Generation model: diffusion vs autoregressive
- **DiffusionGemma** (Google, June 2026): 26B MoE text-diffusion model, ~4x faster
  generation, Apache 2.0. But: no managed/free API (needs your own GPU — impossible on
  8GB Air), quality lower than autoregressive Gemma 4, weaker structured-output tooling.
- **Decision:** RAG doesn't care about decoder type (prompt in, text out), so it's
  *technically* usable — but for MedGuard use a **swappable autoregressive LLM**
  (Gemini free / Groq / local Ollama Qwen2.5-7B). Diffusion is a possible future-work
  ablation, not the workhorse. Speed isn't our bottleneck; faithfulness is.

### uv basics
- `uv add X` installs a package, updates `pyproject.toml`, and writes `uv.lock`.
  `uv add --dev X` = dev-only (not shipped in Docker).
- Don't activate the venv — prefix commands with `uv run` (uses the project `.venv`
  automatically). `.venv/bin/activate` is the mac path (`Scripts/activate` is Windows).
- Corrupt/empty `uv.lock` → `rm uv.lock` then `uv add ...` regenerates it.
- `VIRTUAL_ENV does not match ... will be ignored` = harmless; a stale venv is active.
  Fix with `deactivate` (or `unset VIRTUAL_ENV`).
- **Why commit `uv.lock`:** it pins exact versions + hashes for every (transitive) dep,
  so CI/Docker install byte-for-byte what worked locally. `pyproject.toml` only declares
  allowed *ranges*.

### pydantic-settings `extra="ignore"`
- Controls what happens with env/`.env` keys that don't match a `Settings` field.
- `"ignore"` = silently skip unknown keys (our choice — a shared `.env` can hold more
  vars than any one class declares). `"forbid"` = error on unknown. `"allow"` = attach
  them untyped.

### Dockerfile (per line)
- `FROM python:3.11-slim` — minimal Python base.
- `COPY --from=ghcr.io/astral-sh/uv:latest ...` — grab the `uv` binary from its image.
- `WORKDIR /app` — working dir for later commands.
- `COPY pyproject.toml uv.lock ./` — deps first (both files copied *into* `/app`; last
  arg is the destination). Done before code for **layer caching**.
- `RUN uv sync --frozen --no-dev` — install exact locked deps, runtime-only.
- `COPY app ./app` — app code in a later layer (changes often; deps rarely).
- `RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app` — make a non-root
  user AND give it ownership of `/app` (so it can write the venv at runtime).
- `USER appuser` — drop root; everything after (incl. `CMD`) runs as this user. Root-only
  steps must come *before* this line.
- `CMD ["uv","run","--no-sync","uvicorn","app.main:app","--host","0.0.0.0","--port","7860"]`
  — start server; `--no-sync` stops `uv run` re-syncing the (root-built) venv at boot.
- **Why layer order matters:** deps change rarely, code often. Deps in an earlier layer →
  rebuilds reuse the cached dependency layer when only code changed (seconds not minutes).
- **Non-root / uid 1000:** root (uid 0) can do anything in the container; running as a
  limited user contains damage if the app is compromised. 1000 = conventional first
  normal-user id; HF Spaces expects it.
- **Gotcha:** venv built as root at build time, container runs as appuser → `Permission
  denied` writing `.venv`. Fix = `chown -R appuser:appuser /app` + `uv run --no-sync`.

### Running / testing gotchas
- `localhost:8000` → 404 is **expected**; only `/health` is defined (try `/health` or
  `/docs`). Command typo: it's `uvicorn`, not `unicorn`.
- pytest `ModuleNotFoundError: No module named 'app'` → add to `pyproject.toml`:
  `[tool.pytest.ini_options]` / `pythonpath = ["."]` (pytest was putting the test dir on
  the path, not the project root). Also ensure `app/__init__.py` exists.
- Starlette warning "install httpx2" = harmless deprecation; test still passes. Optional:
  `uv remove --dev httpx` + `uv add --dev httpx2`.

### git remote/push
- `git remote -v` shows each remote **twice** (fetch + push) — not a duplicate.
- "non-fast-forward / pull before push" = remote has commits you don't; run
  `git pull --rebase origin main` (add `--allow-unrelated-histories` if needed) then push.
- `.idea/` is IDE config — keep it gitignored/untracked; its merge conflicts are noise.
  `git rm -r --cached .idea` stops tracking it while keeping it on disk.

### Data sources vs labeled datasets
- **Document sources (corpus, UNLABELED text we retrieve from):**
  - **NCBI** = the US org/infrastructure hosting many biomedical DBs + the E-utilities API.
  - **PubMed** = NCBI database of ~37M article citations/**abstracts** (+ metadata). Free
    text, no task labels. This is what Phase 1 ingests.
  - **NICE / WHO** = publishers of **clinical guidelines** (long authoritative docs, PDFs).
    Free text, not labeled; more clinically actionable but messier to parse.
- **Labeled datasets (used only to MEASURE, Phase 4):** PubMedQA (yes/no/maybe),
  MedQA (USMLE MCQ), MedMCQA (MCQ bank), MIRAGE (medical-RAG benchmark). These have gold
  answers.
- Model: **sources = the library (unlabeled); QA sets = the exam (labeled).**

### Ingestion flow (where data comes from)
- First uv command in Phase 1 just **installs packages** — no data.
- Real ingestion: `uv run python -m app.ingestion.run --query "..." --max 50` pulls live
  from PubMed over the internet: `esearch` → matching PMIDs, `efetch` → title/abstract/
  metadata as XML. Only local artifacts are Qdrant vectors + `data/bm25.pkl`.

### Why Qdrant (over alternatives)
- Chosen for: light footprint (one container), **native dense + sparse vectors** (clean
  hybrid in Phase 2), easy setup, persistent via volume, recognizable name.
- Chroma = easiest but weak hybrid/production story; pgvector = great if you want one SQL
  store for app + vectors, but more setup and DIY hybrid; FAISS = a library not a DB (no
  metadata/persistence); Weaviate/Milvus = heavier RAM; Pinecone = paid SaaS, no local.
- Retrieval sits behind our own functions → swappable later.

### Sparse vs dense vectors
- **Sparse:** vocabulary-length list, mostly zeros, non-zero only for words the text
  contains (weighted). Human-readable; this is BM25-style keyword matching. Catches exact
  terms (drug names, abbreviations).
- **Dense:** short fixed list (384 floats), mostly non-zero, normalized to length 1. Slots
  are abstract learned features, NOT words — capture *meaning* ("high blood pressure" ≈
  "hypertension"). Not individually interpretable.
- **Hybrid = use both.** "Keyword half in the same store" = Qdrant can hold sparse vectors
  natively, so BM25-style matching could live inside Qdrant instead of a separate
  `rank-bm25` pickle. Phase 1 keeps them separate for clarity; folding into Qdrant is a
  Phase 2 optimization (fewer moving parts).

### ONNX and fastembed
- **ONNX** = a portable, framework-free file format for a trained neural net; run via
  **ONNX Runtime**, a small, fast CPU inference engine — **no PyTorch needed**.
- **fastembed** (by Qdrant) = lightweight lib that ships embedding models pre-converted to
  ONNX; `model.embed(texts)` → vectors, CPU-friendly, no torch. Downloads model on first
  use. Contrast: `sentence-transformers` does the same job but drags in heavy PyTorch.
  Chosen for the 8GB machine; no quality loss for `bge-small`.

### Visualizing dense vectors
- A dense vector = a **point in 384-D space**; similar meanings sit close / small angle
  (cosine). Can't picture 384-D, so reduce to 2-D/3-D (PCA / UMAP / t-SNE) and scatter —
  similar chunks form visible clusters. Qdrant dashboard (`/dashboard`) has a built-in 2-D
  projection. The 2-D plot is a lossy shadow; retrieval uses the full 384 numbers.
- Analogy: describing a movie as [comedy, scariness, romance] is a 3-D dense vector;
  embeddings do this with 384 machine-learned axes.

### Article → chunk → vector (important correction)
- Vectors are **per-chunk, not per-article**. Article → split into chunks → **each chunk
  gets one dense AND one sparse vector** (1:1 pair, both describing that same chunk).
  100 articles → ~300–400 chunks → ~300–400 vector pairs.
- We do **not** group/cluster during ingestion. Every chunk is an independent point in
  Qdrant. Similar chunks (e.g. 6 heart-disease articles) naturally land near each other —
  **emergent geometry, not a stored group**. Nothing is labeled "heart group."
- Retrieval payoff: embed the question into the same space, find nearest points. Relevant
  chunks are nearby *because meaning = proximity*, not because we grouped them.

---

## Concepts & Q&A — evening session (2026-07-13)

### Docker: why two ports for Qdrant (6333 + 6334)
- Same server, two APIs. **6333 = REST/HTTP** (curl, browser `/dashboard`, easy debug).
  **6334 = gRPC** (faster binary protocol, better for bulk upserts). The Python client
  can use either. Phase 1 uses HTTP, so only 6333 is strictly needed; 6334 is mapped for
  future `prefer_grpc=True`. Minimal version: `-p 6333:6333` alone works.

### Qdrant health endpoints & the trailing 'z'
- Qdrant exposes `/healthz`, `/readyz`, `/livez` — **not** `/health` (so `/health` 404s).
- The `z` is a Google/Kubernetes convention: appended so the health path won't collide
  with a real app route. Not universal — my own FastAPI `/health` (Phase 0) is fine
  because I define that route myself. Endpoint name depends on the tool.
- Easiest visual check: open `http://localhost:6333/dashboard`.

### Empty-string defaults for api_key / email
- `""` = "not provided". Guards `if api_key:` / `if email:` are False for `""`, so blank
  values are simply not sent → anonymous NCBI access (3 req/sec). Fill them in `.env` to
  get the higher limit (10/sec) + polite identification. `str | None = None` is the
  stricter alternative (distinguishes unset vs deliberately empty); `""` is fine here.

### `resp.json()["esearchresult"]["idlist"]`
- `resp.json()` parses the HTTP body into a dict. esearch (retmode=json) returns
  `{"header": ..., "esearchresult": {"count", "idlist": [...], ...}}`. Drilling into
  `esearchresult` → `idlist` extracts just the list of PMID strings for `efetch` next.

### email on efetch vs esearch
- Was just a code inconsistency, not a rule. `email` is optional NCBI etiquette applying
  to ALL E-utilities calls; I only threaded it into esearch. Added it to `fetch_abstracts`
  for consistency (`run.py` now passes `settings.ncbi_email` to both). Behavior unchanged.

### rettype / retmode etc. — NCBI's or mine?
- Parameter **names** are fixed by NCBI's E-utilities API (can't rename/invent). You
  choose among allowed **values**: `retmode` = xml/json (format), `rettype` = abstract/
  medline/… (record kind), `retmax`/`retstart` (count/pagination), `db` = pubmed/pmc/…
  Valid values differ per endpoint; NCBI docs' parameter tables are authoritative.

### `_model: TextEmbedding | None = None`
- Module-level **lazy-singleton cache**. `_` prefix = "private" (convention). `| None` =
  union type annotation (holds a `TextEmbedding` or `None`; docs only, not enforced).
  Starts `None`; `get_model` builds the model on first call and reuses it after (avoids
  reloading the heavy ONNX model every call). Needs `global _model` to reassign it.

### Does `embed_texts()` make sparse vectors too?
- No — **dense only** (fastembed `model.embed`). The keyword/sparse half is separate:
  `build_bm25()` builds a `rank-bm25` index. In Phase 1 that's a pickled `BM25Okapi`
  object (word statistics), not vectors stored in Qdrant. Two functions, two stores.

### What `ensure_collection` does
- Creates the **empty** Qdrant collection (like a DB table) if missing, declaring vector
  size (384) + distance (cosine). Does NOT insert articles. Data goes in separately via
  `index_chunks` (`client.upsert`). "Ensure" = create only if it doesn't already exist.

### Payloads
- The arbitrary JSON metadata attached to each point (alongside id + vector). Here it's
  the whole chunk dict: `text`, `pmid`, `title`, `journal`, `year`, `url`. Used to return
  answer content + build citations, and for filtering (e.g. `year >= 2020`) at search time.
  Vector = for the math; payload = the human-readable stuff you actually want back.

### `build_bm25` walk-through
- `tokenized = [c["text"].lower().split() for c in chunks]` → lowercase word lists
  (case-insensitive, naive whitespace tokens). `BM25Okapi(tokenized)` builds the scorer
  from corpus stats (word rarity, chunk lengths). `Path(path).parent.mkdir(...)` ensures
  `data/` exists. `pickle.dump({"bm25", "chunks"}, f)` with `"wb"` saves scorer + chunks
  together (need chunks to map a BM25 hit position back to text/metadata).

### One collection or more?
- **One** (`medguard`) for this project. Distinguish topics via **payload filtering**, not
  separate collections. Use multiple only for: different vector dims (e.g. an embedding-
  model ablation — a collection is locked to one size+metric), or hard corpus isolation.
  Native sparse vectors (Phase 2) do NOT need a new collection.

### Adding email/api key to config — need `load_dotenv`?
- No. `pydantic-settings` already loads `.env` via `env_file=".env"` and maps `NCBI_EMAIL`
  → `ncbi_email` (case-insensitive). Just add the keys to `.env`. `load_dotenv`
  (python-dotenv) is the manual alternative (`os.getenv`) — redundant here, loses the
  typed `settings` object. Verify: `uv run python -c "from app.config import settings; print(settings.ncbi_email)"`.

### GOTCHA: `AttributeError: 'Settings' object has no attribute 'PUBMED_API_KEY'`
- `run.py` referenced `settings.PUBMED_API_KEY` but the field is `ncbi_api_key`. Attribute
  names are **lowercase** in code; the UPPERCASE form is only how it's written in `.env`.
  Fix = use `settings.ncbi_api_key` / `settings.ncbi_email`. List real fields:
  `print(list(settings.model_dump()))`.

### GOTCHA: `AssertionError: Unknown arguments: ['vectors_config']`
- `vectors_config` was reaching `collection_exists()` (which takes only the name) instead
  of `create_collection()` — a misplaced parenthesis. Fix: ensure `vectors_config=...` is
  inside `create_collection(...)`. Note: fastembed's "Download complete / Reconstruction
  complete" lines are just the model downloading — not an error.

### Rerun vs cleanup after a mid-pipeline crash
- Safe to just rerun — pipeline is idempotent: `ensure_collection` skips if exists,
  `index_chunks` upserts by deterministic `uuid5` (updates, no dupes), `build_bm25`
  overwrites. Full reset if needed: `curl -X DELETE http://localhost:6333/collections/medguard`
  + delete `data/bm25.pkl`.

### Why `build_bm25` overwrites `data/bm25.pkl` each run
- Mechanical: `open(path, "wb")` truncates on open (write mode replaces, never appends).
- By design: BM25 scores depend on corpus-wide stats, so the whole index must be rebuilt
  from the full chunk list — can't append one chunk.
- **IMPORTANT asymmetry:** Qdrant **accumulates** across runs; BM25 **replaces**. Running
  query A then query B → Qdrant has both, `bm25.pkl` has only B → indexes out of sync,
  which would break hybrid retrieval. Fine for single-run Phase 1; fix later (rebuild BM25
  from all Qdrant chunks, or move sparse into Qdrant).

### Context engineering / generator–evaluator pattern
- MedGuard's Phase 3 self-correction loop **is** a bounded generator–evaluator (reflection
  / CRAG / Self-RAG) pattern:
  - **Task** = clinician question.
  - **Context engineering** = retrieval (hybrid + rerank) curating which chunks fill the
    prompt — RAG *is* context engineering.
  - **Generator** = LLM drafting a cited answer.
  - **Two evaluators** = evidence grader (sufficient/weak/none, before generating) +
    faithfulness check (each claim entailed by evidence, NLI/LLM-judge, after drafting).
  - **Bounded loop** = weak evidence → rewrite query + re-retrieve, capped retries.
  - **Gate** = below threshold → abstain (evaluator vetoes generator).
- **Bounded, NOT agentic** — fixed corrective actions + retry cap keep it in scope
  (open-ended agent = Ops Copilot's identity). Prefer NLI over LLM-judge for
  reproducibility (ties to the NLI publication); use LLM-judge as an eval comparison.
- Not built until Phase 3; Phase 2 is generator + retrieval + citations only. Structure
  Phase 2 code so the evaluator/gate slot in cleanly.

### KEY INSIGHT: don't threshold on raw cosine for relevance/abstention
- Observed: on-topic query (high-BP) top score **0.71**; nonsense query (pneumonia, not in
  corpus) top score **0.67** — only 0.04 apart. Garbage barely below good.
- Why: bge embeddings are anisotropic (all vectors in a narrow cone), so cosine rarely
  drops below ~0.6 even for unrelated text. The score looks like a confidence % but isn't
  one; only **relative ordering within one query** is trustworthy, not absolute values.
- Consequence: no clean cosine cutoff separates "relevant" from "irrelevant" — any
  threshold that keeps good queries also lets garbage through. So **abstention cannot be
  built on a cosine threshold**. This is exactly why Phase 3 uses an evidence grader
  (LLM / NLI: "do these chunks actually answer the question?") instead of a score cutoff.
- Also: cross-topic separation can only be tested with a corpus that contains a correct
  target for each query; an unrelated query on a single-topic corpus proves nothing.
- Side note: identical top-k results = multiple chunks of one article; consider
  dedup/diversify by `pmid` in retrieval later.
