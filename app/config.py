from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    app_name: str = "MedGuard"
    app_version: str = "0.1.0"
    environment: str = "development"

    # --- PubMed / NCBI ---
    ncbi_email: str = ""  # NCBI etiquette: identify yourself (optional)
    ncbi_api_key: str = ""  # optional: raises rate limit 3→10 req/sec

    # --- Chunking ---
    chunk_size: int = 800  # target characters per chunk
    chunk_overlap: int = 100  # characters shared between adjacent chunks

    # --- Embeddings ---
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    embedding_dim: int = 384  # must match the model's output size

    # --- Qdrant ---
    qdrant_url: str = "http://localhost:6333"
    qdrant_collection: str = "medguard"

    # --- BM25 ---
    bm25_path: str = "data/bm25.pkl"


settings = Settings()