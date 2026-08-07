"""Step 5: Qdrant - vector DB (works for both dense and sparse vectors."""

import uuid

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

from app.config import settings


def get_client(url: str) -> QdrantClient:
    return QdrantClient(url=url, api_key=settings.qdrant_api_key or None, timeout=120)


# create a collection (tables) to store points (vectors + chunk) if it already exist
def ensure_collection_exists(client: QdrantClient, collection: str, vector_size: int) -> None:
    if not client.collection_exists(collection):
        client.create_collection(
            collection_name=collection,
            vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
        )


# every point must be a deterministic uuid from the chunk key so every upsert affects the same point rather than duplicating it
def index_chunks(client: QdrantClient, collection: str, chunks: list[dict], vectors: list[list[float]]) -> int:
    points = [
        PointStruct(id=uuid.uuid5(uuid.NAMESPACE_OID, c["id"]), vector=v, payload=c)
        for c, v in zip(chunks, vectors)
    ]
    B = 100
    for i in range(0, len(points), B):
        client.upsert(collection_name=collection, points=points[i:i + B])
    return len(points)

# fetch chunks from Qdrant to be pushed to bm25 pickle so they stay consistent at all times
def fetch_all_chunks(client: QdrantClient, collection: str) -> list[dict]:
    chunks, offset = [], None
    while True:
        points, offset = client.scroll(
            collection_name=collection, limit=256, offset=offset, # few round-trips, modest memory per page
            with_payload=True, with_vectors=False,
        )
        chunks.extend(p.payload for p in points)
        if offset is None:
            break
    return chunks
