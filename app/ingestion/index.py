"""Step 5: Qdrant - vector DB (works for both dense and sparse vectors."""

import uuid

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams


def get_client(url: str) -> QdrantClient:
    return QdrantClient(url=url)


# create a collection (tables) to store points (vectors + chunk) if it already exist
def ensure_collection_exists(client: QdrantClient, collection: str, vector_size: int) -> None:
    if not client.collection_exists(collection):
        client.create_collection(
            collection_name=collection,
            vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
        )


# every point must be a deterministic uuid from the chunk key so every upsert affects the same point rather than duplicating it
def index_chunks(client: QdrantClient, collection: str, chunks: list[dict], vectors: list[list[float]]) -> int:
    points = []
    for chunk, vector in zip(chunks, vectors):
        point_id = uuid.uuid5(uuid.NAMESPACE_OID, chunk["id"])
        points.append(PointStruct(id=point_id, vector=vector, payload=chunk))

    client.upsert(collection_name=collection, points=points)
    return len(points)


