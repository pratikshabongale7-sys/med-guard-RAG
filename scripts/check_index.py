from app.config import settings
from app.ingestion.embed import embed_texts
from app.ingestion.index import get_client

client = get_client(settings.qdrant_url)
count = client.count(collection_name=settings.qdrant_collection).count
print(f"Qdrant holds {count} articles in '{settings.qdrant_collection}' at {settings.qdrant_url}")

query = "treatment of genital health in adults"
query_vec = embed_texts([query], settings.embedding_model)[0]
hits = client.query_points(collection_name=settings.qdrant_collection, query=query_vec, limit=3).points

for hit in hits:
    print(round(hit.score, 3), "|", hit.payload["title"][:70])
