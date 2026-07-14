"""Hybrid retrieval: dense (Qdrant) + sparse (BM25) -> RRF fuse -> cross-encoder rerank."""
import pickle

from fastembed.rerank.cross_encoder import TextCrossEncoder

from app.config import settings
from app.ingestion.embed import get_model
from app.ingestion.index import get_client

_reranker: TextCrossEncoder | None = None


# embed query string using fastembed
def embed_query(text: str) -> list[float]:
    model = get_model(settings.embedding_model)

    return list(model.query_embed(text))[0].tolist()


# searching based on the embedded query through Qdrant's space for top_k candidates
def dense_search(query: str, top_k: int) -> list[dict]:
    client = get_client(settings.qdrant_url)
    query_vector = embed_query(query)
    hits = client.query_points(
        collection_name=settings.qdrant_collection,
        query=query_vector,
        limit=top_k
    ).points

    return [{"id": hit.payload["id"], "payload": hit.payload} for hit in hits]


def sparse_search(query: str, top_k: int) -> list[dict]:
    with open(settings.bm25_path, "rb") as f:
        data = pickle.load(f)

    # pickle file stores the corpus statistic - TF, DF, chunk length, etc. and the chunks themselves
    bm25, chunks = data["bm25"], data["chunks"]
    scores = bm25.get_scores(query.lower().split())
    # sort w.r.t. scores of chunks and return top_k candidates
    ranked = sorted(zip(chunks, scores), key=lambda pair: pair[1], reverse=True)[:top_k]

    return [{"id": chunk[0]["id"], "payload": chunk[0]} for chunk in ranked]


def reciprocal_rank_fusion(results_lists: list[list[dict]], k: int = 60) -> list[dict]:
    scores: dict[str, float] = {}
    payloads: dict[str, dict] = {}
    for results in results_lists:
        for rank, item in enumerate(results):
            _id = item["id"]
            # scores.get(_id, 0.0) => chunk's current score or 0 if it has been seen for the 1st time
            # 1.0 / (k + rank + 1) => k=60 is widely used on testing, to smooth out the ranks from either of the encoders and 1 is so that when rank=0
            # A: 1/61 (dense) + 1/66 (sparse) = 0.0164 + 0.0152 = 0.0316 => lower than below even though ranked higher in dense
            # B: 1/62 (dense) + 1/62 (sparse) = 0.0161 + 0.0161 = 0.0323 => same ranks on both resulting in overall higher score than above
            scores[_id] = scores.get(_id, 0.0) + 1.0 / (k + rank + 1)
            payloads[_id] = item["payload"]

    # sort w.r.t. scores of chunks
    fused_score = sorted(scores.items(), key=lambda pair: pair[1], reverse=True)
    return [{"id": _id, "payload": payloads[_id]} for _id, _ in fused_score]

def get_reranker() -> TextCrossEncoder:
    global _reranker
    if _reranker is None:
        _reranker = TextCrossEncoder(model_name=settings.reranker_model)

    return _reranker

# allowing the model to read the query and doc at once
def rerank(query: str, candidates: list[dict], top_k: int) -> list[dict]:
    reranker = get_reranker()
    docs = [chunk["payload"]["text"] for chunk in candidates] # passing entire text of payload to the reranker to read
    scores = list(reranker.rerank(query, docs)) # cross-encoding with query and docs together
    ranked = sorted(zip(candidates, scores), key=lambda pair: pair[1], reverse=True) # rank candidates based on scores

    return [chunk for chunk, _ in ranked[:top_k]] # return top_k ranked chunks

def retrieve(query: str) -> list[dict]:
    # Dense (Qdrant) returns ~20 and BM25 returns ~20 for the query
    dense = dense_search(query, settings.retrieval_top_k)
    sparse = sparse_search(query, settings.retrieval_top_k)
    # RRF fuses them - but it's a union with overlap merged, so you get up to 40, usually fewer.
    # Chunks that appear in both lists collapse into one entry (and get a higher summed score for showing up twice)
    fused = reciprocal_rank_fusion([dense, sparse])
    if not fused:
        return []
    # Cross-encoder reranks all fused candidates and returns the top 5
    return rerank(query, fused, settings.rerank_top_k)