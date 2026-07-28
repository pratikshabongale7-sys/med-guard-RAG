"""Step 7: the Orchestrator """

import argparse
import json

from pathlib import Path
from app.config import settings
from app.corpus_version import bump
from app.ingestion.bm25_index import build_bm25
from app.ingestion.chunk import chunk_articles
from app.ingestion.clean import clean_text
from app.ingestion.embed import embed_texts
from app.ingestion.fetch import fetch_abstracts, search_pubmed
from app.ingestion.index import ensure_collection_exists, get_client, index_chunks, fetch_all_chunks


def load_articles_from_file(path: str) -> list[dict]:
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]

def main() -> None:
    parser = argparse.ArgumentParser(description="MedGuard ingestion pipeline")
    parser.add_argument("--query", type=str, help="PubMed search query")
    parser.add_argument("--max", type=int, default=50, help="Max articles to fetch")
    parser.add_argument("--from-file", type=str, help="JSONL of pre-fetched articles to ingest")
    args = parser.parse_args()

    if not args.query and not args.from_file:
        parser.error("provide either --query or --from-file")

    # step 1+2: get articles - from a file (build_goldset.py - pubmedqa_corpus.jsonl), or via live PubMed search
    if args.from_file:
        articles = load_articles_from_file(args.from_file)
        print(f"Loaded {len(articles)} articles from {args.from_file}")
    else:
        pmids = search_pubmed(query=args.query, max_results=args.max,
                              api_key=settings.ncbi_api_key, email=settings.ncbi_email)
        print(f"Found {len(pmids)} PMIDs")
        articles = fetch_abstracts(pmids=pmids, api_key=settings.ncbi_api_key,
                                   email=settings.ncbi_email)

    # step 3
    for article in articles:
        article["title"] = clean_text(article["title"])
        article["abstract"] = clean_text(article["abstract"])
    print(f"Fetched {len(articles)} articles")

    # step 4
    chunks = chunk_articles(articles=articles, chunk_size=settings.chunk_size, chunk_overlap=settings.chunk_overlap)
    print(f"Created {len(chunks)} chunks")

    # step 5
    dense_vectors = embed_texts(texts=[chunk["text"] for chunk in chunks], model_name=settings.embedding_model)
    print(f"Embedded {len(dense_vectors)} chunks (dim={len(dense_vectors[0])})")

    # step 6
    client = get_client(settings.qdrant_url)
    ensure_collection_exists(client=client, collection=settings.qdrant_collection, vector_size=len(dense_vectors[0]))
    n = index_chunks(client=client, collection=settings.qdrant_collection, chunks=chunks, vectors=dense_vectors)
    print(f"Indexed {n} chunks/vectors into Qdrant collection '{settings.qdrant_collection}'")

    # step 7 - build BM25 over EVERYTHING now in Qdrant (keeps the two indexes in sync)
    all_chunks = fetch_all_chunks(client=client, collection=settings.qdrant_collection)
    build_bm25(all_chunks, settings.bm25_path)
    version = bump()
    print(f"Corpus version -> {version}")
    print(f"Built BM25 over {len(all_chunks)} chunks (full corpus)")


if __name__ == "__main__":
    main()