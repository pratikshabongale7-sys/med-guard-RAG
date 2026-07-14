"""Step 6: Best Matching 25 for sparse vectors - lexical matching"""

import pickle
from pathlib import Path
from rank_bm25 import BM25Okapi

# turn each chunk into list of words (tokens), build the bm25 scorer
# bm25 search returns positions and chunks are used to retrieve the actual text and metadata
def build_bm25(chunks: list[dict], path: str) -> None:
    tokenized = [chunk["text"] for chunk in chunks] # whitespace delimited for Phase 1
    bm25 = BM25Okapi(tokenized)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump({"bm25": bm25, "chunks": chunks}, f)