"""
CLI semantic search over ArXiv papers.

Usage:
    python semantic_search/search.py "attention mechanism in transformers"
    python semantic_search/search.py "graph neural networks" --top 10 --ef 100

On first run: builds the HNSW index from embeddings and caches it to
semantic_search/embeddings/hnsw_index.pkl — subsequent runs load from cache.
"""

import argparse
import os
import pickle
import sys
import time

import numpy as np
from sentence_transformers import SentenceTransformer

BASE   = os.path.dirname(__file__)
VECS   = os.path.join(BASE, "embeddings", "vectors.npy")
IDS    = os.path.join(BASE, "embeddings", "ids.npy")
TITLES = os.path.join(BASE, "embeddings", "titles.npy")
INDEX  = os.path.join(BASE, "embeddings", "hnsw_index.pkl")

sys.path.insert(0, BASE)
from hnsw.hnsw_index  import HNSWIndex
from hnsw.distance    import cosine_distance


def load_or_build_index(vectors: np.ndarray) -> HNSWIndex:
    if os.path.exists(INDEX):
        print("Loading cached HNSW index...")
        with open(INDEX, "rb") as f:
            return pickle.load(f)

    print(f"Building HNSW index over {len(vectors)} vectors (this runs once)...")
    t0    = time.time()
    index = HNSWIndex(M=16, ef_construction=200, distance_fn=cosine_distance)
    index.add_batch(vectors)
    elapsed = time.time() - t0
    print(f"Index built in {elapsed:.1f}s — saving to cache...")

    with open(INDEX, "wb") as f:
        pickle.dump(index, f)
    return index


def main():
    parser = argparse.ArgumentParser(description="Semantic search over ArXiv papers")
    parser.add_argument("query", type=str,            help="Search query")
    parser.add_argument("--top", type=int, default=5, help="Number of results (default: 5)")
    parser.add_argument("--ef",  type=int, default=50, help="HNSW ef_search — higher = better recall (default: 50)")
    args = parser.parse_args()

    for path in (VECS, IDS, TITLES):
        if not os.path.exists(path):
            sys.exit(f"Missing file: {path}\nRun build_embeddings.py first.")

    vectors = np.load(VECS)
    ids     = np.load(IDS,     allow_pickle=True)
    titles  = np.load(TITLES,  allow_pickle=True)

    index = load_or_build_index(vectors)

    model       = SentenceTransformer("all-MiniLM-L6-v2")
    query_vec   = model.encode(
        [args.query], normalize_embeddings=True, convert_to_numpy=True
    )[0].astype(np.float32)

    t0      = time.time()
    results = index.search(query_vec, k=args.top, ef=args.ef)
    latency = (time.time() - t0) * 1000

    print(f"\nQuery : \"{args.query}\"")
    print(f"Top {args.top} results  (HNSW, ef={args.ef}, latency={latency:.2f}ms)\n")
    print(f"{'Rank':<5} {'Distance':>10}  {'ArXiv ID':<15}  Title")
    print("-" * 90)
    for rank, (dist, idx) in enumerate(results, 1):
        title = titles[idx][:75] + "..." if len(titles[idx]) > 75 else titles[idx]
        print(f"{rank:<5} {dist:>10.4f}  {ids[idx]:<15}  {title}")


if __name__ == "__main__":
    main()
