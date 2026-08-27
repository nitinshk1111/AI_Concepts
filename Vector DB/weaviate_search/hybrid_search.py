"""
Step 4 — Sparse, Dense, and Hybrid search.

Dense  (near_vector) — pure embedding similarity. Great for semantic queries
                        like "what papers discuss graph attention?"
Sparse (bm25)        — pure keyword matching. Great for exact terms like
                        "BERT" or "ResNet-50".
Hybrid               — weighted blend of both, controlled by alpha:
                        alpha=0.0 → pure sparse (BM25 only)
                        alpha=0.5 → balanced
                        alpha=1.0 → pure dense (vector only)

Run from the 'Vector DB/' root:
    python weaviate_search/hybrid_search.py
    python weaviate_search/hybrid_search.py --query "BERT transformers" --alpha 0.25
"""

import argparse
import os
import sys

import certifi
os.environ.setdefault("SSL_CERT_FILE", certifi.where())

import numpy as np
import pandas as pd
import weaviate
import weaviate.classes as wvc
from sentence_transformers import SentenceTransformer

from paths import VECTORS_PATH, IDS_PATH, TITLES_PATH, DATA_PATH

COLLECTION = "ArxivHybrid"
MODEL_NAME = "all-MiniLM-L6-v2"
TOP_K      = 5


def build_collection(client, vectors, ids, titles, abstracts):
    if client.collections.exists(COLLECTION):
        client.collections.delete(COLLECTION)

    col = client.collections.create(
        name=COLLECTION,
        vector_config=wvc.config.Configure.Vectors.self_provided(
            vector_index_config=wvc.config.Configure.VectorIndex.hnsw(
                distance_metric=wvc.config.VectorDistances.COSINE,
            ),
        ),
        properties=[
            wvc.config.Property(
                name="arxiv_id", data_type=wvc.config.DataType.TEXT,
                index_searchable=False,
            ),
            wvc.config.Property(
                name="title", data_type=wvc.config.DataType.TEXT,
                index_searchable=True,   # BM25 on title
            ),
            wvc.config.Property(
                name="abstract", data_type=wvc.config.DataType.TEXT,
                index_searchable=True,   # BM25 on abstract
            ),
        ],
    )

    print(f"Inserting {len(vectors):,} papers (with abstracts for BM25)...")
    with col.batch.dynamic() as batch:
        for i in range(len(vectors)):
            batch.add_object(
                properties={
                    "arxiv_id": str(ids[i]),
                    "title":    str(titles[i]),
                    "abstract": str(abstracts[i]),
                },
                vector=vectors[i].tolist(),
            )
    print("Done.\n")
    return col


def print_results(label, results):
    print(f"\n── {label} ──")
    print(f"  {'Rank':<4} {'ArXiv ID':<15}  Title")
    print(f"  {'─'*60}")
    for rank, obj in enumerate(results.objects, 1):
        title = obj.properties["title"]
        title = title[:60] + "..." if len(title) > 60 else title
        print(f"  {rank:<4} {obj.properties['arxiv_id']:<15}  {title}")


def main():
    parser = argparse.ArgumentParser(description="Hybrid search over ArXiv papers")
    parser.add_argument("--query", type=str,
                        default="attention mechanism transformers",
                        help="Search query (default: 'attention mechanism transformers')")
    parser.add_argument("--alpha", type=float, default=0.5,
                        help="Hybrid alpha: 0=pure BM25, 1=pure vector (default: 0.5)")
    args = parser.parse_args()

    for path in (VECTORS_PATH, IDS_PATH, TITLES_PATH, DATA_PATH):
        if not path.exists():
            sys.exit(f"Missing: {path}")

    vectors   = np.load(VECTORS_PATH)
    ids       = np.load(IDS_PATH,    allow_pickle=True)
    titles    = np.load(TITLES_PATH, allow_pickle=True)
    df        = pd.read_csv(DATA_PATH)
    abstracts = df["abstract"].values

    model     = SentenceTransformer(MODEL_NAME)
    query_vec = model.encode(
        [args.query], normalize_embeddings=True, convert_to_numpy=True
    )[0].astype(np.float32)

    print(f'Query : "{args.query}"')
    print(f"Alpha : {args.alpha}  (0.0=pure BM25 · 1.0=pure vector)\n")

    client = weaviate.connect_to_embedded()
    try:
        col = build_collection(client, vectors, ids, titles, abstracts)

        # Dense — pure vector similarity
        dense = col.query.near_vector(
            near_vector=query_vec.tolist(),
            limit=TOP_K,
        )

        # Sparse — BM25 keyword match
        sparse = col.query.bm25(
            query=args.query,
            limit=TOP_K,
        )

        # Hybrid — weighted blend
        hybrid = col.query.hybrid(
            query=args.query,
            vector=query_vec.tolist(),
            alpha=args.alpha,
            limit=TOP_K,
        )

        print_results(f"Dense  (near_vector, alpha=1.0)", dense)
        print_results(f"Sparse (BM25,         alpha=0.0)", sparse)
        print_results(f"Hybrid (alpha={args.alpha})", hybrid)

        print("\n\nTip: run with --alpha 0.0 or --alpha 1.0 to see pure modes.")
        print("     run with a specific keyword like --query 'BERT' to see")
        print("     when BM25 outperforms dense search.")

    finally:
        client.close()


if __name__ == "__main__":
    main()
