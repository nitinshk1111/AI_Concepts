"""
Step 1 — Weaviate CLI search over ArXiv papers.

Mirrors semantic_search/search.py but uses Weaviate embedded instead of the
hand-built HNSWIndex. Same dataset, same embeddings, same output format.

Usage:
    python weaviate_search/weaviate_search.py "attention mechanism"
    python weaviate_search/weaviate_search.py "graph neural networks" --top 10

Run from the 'Vector DB/' root.
"""

import argparse
import os
import sys
import time

import certifi
os.environ.setdefault("SSL_CERT_FILE", certifi.where())

import numpy as np
import weaviate
import weaviate.classes as wvc
from sentence_transformers import SentenceTransformer

from paths import VECTORS_PATH, IDS_PATH, TITLES_PATH

COLLECTION = "ArxivPapers"
MODEL_NAME = "all-MiniLM-L6-v2"


def build_collection(client, vectors, ids, titles):
    if client.collections.exists(COLLECTION):
        client.collections.delete(COLLECTION)

    collection = client.collections.create(
        name=COLLECTION,
        vector_config=wvc.config.Configure.Vectors.self_provided(
            vector_index_config=wvc.config.Configure.VectorIndex.hnsw(
                distance_metric=wvc.config.VectorDistances.COSINE,
            ),
        ),
        properties=[
            wvc.config.Property(name="arxiv_id", data_type=wvc.config.DataType.TEXT),
            wvc.config.Property(name="title",    data_type=wvc.config.DataType.TEXT),
        ],
    )

    print(f"Inserting {len(vectors):,} papers into Weaviate...")
    t0 = time.time()
    with collection.batch.dynamic() as batch:
        for i in range(len(vectors)):
            batch.add_object(
                properties={"arxiv_id": str(ids[i]), "title": str(titles[i])},
                vector=vectors[i].tolist(),
            )
    print(f"Inserted in {time.time() - t0:.1f}s\n")
    return collection


def main():
    parser = argparse.ArgumentParser(description="Weaviate semantic search over ArXiv papers")
    parser.add_argument("query", type=str)
    parser.add_argument("--top", type=int, default=5, help="Number of results (default: 5)")
    args = parser.parse_args()

    for path in (VECTORS_PATH, IDS_PATH, TITLES_PATH):
        if not path.exists():
            sys.exit(f"Missing: {path}\nRun semantic_search/build_embeddings.py first.")

    vectors = np.load(VECTORS_PATH)
    ids     = np.load(IDS_PATH,    allow_pickle=True)
    titles  = np.load(TITLES_PATH, allow_pickle=True)

    model     = SentenceTransformer(MODEL_NAME)
    query_vec = model.encode(
        [args.query], normalize_embeddings=True, convert_to_numpy=True
    )[0].astype(np.float32)

    client = weaviate.connect_to_embedded()
    try:
        collection = build_collection(client, vectors, ids, titles)

        t0 = time.time()
        results = collection.query.near_vector(
            near_vector=query_vec.tolist(),
            limit=args.top,
            return_metadata=wvc.query.MetadataQuery(distance=True),
        )
        latency = (time.time() - t0) * 1000

        print(f'Query : "{args.query}"')
        print(f"Top {args.top} results  (Weaviate HNSW, latency={latency:.2f}ms)\n")
        print(f"{'Rank':<5} {'Distance':>10}  {'ArXiv ID':<15}  Title")
        print("-" * 90)
        for rank, obj in enumerate(results.objects, 1):
            title = obj.properties["title"]
            title = title[:75] + "..." if len(title) > 75 else title
            print(f"{rank:<5} {obj.metadata.distance:>10.4f}  {obj.properties['arxiv_id']:<15}  {title}")
    finally:
        client.close()


if __name__ == "__main__":
    main()
