"""
Step 5 — Multilingual semantic search.

The key idea: multilingual embedding models map text from different languages
into the SAME vector space. A query in Spanish lands near English papers about
the same topic — because meaning is shared across languages even when words differ.

Model: paraphrase-multilingual-MiniLM-L12-v2
       — 50+ languages, same 384-dim space as all-MiniLM-L6-v2

We embed a subset of ArXiv paper titles/abstracts with this model, insert them
into Weaviate, then query in multiple languages and compare results.

Run from the 'Vector DB/' root:
    python weaviate_search/multilingual_search.py
"""

import os
import sys

import certifi
os.environ.setdefault("SSL_CERT_FILE", certifi.where())

import numpy as np
import pandas as pd
import weaviate
import weaviate.classes as wvc
from sentence_transformers import SentenceTransformer

from paths import DATA_PATH

COLLECTION  = "ArxivMultilingual"
MODEL_NAME  = "paraphrase-multilingual-MiniLM-L12-v2"
SUBSET_SIZE = 2_000   # embed a subset — avoids 20K re-embed wait
TOP_K       = 5

# Same concept, queried in different languages
DEMO_QUERIES = [
    ("English",    "attention mechanism in neural networks"),
    ("Spanish",    "mecanismo de atención en redes neuronales"),
    ("French",     "mécanisme d'attention dans les réseaux de neurones"),
    ("German",     "Aufmerksamkeitsmechanismus in neuronalen Netzen"),
    ("Portuguese", "mecanismo de atenção em redes neurais"),
]


def section(title):
    print(f"\n{'═' * 60}")
    print(f"  {title}")
    print(f"{'═' * 60}")


def main():
    if not DATA_PATH.exists():
        sys.exit(f"Missing: {DATA_PATH}")

    df = pd.read_csv(DATA_PATH).head(SUBSET_SIZE)

    print(f"Loading multilingual model: {MODEL_NAME}")
    model = SentenceTransformer(MODEL_NAME)

    print(f"Embedding {SUBSET_SIZE:,} paper titles + abstracts...")
    texts   = (df["title"] + " " + df["abstract"]).tolist()
    vectors = model.encode(
        texts,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=True,
        batch_size=64,
    ).astype(np.float32)

    ids    = df["id"].astype(str).tolist()
    titles = df["title"].tolist()

    client = weaviate.connect_to_embedded()
    try:
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
                wvc.config.Property(name="arxiv_id", data_type=wvc.config.DataType.TEXT),
                wvc.config.Property(name="title",    data_type=wvc.config.DataType.TEXT),
            ],
        )

        print(f"\nInserting {SUBSET_SIZE:,} papers into Weaviate...")
        with col.batch.dynamic() as batch:
            for i in range(len(vectors)):
                batch.add_object(
                    properties={"arxiv_id": ids[i], "title": titles[i]},
                    vector=vectors[i].tolist(),
                )
        print("Ready.\n")

        section("Multilingual Semantic Search Demo")
        print(f"Same concept, {len(DEMO_QUERIES)} languages — should return similar papers:\n")

        for language, query in DEMO_QUERIES:
            query_vec = model.encode(
                [query], normalize_embeddings=True, convert_to_numpy=True
            )[0].astype(np.float32)

            results = col.query.near_vector(
                near_vector=query_vec.tolist(),
                limit=TOP_K,
                return_metadata=wvc.query.MetadataQuery(distance=True),
            )

            print(f"  [{language}] \"{query}\"")
            for rank, obj in enumerate(results.objects, 1):
                title = obj.properties["title"]
                title = title[:65] + "..." if len(title) > 65 else title
                print(f"    {rank}. {title}  (dist={obj.metadata.distance:.4f})")
            print()

        print("─" * 60)
        print("Observation: all 5 languages should return largely the same")
        print("papers — because the multilingual model maps 'attention in")
        print("neural networks' to roughly the same vector regardless of")
        print("which language you use to express it.")

    finally:
        client.close()


if __name__ == "__main__":
    main()
