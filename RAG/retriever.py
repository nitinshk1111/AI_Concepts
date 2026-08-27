"""
Retriever — embeds a query and finds the most relevant ArXiv papers via Weaviate.

Used by app.py. Not meant to be run directly.
"""

import os
import certifi
os.environ.setdefault("SSL_CERT_FILE", certifi.where())

import numpy as np
import pandas as pd
import weaviate
import weaviate.classes as wvc
from sentence_transformers import SentenceTransformer

from paths import (
    VECTORS_PATH, IDS_PATH, TITLES_PATH, DATA_PATH,
    EMBED_MODEL, COLLECTION
)


def load_data():
    vectors   = np.load(VECTORS_PATH)
    ids       = np.load(IDS_PATH,    allow_pickle=True)
    titles    = np.load(TITLES_PATH, allow_pickle=True)
    df        = pd.read_csv(DATA_PATH)
    abstracts = df["abstract"].values
    return vectors, ids, titles, abstracts


def build_weaviate_collection(client, vectors, ids, titles, abstracts):
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
            wvc.config.Property(name="abstract", data_type=wvc.config.DataType.TEXT),
        ],
    )

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
    return col


def retrieve(query: str, collection, model: SentenceTransformer, top_k: int = 5):
    query_vec = model.encode(
        [query], normalize_embeddings=True, convert_to_numpy=True
    )[0].astype(np.float32)

    results = collection.query.near_vector(
        near_vector=query_vec.tolist(),
        limit=top_k,
        return_metadata=wvc.query.MetadataQuery(distance=True),
    )

    return [
        {
            "arxiv_id": obj.properties["arxiv_id"],
            "title":    obj.properties["title"],
            "abstract": obj.properties["abstract"],
            "distance": round(obj.metadata.distance, 4),
        }
        for obj in results.objects
    ]
