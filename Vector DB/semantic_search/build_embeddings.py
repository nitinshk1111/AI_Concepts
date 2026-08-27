"""
Embeds ArXiv paper abstracts using sentence-transformers.
Reads:  semantic_search/data/arxiv_papers.csv
Saves:
  semantic_search/embeddings/vectors.npy   — float32 array, shape (N, 384)
  semantic_search/embeddings/ids.npy       — string array of paper IDs (same order)

Run after download_data.py.
"""

import csv
import os

import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

DATA_PATH       = os.path.join(os.path.dirname(__file__), "data",       "arxiv_papers.csv")
VECTORS_PATH    = os.path.join(os.path.dirname(__file__), "embeddings", "vectors.npy")
IDS_PATH        = os.path.join(os.path.dirname(__file__), "embeddings", "ids.npy")
TITLES_PATH     = os.path.join(os.path.dirname(__file__), "embeddings", "titles.npy")
MODEL_NAME      = "all-MiniLM-L6-v2"   # 384-dim, fast, good semantic quality
BATCH_SIZE      = 64


def main():
    print(f"Loading data from {DATA_PATH}")
    abstracts, ids, titles = [], [], []
    with open(DATA_PATH, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            abstracts.append(row["abstract"])
            ids.append(row["id"])
            titles.append(row["title"])

    print(f"Loaded {len(abstracts)} papers")
    print(f"Embedding with {MODEL_NAME}...")

    model = SentenceTransformer(MODEL_NAME)

    # encode() handles batching internally; show_progress_bar gives a tqdm bar
    vectors = model.encode(
        abstracts,
        batch_size=BATCH_SIZE,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,   # L2-normalised → dot product == cosine similarity
    )

    np.save(VECTORS_PATH, vectors.astype(np.float32))
    np.save(IDS_PATH,     np.array(ids,    dtype=object))
    np.save(TITLES_PATH,  np.array(titles, dtype=object))

    print(f"Saved vectors : {VECTORS_PATH}  shape={vectors.shape}")
    print(f"Saved ids     : {IDS_PATH}")
    print(f"Saved titles  : {TITLES_PATH}")


if __name__ == "__main__":
    main()
