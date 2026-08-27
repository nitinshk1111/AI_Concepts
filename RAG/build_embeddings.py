"""
Embeds paper abstracts using sentence-transformers.
Reads:  RAG/data/arxiv_papers.csv
Saves:
  RAG/embeddings/vectors.npy  — float32 array, shape (N, 384)
  RAG/embeddings/ids.npy      — string array of paper IDs
  RAG/embeddings/titles.npy   — string array of paper titles

Run after download_data.py.
"""

import csv
import os

import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

BASE_DIR     = os.path.dirname(__file__)
DATA_PATH    = os.path.join(BASE_DIR, "data",       "arxiv_papers.csv")
EMBEDDINGS   = os.path.join(BASE_DIR, "embeddings")
VECTORS_PATH = os.path.join(EMBEDDINGS, "vectors.npy")
IDS_PATH     = os.path.join(EMBEDDINGS, "ids.npy")
TITLES_PATH  = os.path.join(EMBEDDINGS, "titles.npy")
MODEL_NAME   = "all-MiniLM-L6-v2"
BATCH_SIZE   = 64


def main():
    os.makedirs(EMBEDDINGS, exist_ok=True)

    print(f"Loading papers from {DATA_PATH}")
    abstracts, ids, titles = [], [], []
    with open(DATA_PATH, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            abstracts.append(row["abstract"])
            ids.append(row["id"])
            titles.append(row["title"])

    print(f"Loaded {len(abstracts):,} papers")
    print(f"Embedding with {MODEL_NAME}...")

    model = SentenceTransformer(MODEL_NAME)
    vectors = model.encode(
        abstracts,
        batch_size=BATCH_SIZE,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    np.save(VECTORS_PATH, vectors.astype(np.float32))
    np.save(IDS_PATH,     np.array(ids,    dtype=object))
    np.save(TITLES_PATH,  np.array(titles, dtype=object))

    print(f"\nSaved vectors : {VECTORS_PATH}  shape={vectors.shape}")
    print(f"Saved ids     : {IDS_PATH}")
    print(f"Saved titles  : {TITLES_PATH}")
    print("\nNext step: streamlit run app.py")


if __name__ == "__main__":
    main()
