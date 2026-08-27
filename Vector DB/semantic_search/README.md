# Semantic Search — Phase 1: HNSW from Scratch

A semantic search engine built in two phases:

- **Phase 1 (this folder):** implement HNSW from scratch using only NumPy — no ANN libraries
- **Phase 2 (coming later):** swap the hand-built index for Weaviate and compare

Dataset: ArXiv research paper abstracts (~20K papers).  
Embeddings: `sentence-transformers` (`all-MiniLM-L6-v2`, 384 dimensions).

---

## Folder structure

```
semantic_search/
├── hnsw/
│   ├── distance.py      ← distance metric functions (cosine, euclidean, dot product)
│   ├── flat_index.py    ← brute-force KNN baseline (ground truth for recall measurement)
│   └── hnsw_index.py   ← HNSW implementation from scratch (no hnswlib, no faiss)
├── data/                ← raw dataset CSV (gitignored — download via download_data.py)
├── embeddings/          ← precomputed .npy vector files (gitignored — build via build_embeddings.py)
├── download_data.py     ← fetch ArXiv abstracts from HuggingFace datasets
├── build_embeddings.py  ← embed abstracts → save as .npy
├── search.py            ← CLI: type a query, get top-K matching papers
└── benchmark.py         ← compare FlatIndex vs HNSWIndex on recall@10 and query latency
```

---

## How to run (once all files exist)

```bash
# 1. Download dataset
python semantic_search/download_data.py

# 2. Embed the abstracts
python semantic_search/build_embeddings.py

# 3. Search
python semantic_search/search.py "attention mechanism in transformers"

# 4. Benchmark flat vs HNSW
python semantic_search/benchmark.py
```

---

## What you will learn

| File | Concept |
|---|---|
| `hnsw/distance.py` | Cosine distance, euclidean distance, dot product — the math that defines "similar" |
| `hnsw/flat_index.py` | Why brute-force KNN doesn't scale — O(n×d) per query |
| `hnsw/hnsw_index.py` | How HNSW achieves O(log n) — layer assignment, graph construction, greedy beam search |
| `benchmark.py` | recall@K — the metric that measures how close "approximate" is to "exact" |

See [`hnsw/README.md`](hnsw/README.md) for a deep dive into the algorithm itself.
See [`CONCEPTS.md`](CONCEPTS.md) for everything learned while building this project — distance metrics, KNN, ANN, NSW, HNSW, recall@K, pickle, benchmark results.
