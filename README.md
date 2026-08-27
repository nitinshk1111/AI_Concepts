# AI Concepts

Hands-on learning projects built while studying AI/ML engineering.
Each folder is a self-contained project with its own README, concepts file, and runnable code.

---

## Projects

### [Embedding/](Embedding/)
**Sentence embeddings — from similarity to classification**

Covers how text is converted into vectors, how to measure similarity between them,
and how to build a language classifier using only embeddings — no neural network training.

| What | Where |
|------|-------|
| Core concepts | [Embedding/README.md](Embedding/README.md) |
| Sentence similarity | [compare_embeddings.py](Embedding/compare_embeddings.py) |
| PCA + t-SNE visualization | [visualization/](Embedding/visualization/) |
| Language classifier (KMeans, KNN, outlier detection) | [language_classifier/](Embedding/language_classifier/) |

**Key concepts:** embeddings, cosine similarity, L2 normalization, KMeans clustering,
KNN classification, PCA, t-SNE, outlier detection.

---

### [Vector DB/](Vector%20DB/)
**Vector databases — from HNSW from scratch to Weaviate**

Built a full semantic search engine over 20K ArXiv AI/ML papers. Phase 1 implements
HNSW from scratch. Phase 2 replaces it with Weaviate and extends to hybrid and
multilingual search.

| What | Where |
|------|-------|
| Core concepts | [Vector DB/README.md](Vector%20DB/README.md) |
| Phase 1 — HNSW from scratch | [semantic_search/](Vector%20DB/semantic_search/) |
| Phase 1 concepts (distance, KNN, ANN, HNSW, recall@K) | [semantic_search/CONCEPTS.md](Vector%20DB/semantic_search/CONCEPTS.md) |
| Phase 1 algorithm deep dive | [semantic_search/hnsw/README.md](Vector%20DB/semantic_search/hnsw/README.md) |
| Phase 2 — Weaviate integration | [weaviate_search/](Vector%20DB/weaviate_search/) |
| Phase 2 concepts (CRUD, hybrid, multilingual) | [weaviate_search/CONCEPTS.md](Vector%20DB/weaviate_search/CONCEPTS.md) |

**Key concepts:** vector search, brute-force KNN, ANN, HNSW, recall@K, Weaviate,
CRUD, BM25, dense vs sparse vs hybrid search, multilingual embeddings.

**Phase 1 benchmark (20K vectors, 384 dims):**
```
Flat (brute force): 27ms/query
HNSW (hand-built):   1ms/query  →  26.5× faster, recall@10 = 0.996
```

---

## Stack

- **Language:** Python 3.14
- **Embeddings:** `sentence-transformers` (`all-MiniLM-L6-v2`, `paraphrase-multilingual-MiniLM-L12-v2`)
- **Vector DB:** Weaviate (embedded mode)
- **Numerics:** NumPy, pandas
- **Visualization:** Plotly

## Course references

- [DeepLearning.AI — Vector Databases: From Embeddings to Applications](https://learn.deeplearning.ai/courses/vector-databases-embeddings-applications)
