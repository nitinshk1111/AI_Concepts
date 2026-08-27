# Vector DB

Hands-on projects for learning vector databases, built alongside the DeepLearning.AI course
[Vector Databases: From Embeddings to Applications](https://learn.deeplearning.ai/courses/vector-databases-embeddings-applications) (Weaviate).

Two phases, same dataset (25K ArXiv AI/ML papers), increasing complexity:
Phase 1 builds everything from scratch. Phase 2 replaces it with a production vector database.

---

## Phase 1 — Semantic Search from Scratch

[`semantic_search/`](semantic_search/)

Built a full semantic search engine over 25K ArXiv papers without using any vector database library.
Implements flat brute-force search and HNSW (Hierarchical Navigable Small World) from scratch.

| What | Where |
|------|-------|
| Concepts (distance, KNN, ANN, HNSW, recall@K) | [semantic_search/CONCEPTS.md](semantic_search/CONCEPTS.md) |
| Data download | [semantic_search/download_data.py](semantic_search/download_data.py) |
| Build embeddings | [semantic_search/build_embeddings.py](semantic_search/build_embeddings.py) |
| CLI search | [semantic_search/search.py](semantic_search/search.py) |
| Flat index (brute force) | [semantic_search/flat_index.py](semantic_search/flat_index.py) |
| HNSW index (hand-built) | [semantic_search/hnsw/](semantic_search/hnsw/) |
| Benchmark | [semantic_search/benchmark.py](semantic_search/benchmark.py) |
| HNSW algorithm deep dive | [semantic_search/hnsw/README.md](semantic_search/hnsw/README.md) |

**Phase 1 benchmark (25K vectors, 384 dims):**
```
Flat (brute force): ~27ms/query
HNSW (hand-built):  ~ 1ms/query  →  26× faster, recall@10 = 0.996
```

**Key concepts:** vector search, dot product similarity, cosine similarity, KNN,
approximate nearest neighbour, HNSW layers and skip-list structure, recall@K.

---

## Phase 2 — Weaviate

[`weaviate_search/`](weaviate_search/)

Replaces the hand-built HNSW with Weaviate — an open-source production vector database.
Same dataset and embeddings, new capabilities: hybrid search, CRUD, multilingual queries.

| Step | File | What it does |
|------|------|-------------|
| 1 | `weaviate_search.py` | CLI semantic search via Weaviate — mirrors Phase 1's `search.py` |
| 2 | `benchmark_weaviate.py` | 3-way benchmark: FlatIndex vs HNSW (Phase 1) vs Weaviate |
| 3 | `weaviate_crud.py` | Full CRUD: insert, read, update, delete in a vector DB |
| 4 | `hybrid_search.py` | Dense vs BM25 vs Hybrid search with configurable alpha |
| 5 | `multilingual_search.py` | Cross-language semantic search — query in 5 languages |

| What | Where |
|------|-------|
| Concepts (Weaviate, CRUD, BM25, hybrid, multilingual) | [weaviate_search/CONCEPTS.md](weaviate_search/CONCEPTS.md) |
| Phase 2 README | [weaviate_search/README.md](weaviate_search/README.md) |

**Key concepts:** Weaviate embedded mode, HNSW via Weaviate, BM25 sparse retrieval,
hybrid search (alpha parameter), multilingual embeddings, CRUD lifecycle in a vector DB.

---

## How to run

```bash
cd "/Users/nitinshekar/Desktop/Work/Vector DB"
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

**Phase 1 — download data and build embeddings (one-time):**
```bash
python semantic_search/download_data.py
python semantic_search/build_embeddings.py
```

**Phase 1 — search:**
```bash
python semantic_search/search.py "attention mechanism in transformers"
```

**Phase 2 — Weaviate search:**
```bash
python weaviate_search/weaviate_search.py "graph neural networks"
```

See each phase's README for all available commands.

---

## Stack

- **Embeddings:** `all-MiniLM-L6-v2` (384-dim) and `paraphrase-multilingual-MiniLM-L12-v2` (384-dim)
- **Vector DB:** Weaviate (embedded mode, no separate server needed)
- **Data:** 25K ArXiv AI/ML papers (2018–2021)
- **Language:** Python 3.14

---

## Phase 3 — RAG App

[`../RAG/`](../RAG/)

Takes Phase 1 (embeddings) and Phase 2 (Weaviate) and adds an LLM generation layer on top,
turning semantic search into a full question-answering application.

| What | Where |
|------|-------|
| RAG app (Streamlit UI) | [RAG/app.py](../RAG/app.py) |
| Concepts (RAG, LLMs, prompt engineering) | [RAG/CONCEPTS.md](../RAG/CONCEPTS.md) |
| Setup guide | [RAG/SETUP.md](../RAG/SETUP.md) |

---

## Course reference

[DeepLearning.AI — Vector Databases: From Embeddings to Applications](https://learn.deeplearning.ai/courses/vector-databases-embeddings-applications)
