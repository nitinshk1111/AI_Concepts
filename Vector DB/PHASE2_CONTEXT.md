# Phase 2 Context — Weaviate Integration

Read this at the start of any session working on Phase 2.
For general repo orientation and how to work with Nitin, see [`CONTEXT.md`](CONTEXT.md).

---

## What Phase 2 is

Phase 1 built HNSW from scratch to understand the algorithm.
Phase 2 replaces that hand-built index with **Weaviate** — an open-source
production vector database that runs HNSW internally — while keeping the
same dataset, same embeddings, and same queries so the comparison is
meaningful.

Goal at the end of Phase 2: Nitin can say:
> "I built a semantic search engine, implemented HNSW from scratch,
> benchmarked it, then replaced it with Weaviate and compared the two."

---

## What Nitin already knows going into Phase 2

Everything in [`semantic_search/CONCEPTS.md`](semantic_search/CONCEPTS.md).
Specifically — don't re-teach these from scratch:

- What an embedding is and how sentence-transformers produces one
- Cosine distance vs euclidean vs dot product — and why cosine is used for text
- Why brute-force KNN doesn't scale (O(n×d) per query)
- How HNSW works internally: layer assignment, graph construction, greedy beam search, M, ef_construction, ef_search
- What recall@K means and how to compute it
- What Weaviate is (seen in course lesson 4) — collection schema, batch insert, near_vector query, filtered search, near_object search

What he has NOT yet done hands-on:
- Actually connected to Weaviate and run queries against it
- CRUD operations in Weaviate (lesson 5)
- Sparse, dense, hybrid search (lesson 6)
- Multilingual search (lesson 7)

---

## The codebase Phase 2 builds on

```
semantic_search/
├── hnsw/
│   ├── distance.py      ← distance functions (reuse as-is)
│   ├── flat_index.py    ← brute-force baseline (keep for recall comparison)
│   └── hnsw_index.py   ← hand-built HNSW (keep — Phase 2 compares against it)
├── data/
│   └── arxiv_papers.csv ← 20K AI/ML ArXiv papers (already downloaded)
├── embeddings/
│   ├── vectors.npy      ← 20K × 384 float32 embeddings (already built)
│   ├── ids.npy          ← ArXiv paper IDs
│   ├── titles.npy       ← paper titles
│   └── hnsw_index.pkl  ← cached Phase 1 index (Phase 2 won't use this)
├── download_data.py     ← already ran, don't re-run unless starting fresh
├── build_embeddings.py  ← already ran, don't re-run unless starting fresh
├── search.py            ← Phase 1 CLI search (keep as reference)
└── benchmark.py         ← Phase 1 benchmark (keep for comparison)
```

**Phase 2 adds new files alongside these — it does not overwrite Phase 1.**

---

## Weaviate basics (what the course covered in lesson 4)

`weaviate-client` is already installed in `.venv`.

```python
import weaviate

# Embedded mode: runs Weaviate inside the process, no server needed
client = weaviate.connect_to_embedded()

# Create a collection (vectorizer=None = we bring our own vectors)
collection = client.collections.create(
    name="ArxivPapers",
    vectorizer_config=wvc.config.Configure.Vectorizer.none(),
    vector_index_config=wvc.config.Configure.VectorIndex.hnsw(
        distance_metric=wvc.config.VectorDistances.COSINE
    ),
)

# Batch insert
with collection.batch.dynamic() as batch:
    for i, vec in enumerate(vectors):
        batch.add_object(
            properties={"arxiv_id": ids[i], "title": titles[i]},
            vector=vec.tolist(),
        )

# Search by vector
results = collection.query.near_vector(
    near_vector=query_vec.tolist(),
    limit=5,
    return_metadata=wvc.query.MetadataQuery(distance=True),
)

# Filtered search
from weaviate.classes.query import Filter
results = collection.query.near_vector(
    near_vector=query_vec.tolist(),
    filters=Filter.by_property("arxiv_id").equal("1703.08366"),
    limit=5,
)

client.close()
```

---

## Phase 2 build plan

### Step 1 — Weaviate search (equivalent of Phase 1's search.py)

New file: `semantic_search/weaviate_search.py`

- Connect to Weaviate embedded
- Load `vectors.npy`, `ids.npy`, `titles.npy`
- Insert all 20K papers into a Weaviate collection (batch insert)
- Accept CLI query → embed → `near_vector` search → display results
- Same output format as `search.py` so results are directly comparable

### Step 2 — Side-by-side benchmark

New file: `semantic_search/benchmark_weaviate.py`

- Same 50 queries used in Phase 1 benchmark
- Run against: FlatIndex, HNSWIndex (Phase 1), Weaviate
- Measure: query latency, recall@10 (flat as ground truth)
- Print a three-way comparison table

### Step 3 — CRUD operations (lesson 5)

New file: `semantic_search/weaviate_crud.py`

- Insert a new paper
- Read it back by ID
- Update a property
- Delete it
- Verify count before and after

### Step 4 — Sparse, Dense, Hybrid Search (lesson 6)

New file: `semantic_search/hybrid_search.py`

- **Dense search:** `near_vector` (what we've been doing — embedding similarity)
- **Sparse search:** BM25 keyword search (`query.bm25`)
- **Hybrid search:** combine both (`query.hybrid`) with alpha to weight them
- Compare: which returns better results for keyword-specific queries vs. semantic queries?

### Step 5 — Multilingual Search (lesson 7)

New file: `semantic_search/multilingual_search.py`

- Use a multilingual embedding model (e.g. `paraphrase-multilingual-MiniLM-L12-v2`)
- Insert papers embedded with this model
- Query in one language, retrieve papers from another
- Demonstrates that semantic search works across languages because the
  embedding space is shared — the meaning is the same even if the words differ

---

## Weaviate-specific concepts to teach as they come up

These weren't in Phase 1 — introduce them when the relevant step is being built:

| Concept | When to introduce |
|---|---|
| **Embedded vs server mode** | Step 1 setup |
| **Collection / schema** | Step 1 — Weaviate's equivalent of a database table |
| **Batch insert + dynamic batching** | Step 1 — why batch matters at scale |
| **CRUD in a vector DB** | Step 3 — updating/deleting vectors, not just inserting |
| **BM25 / sparse search** | Step 4 — TF-IDF style keyword matching, contrast with dense |
| **Hybrid search + alpha parameter** | Step 4 — 0.0=pure sparse, 1.0=pure dense, 0.5=balanced |
| **Multilingual embeddings** | Step 5 — shared embedding space across languages |

---

## How to run Phase 1 search (for reference / comparison)

```bash
cd "/Users/nitinshekar/Desktop/Work/Vector DB"

# Phase 1 (hand-built HNSW)
python semantic_search/search.py "graph neural networks"
python semantic_search/benchmark.py

# Phase 2 (Weaviate) — once built
python semantic_search/weaviate_search.py "graph neural networks"
python semantic_search/benchmark_weaviate.py
```

---

## Where the real depth lives

| File | What it covers |
|---|---|
| [`semantic_search/CONCEPTS.md`](semantic_search/CONCEPTS.md) | Everything from Phase 1 — distance metrics, KNN, ANN, HNSW, recall@K |
| [`semantic_search/hnsw/README.md`](semantic_search/hnsw/README.md) | HNSW algorithm internals + real benchmark numbers |
| [`semantic_search/README.md`](semantic_search/README.md) | Project overview, folder structure, how to run everything |
| This file | Phase 2 plan, Weaviate API patterns, what to teach when |

---

## Expected Phase 2 benchmark outcome

Phase 1 result for reference:
```
20K vectors, 384 dims, ef_search=50
HNSW (hand-built): ~1ms/query, recall@10 = 0.996, 26.5× speedup vs flat
```

Weaviate (expected): similar query latency, recall@10 close to 1.0.
The interesting part is not whether Weaviate is "better" — it's that
you built both, understand what Weaviate is doing internally, and can
explain the tradeoffs from first principles.
