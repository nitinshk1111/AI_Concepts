# Context for this repo

A quick-orientation file for a fresh Claude session (or future Nitin).
Read this first — it tells you who Nitin is, how to work with him,
what has already been built, and what comes next.

---

## Who this is for

Nitin — 11 years of software engineering experience, learning Python and
AI with the explicit goal of becoming an AI engineer. This repo is his
hands-on learning project built alongside the DeepLearning.AI short course
**[Vector Databases: From Embeddings to Applications](https://learn.deeplearning.ai/courses/vector-databases-embeddings-applications)**
(instructor: Sebastian Vitalets, Weaviate).

Sibling repo: `../Embedding/` — Nitin's earlier project covering embeddings,
cosine similarity, KMeans, KNN classification, PCA/t-SNE, outlier detection.
Those concepts don't need re-teaching from scratch — a quick reminder is enough.

---

## How to work with Nitin (read this every session)

- **Address him as "Hey Nitin"** at the start of responses.
- **Never run `git commit` unless he explicitly asks.**
- **Act as an expert AI/ML teacher, not just a coding assistant.**
  Proactively surface missing foundational concepts — don't wait to be asked.
- **Teaching style:** He has asked to be taught like a Principal AI Engineer
  mentoring someone — technical depth with good analogies and ASCII diagrams,
  not pure academic formalism and not dumbed-down basics. When he asks for
  "simple analogy + deep dive," deliver both in the same response.
- **Explanation sequence:** plain language first → one concrete example →
  check understanding → formalize (formula/table) only after he confirms.
- **If an analogy doesn't land:** swap its structure entirely, don't add
  more detail to the same broken analogy.
- **If he re-asks something already answered:** that means it didn't land —
  change approach, don't repeat with more rigor.
- **Verify before teaching:** run real code to confirm technical claims
  rather than asserting from general knowledge.
- **READMEs:** every project/subfolder gets its own README. Concept depth
  lives in README files and CONCEPTS.md — not in chat history.
- **Comments in code:** lean — only non-obvious WHYs. Full depth goes in
  the README, not inline comments.

---

## Repo structure

```
Vector DB/
├── pyproject.toml              — dependencies: weaviate-client, sentence-transformers,
│                                  datasets, numpy, tqdm, python-dotenv
├── .venv/                      — Python 3.14 virtualenv (gitignored)
├── src/vector_db/              — package skeleton
├── CONTEXT.md                  — this file (root orientation)
├── PHASE2_CONTEXT.md           — orientation specifically for Phase 2 (Weaviate)
└── semantic_search/            — the main project
    ├── CONCEPTS.md             — ★ everything learned in Phase 1, in one place
    ├── README.md               — project overview + how to run
    ├── download_data.py        — fetch 20K AI/ML ArXiv abstracts (HuggingFace)
    ├── build_embeddings.py     — embed abstracts → vectors.npy (384 dims)
    ├── search.py               — CLI search: query → top K papers
    ├── benchmark.py            — FlatIndex vs HNSWIndex: recall@K + latency
    ├── data/                   — arxiv_papers.csv (gitignored, ~20K rows)
    ├── embeddings/             — vectors.npy, ids.npy, titles.npy, hnsw_index.pkl
    └── hnsw/                   — ★ HNSW implemented from scratch
        ├── README.md           — deep dive: algorithm internals + real benchmark numbers
        ├── distance.py         — cosine, euclidean, dot product distance functions
        ├── flat_index.py       — brute-force KNN (ground truth baseline)
        └── hnsw_index.py      — full HNSW: layer assignment, insertion, beam search
```

---

## Phase 1 — DONE ✅

**What was built:** a full semantic search engine over 20K ArXiv AI/ML papers,
with HNSW implemented from scratch using only NumPy.

**Key files to read to get oriented:**
- [`semantic_search/CONCEPTS.md`](semantic_search/CONCEPTS.md) — everything learned, start here
- [`semantic_search/hnsw/README.md`](semantic_search/hnsw/README.md) — HNSW internals + benchmark numbers

**How to run:**
```bash
# search (index cached after first run)
python semantic_search/search.py "graph neural networks"
python semantic_search/search.py "reinforcement learning robotics" --top 10 --ef 100

# benchmark flat vs HNSW
python semantic_search/benchmark.py
```

**Benchmark results (M=16, ef_construction=200, ef_search=50):**
```
 Size  │ Flat (ms) │ HNSW build (s) │ HNSW (ms) │ Speedup │ Recall@10
───────┼───────────┼────────────────┼───────────┼─────────┼──────────
 1,000 │      1.39 │           2.43 │      0.61 │    2.3x │     0.998
 5,000 │      6.74 │          16.31 │      0.91 │    7.4x │     1.000
10,000 │     13.69 │          36.34 │      0.95 │   14.3x │     0.986
20,000 │     27.08 │          79.24 │      1.02 │   26.5x │     0.996
```

**Concepts covered in Phase 1:**
cosine/euclidean/dot product distance, brute-force KNN, ANN, NSW, HNSW
(layer assignment, graph construction, greedy beam search, M, ef_construction,
ef_search), recall@K, sentence embeddings, L2 normalisation, pickle serialisation.

---

## Phase 2 — NEXT ⬜

**Goal:** replace the hand-built `HNSWIndex` with Weaviate — a production
vector database — and keep everything else (same dataset, same embeddings,
same queries) so the comparison is meaningful.

**See [`PHASE2_CONTEXT.md`](PHASE2_CONTEXT.md) for a full brief on Phase 2.**

---

## Course syllabus — where we are

```
✅ Lesson 1  Introduction
✅ Lesson 2  How to Obtain Vector Representations of Data (autoencoder, embeddings)
✅ Lesson 3  Search for Similar Vectors (brute-force KNN, scaling problem)
✅ Lesson 4  Approximate Nearest Neighbours (NSW, HNSW, Weaviate intro)
⬜ Lesson 5  Vector Databases                          ← Phase 2 starts here
⬜ Lesson 6  Sparse, Dense, and Hybrid Search
⬜ Lesson 7  Application — Multilingual Search
⬜ Lesson 8  Conclusion
⬜ Lesson 9  Quiz
```
