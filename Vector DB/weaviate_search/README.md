# Phase 2 — Weaviate Search

Phase 2 replaces the hand-built `HNSWIndex` from Phase 1 with **Weaviate** — an
open-source production vector database. Same dataset (20K ArXiv papers), same
embeddings, same queries. The goal: understand what Weaviate is doing internally,
not just how to call its API.

> **Concepts reference:** [CONCEPTS.md](CONCEPTS.md) — everything learned in Phase 2.
> **Phase 1 concepts:** [../semantic_search/CONCEPTS.md](../semantic_search/CONCEPTS.md)

---

## What was built

| Step | File | What it does |
|------|------|-------------|
| 1 | `weaviate_search.py` | CLI semantic search via Weaviate — mirrors Phase 1's `search.py` |
| 2 | `benchmark_weaviate.py` | 3-way benchmark: FlatIndex vs HNSW (Phase 1) vs Weaviate |
| 3 | `weaviate_crud.py` | Full CRUD: insert, read, update, delete in a vector DB |
| 4 | `hybrid_search.py` | Dense vs BM25 vs Hybrid search with configurable alpha |
| 5 | `multilingual_search.py` | Cross-language semantic search — query in 5 languages |

---

## Folder structure

```
weaviate_search/
├── paths.py                  ← shared path config + Phase 1 import helper
├── weaviate_search.py        ← Step 1: Weaviate CLI search
├── benchmark_weaviate.py     ← Step 2: three-way latency + recall benchmark
├── weaviate_crud.py          ← Step 3: CRUD lifecycle demo
├── hybrid_search.py          ← Step 4: dense vs sparse vs hybrid
├── multilingual_search.py   ← Step 5: cross-language semantic search
├── README.md                 ← this file
└── CONCEPTS.md               ← Phase 2 theory and key ideas

Data (shared with Phase 1, read-only):
../semantic_search/embeddings/
    ├── vectors.npy    ← 20K × 384 float32 (all-MiniLM-L6-v2)
    ├── ids.npy        ← ArXiv paper IDs
    └── titles.npy     ← paper titles
../semantic_search/data/
    └── arxiv_papers.csv  ← 20K rows: id, title, abstract, categories
```

---

## How to run

All commands from the **`Vector DB/`** root with venv active:

```bash
cd "/Users/nitinshekar/Desktop/Work/Vector DB"
source .venv/bin/activate
```

**Step 1 — Weaviate search**
```bash
python weaviate_search/weaviate_search.py "attention mechanism"
python weaviate_search/weaviate_search.py "graph neural networks" --top 10
```

**Step 2 — Three-way benchmark**
```bash
python weaviate_search/benchmark_weaviate.py
```

**Step 3 — CRUD operations**
```bash
python weaviate_search/weaviate_crud.py
```

**Step 4 — Hybrid search**
```bash
python weaviate_search/hybrid_search.py
python weaviate_search/hybrid_search.py --query "BERT language model" --alpha 0.25
python weaviate_search/hybrid_search.py --query "ResNet image classification" --alpha 0.0
```

**Step 5 — Multilingual search**
```bash
python weaviate_search/multilingual_search.py
```

---

## What you will learn

| Script | Concept |
|--------|---------|
| `weaviate_search.py` | Weaviate embedded mode, collection schema, batch insert, near_vector query |
| `benchmark_weaviate.py` | Why Weaviate latency > hand-built HNSW even though Weaviate's Go HNSW is faster |
| `weaviate_crud.py` | UUID as primary key, update without re-embedding, delete with count verification |
| `hybrid_search.py` | BM25 (sparse), dense, hybrid — when each wins and what alpha controls |
| `multilingual_search.py` | Parallel text training, shared embedding space across languages, why re-embedding is required |

---

## Step-by-step deep dive

### Step 1 — `weaviate_search.py`

**What it does:** Inserts all 20K ArXiv papers into Weaviate embedded, then accepts
a CLI query, embeds it, and runs a `near_vector` search.

**How Weaviate embedded works:**
`weaviate.connect_to_embedded()` downloads a Go binary (~30MB, one-time) and
starts it as a subprocess on localhost. You talk to it over gRPC/HTTP. When the
script exits, the process shuts down and data is gone — no persistence between runs.

**Collection schema:**
```python
client.collections.create(
    name="ArxivPapers",
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
```

Key choices:
- `self_provided` — we supply vectors ourselves; Weaviate will not call an external API to generate them
- `COSINE` distance — must match how embeddings were created (L2-normalized → cosine)
- Properties define metadata stored alongside each vector

**Batch insert:**
Inserting 20K objects one-by-one over HTTP = 20K round-trips = slow. `batch.dynamic()`
groups objects automatically and adjusts batch size for throughput:
```python
with collection.batch.dynamic() as batch:
    for i in range(len(vectors)):
        batch.add_object(properties={...}, vector=vectors[i].tolist())
```
Inserts 20K papers in ~4 seconds.

**Compare with Phase 1:**
```
Phase 1 search.py          Phase 2 weaviate_search.py
────────────────────       ──────────────────────────
HNSWIndex.search()    →    collection.query.near_vector()
Python list of tuples →    results.objects (each with .properties, .metadata.distance)
no insert overhead    →    ~4s insert per run (no persistence)
~1ms query            →    ~5ms query (includes gRPC overhead)
```

---

### Step 2 — `benchmark_weaviate.py`

**What it does:** Runs 50 queries against all three indexes simultaneously and
prints a side-by-side comparison table.

**Methodology:**
- Same 50 queries drawn with `np.random.seed(42)` — same queries used in Phase 1 `benchmark.py`
- FlatIndex = ground truth for recall@10 (exact brute force)
- Recall@10 for HNSW and Weaviate = fraction of FlatIndex's top-10 that appear in their top-10
- Only tested at 20K (full dataset) — unlike Phase 1 which tested at 1K/5K/10K/20K

**Why not test multiple sizes for Weaviate?**
Phase 1 tested multiple sizes because HNSW *build time* was interesting to measure as dataset grows.
For Weaviate, the interesting story is the *query latency difference* vs the hand-built index — and
that's best shown at scale where the Go vs Python difference matters most.

**Why is Weaviate query latency higher than hand-built HNSW?**
```
Phase 1 HNSW (Python, in-process):
  Python function call → result list
  Zero network overhead

Weaviate (Go, separate process):
  Python → serialize → gRPC → Weaviate (Go) → HNSW search → gRPC response → deserialize → Python
  Adds ~2–5ms per query regardless of dataset size
```
At 20K vectors, the gRPC overhead dominates. At 1M+ vectors, Weaviate's compiled
Go HNSW would massively outpace the Python implementation — the overhead becomes
negligible relative to the speedup.

**Expected output (fill in your numbers after running):**
```
Method                    Setup     Latency (ms)   Speedup   Recall@10
────────────────────────────────────────────────────────────────────────
FlatIndex (exact)            —              —          1.0x       1.000
HNSW (hand-built)        XXs build        X.XXms    XX.Xx       X.XXX
Weaviate                 XXs insert       X.XXms    XX.Xx       X.XXX
```

---

### Step 3 — `weaviate_crud.py`

**What it does:** Demonstrates that Weaviate is a full database, not just a search index.

**CRUD operations and their Weaviate API:**

| Operation | Method | Notes |
|-----------|--------|-------|
| Create | `col.data.insert(properties, vector)` | Returns a UUID Weaviate assigns |
| Read | `col.query.fetch_object_by_id(uuid)` | Direct lookup by UUID, O(1) |
| Update | `col.data.update(uuid, properties)` | Updates metadata only — vector unchanged |
| Delete | `col.data.delete_by_id(uuid)` | Permanent — no soft delete by default |
| Count | `col.aggregate.over_all(total_count=True)` | Cheap aggregation, no scan |

**Important: update does NOT change the vector.**
If you update a paper's title, the embedding that was inserted at creation time stays as-is.
To update the vector you must delete and re-insert, or use `col.data.replace()` which
accepts both new properties and a new vector.

**UUID as primary key:**
Every object in Weaviate gets a UUID at insert time. This is your handle for
read/update/delete — Weaviate doesn't use your `arxiv_id` property as a primary key.
If you need to look up by your own ID, you'd use a filtered `near_vector` or `bm25` query.

---

### Step 4 — `hybrid_search.py`

**What it does:** Runs the same query through dense, sparse, and hybrid modes and prints
results side by side — so you can see what each mode gets right and wrong.

**Three modes:**

**Dense (near_vector):**
- Converts query to a vector, finds nearest vectors
- Captures semantic meaning — paraphrases, synonyms, related concepts all match
- Blind to exact keywords — "BERT" and "GPT" may be far apart in embedding space
  even though they're both transformer models

**Sparse (BM25):**
- Term Frequency × Inverse Document Frequency — classic information retrieval
- Counts how often query words appear in a document, weighted by how rare those
  words are across the whole corpus
- Exact match strength — catches specific model names, acronyms, proper nouns
- Blind to meaning — "neural network" won't match "deep learning" unless those
  exact words appear

**Hybrid:**
```
final_score = alpha × dense_score + (1 - alpha) × sparse_score
```
- `alpha=0.0` → pure BM25
- `alpha=0.5` → balanced (good default)
- `alpha=1.0` → pure dense

**When you saw this live (Step 4 output with query "attention mechanism transformers"):**
```
Dense  → cognitive science papers  (matched meaning of "attention" broadly)
BM25   → NLP papers                (matched exact words in ML context)
Hybrid → mix of both               (balanced result)
```
This is the best real-world illustration of why hybrid exists: dense is too liberal
(matches wrong-domain papers), BM25 is too literal (misses paraphrases), hybrid
finds the middle ground.

**For BM25 to work**, properties need `index_searchable=True` in the schema:
```python
wvc.config.Property(name="abstract", data_type=wvc.config.DataType.TEXT, index_searchable=True)
```
This tells Weaviate to build an inverted index on that field. Without it, `bm25`
queries return nothing.

---

### Step 5 — `multilingual_search.py`

**What it does:** Embeds 2K papers with a multilingual model, inserts into Weaviate,
then queries the same concept in 5 languages — and shows that all 5 return similar results.

**Why a different model?**
`all-MiniLM-L6-v2` was trained only on English. It has no concept of Spanish or French.
`paraphrase-multilingual-MiniLM-L12-v2` was trained on parallel text — pairs of sentences
meaning the same thing across 50+ languages. It maps them to the same region of vector space.

**Why re-embed instead of reusing vectors.npy?**
The two models produce *incompatible* vector spaces. Same 384 dimensions, but the
coordinate axes mean completely different things. Mixing them would be like using
a GPS coordinate from Tokyo to navigate in London — same format, wrong reference frame.

**The multilingual model's training:**
```
Input during training:
  ("attention mechanism", "mecanismo de atención")     ← English + Spanish
  ("graph neural network", "réseau de neurones en graphe")  ← English + French
  ...

Loss function forces: embed("attention mechanism") ≈ embed("mecanismo de atención")

Result: queries in any language land near the right English papers.
```

**What to look for in the output:**
If the model works correctly, all 5 language queries should return largely
overlapping sets of papers. The distances may vary slightly between languages
(some languages have better coverage in training data than others), but the
top results should be recognizably about the same topic.

**Model specs:**
- `paraphrase-multilingual-MiniLM-L12-v2`
- 50+ languages
- 384 dimensions (same as Phase 1's English model)
- Slightly lower English-only accuracy vs `all-MiniLM-L6-v2` — the price of multilingual coverage

---

## Benchmark results

Run `benchmark_weaviate.py` and record your numbers here:

```
Three-way benchmark  (recall@10, 20K vectors, 50 queries)
M=16  ef_construction=200  ef_search=50

Method                    Setup     Latency (ms)   Speedup   Recall@10
────────────────────────────────────────────────────────────────────────
FlatIndex (exact)            —              —          1.0x       1.000
HNSW (hand-built)                 build        ms      x       
Weaviate                    insert        ms      x       
────────────────────────────────────────────────────────────────────────
```

Phase 1 reference (for comparison):
```
 Size  │ Flat (ms) │ HNSW build (s) │ HNSW (ms) │ Speedup │ Recall@10
───────┼───────────┼────────────────┼───────────┼─────────┼──────────
20,000 │     27.08 │          79.24 │      1.02 │   26.5x │     0.996
```

---

## Environment note (SSL on macOS)

Weaviate embedded downloads its Go binary on first run over HTTPS. Python 3.14 on
macOS doesn't use system certificates by default, causing an SSL error. Every script
in this folder patches this automatically at the top:

```python
import certifi, os
os.environ.setdefault("SSL_CERT_FILE", certifi.where())
```

This must come **before** `import weaviate`. No other action needed.

