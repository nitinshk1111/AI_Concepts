# Phase 2 Concepts — Weaviate, Hybrid Search, Multilingual

Everything learned in Phase 2, written down so you never have to re-derive it.
For Phase 1 concepts (embeddings, cosine distance, KNN, HNSW) see
[../semantic_search/CONCEPTS.md](../semantic_search/CONCEPTS.md).

---

## 1. What Weaviate is

Weaviate is a **production vector database**. It stores objects (like documents)
alongside their vector representations and supports fast ANN search, filtering,
CRUD, and multiple search modes — all in one system.

Internally it runs **HNSW** (same algorithm you built in Phase 1) for the vector
index. The difference is that Weaviate's HNSW is:
- Written in Go (compiled, not interpreted Python)
- Persistent (survives restarts)
- Integrated with BM25, filtering, and a query planner
- Exposed over gRPC/HTTP, not in-process memory

Think of it this way:

```
Phase 1 (hand-built)       Phase 2 (Weaviate)
─────────────────────      ──────────────────────────────────────
HNSWIndex (Python)    →    Weaviate embedded (Go binary, local)
vectors.npy (file)    →    persistent collection on disk
search.py (custom)    →    client.query.near_vector()
no CRUD               →    full insert / read / update / delete
no BM25               →    BM25 + hybrid + multilingual support
```

---

## 2. Embedded vs Server mode

| Mode | How | When |
|------|-----|------|
| **Embedded** | Weaviate binary runs inside your Python process | Local dev, learning, no Docker needed |
| **Docker** | Weaviate runs as a container, your code connects via HTTP | Staging/prod, shared team instance |
| **Weaviate Cloud** | Managed cloud instance | Production, no infra to manage |

In this project: embedded mode. `weaviate.connect_to_embedded()` downloads the
binary on first use (~30MB) and starts it as a subprocess. You interact with it
over localhost HTTP — that's why query latency is slightly higher than Phase 1's
in-process HNSW.

---

## 3. Collection schema

A **collection** in Weaviate is like a table in SQL — it has a fixed schema
(property names and types) and stores objects with vectors.

```python
client.collections.create(
    name="ArxivPapers",
    vectorizer_config=wvc.config.Configure.Vectorizer.none(),  # we bring our own vectors
    vector_index_config=wvc.config.Configure.VectorIndex.hnsw(
        distance_metric=wvc.config.VectorDistances.COSINE,
    ),
    properties=[
        wvc.config.Property(name="title",    data_type=wvc.config.DataType.TEXT),
        wvc.config.Property(name="arxiv_id", data_type=wvc.config.DataType.TEXT),
    ],
)
```

Key parameters:
- `vectorizer_config=none` — you supply vectors yourself; Weaviate won't call an API to create them
- `distance_metric=COSINE` — must match how your embeddings were normalized (L2-normalized → cosine)
- `index_searchable=True` on a TEXT property — enables BM25 keyword search on that field

---

## 4. Batch insert and dynamic batching

Inserting 20K objects one-by-one over HTTP is slow (20K round trips). Batch
insert groups objects into larger chunks:

```python
with collection.batch.dynamic() as batch:
    for i in range(len(vectors)):
        batch.add_object(
            properties={"title": titles[i]},
            vector=vectors[i].tolist(),
        )
```

`dynamic()` automatically adjusts the batch size based on response times — it
starts small and grows until it finds the optimal throughput. Weaviate also
supports `fixed_size(batch_size=100)` if you want manual control.

---

## 5. CRUD in a vector database

A vector DB is not just a search index — you can modify it after the fact:

| Operation | Weaviate method | SQL equivalent |
|-----------|-----------------|----------------|
| Create | `col.data.insert(properties, vector)` | `INSERT INTO` |
| Read | `col.query.fetch_object_by_id(uuid)` | `SELECT WHERE id=` |
| Update | `col.data.update(uuid, properties)` | `UPDATE SET` |
| Delete | `col.data.delete_by_id(uuid)` | `DELETE WHERE id=` |

Weaviate assigns a UUID to every object at insert time. That UUID is your
primary key — you use it for read/update/delete.

Note: updating `properties` does NOT update the vector. To change the vector
you'd need to delete and re-insert, or use `col.data.replace()` which takes
both a new properties dict and a new vector.

---

## 6. Dense, Sparse, and Hybrid search

Three fundamentally different ways to find relevant documents:

### Dense search (near_vector)
- Compares query embedding to document embeddings
- Measures *semantic* similarity — "what does this mean?"
- Great for: vague/conceptual queries ("papers about graph learning")
- Blind to: exact keywords — "BERT" and "GPT" might be far apart

```python
col.query.near_vector(near_vector=query_vec.tolist(), limit=5)
```

### Sparse search (BM25)
- Counts term frequency weighted by inverse document frequency (TF-IDF family)
- Measures *lexical* similarity — "does this exact word appear?"
- Great for: exact terms ("ResNet-50", "BLEU score"), proper nouns
- Blind to: paraphrases — "neural network" won't match "deep learning" unless those words appear

```python
col.query.bm25(query="attention transformer", limit=5)
```

### Hybrid search
- Weighted combination of both scores
- Controlled by **alpha**: 0.0 = pure sparse, 1.0 = pure dense, 0.5 = balanced

```python
col.query.hybrid(query="...", vector=query_vec.tolist(), alpha=0.5, limit=5)
```

**When to use which:**

| Query type | Best mode | Example |
|-----------|-----------|---------|
| Conceptual / vague | Dense | "papers about how models learn from few examples" |
| Exact keyword | Sparse | "BERT" or "GPT-3" |
| Mixed (most real queries) | Hybrid | "BERT fine-tuning for NER" |

Alpha tuning: start at 0.5, push toward 1.0 if results feel "keyword-deaf",
push toward 0.0 if exact terms matter most.

---

## 7. Multilingual embeddings

Standard models like `all-MiniLM-L6-v2` are English-only — they embed text
from one language and the vectors don't generalise across languages.

**Multilingual models** (`paraphrase-multilingual-MiniLM-L12-v2` and similar)
are trained on parallel text across 50+ languages. The result: semantically
equivalent sentences in different languages end up at roughly the same position
in the vector space.

```
"attention mechanism" (EN)  ──┐
"mecanismo de atención" (ES) ──┼──→  ≈ same vector region
"mécanisme d'attention" (FR) ──┘
```

This means:
- Documents indexed in English
- Query in Spanish → relevant English papers still surface
- The model has learned that "attention" and "atención" carry the same meaning

Important constraints:
- The corpus and the query must both be embedded with the **same multilingual model**
- You can't mix vectors from different models — their spaces are incompatible
- Multilingual models are slightly less accurate than monolingual models on
  English-only tasks (they trade peak English performance for language coverage)

---

## 8. Why Weaviate latency is higher than hand-built HNSW

Phase 1 HNSW runs in-process — query is a Python function call, result is a
Python list. Zero network overhead.

Weaviate embedded runs as a separate Go process — query goes:

```
Python → gRPC call → Weaviate (Go) → HNSW search → gRPC response → Python
```

The HNSW search itself is faster (compiled Go vs interpreted Python), but the
gRPC serialization adds ~1–5ms overhead per query. At 20K vectors this often
dominates.

At scale (millions of vectors), Weaviate's HNSW completely outpaces the Python
implementation — the overhead becomes negligible relative to the speedup from
compiled code and optimized memory access.

---

## 9. Recall@K — how accuracy is measured in ANN search

Weaviate (like the hand-built HNSW) is an **approximate** nearest neighbour search.
It trades a small amount of accuracy for a large gain in speed. Recall@K is the
metric that quantifies exactly how much accuracy was traded away.

```
recall@K = |ANN top-K ∩ FlatIndex top-K| / K
```

**Example:**
```
FlatIndex top-5 (ground truth, exact): {A, B, C, D, E}
Weaviate top-5  (approximate):         {A, B, C, D, F}  ← missed E, returned F

recall@5 = 4/5 = 0.80
```

**How the benchmark uses it:**
`benchmark_weaviate.py` runs 50 queries through FlatIndex (exact brute force)
to get the ground truth top-10. It then runs the same queries through both
HNSW (hand-built) and Weaviate, and measures what fraction of the true top-10
each one finds.

```python
def compute_recall(true_results, approx_ids, k):
    true_ids = {idx for _, idx in true_results[:k]}
    return len(true_ids & set(approx_ids[:k])) / k
```

**Production target:** recall@10 ≥ 0.95 — meaning at least 9 of the true top-10
appear in the results. Both the hand-built HNSW and Weaviate achieve this comfortably
at 20K vectors.

**Recall vs latency tradeoff:**
- Higher recall → more candidates explored → slower query
- In the hand-built HNSW: controlled by `ef_search` parameter
- In Weaviate: controlled internally by its HNSW configuration (set at collection creation)

> Full derivation and Phase 1 numbers: [../semantic_search/CONCEPTS.md](../semantic_search/CONCEPTS.md#6-recallk--measuring-ann-accuracy)

---

## 10. The full picture — what you can now say

> "I implemented HNSW from scratch in Python, measured its recall and latency
> against brute-force, then replaced it with Weaviate — a production vector
> database that runs HNSW internally. I benchmarked both side-by-side, performed
> CRUD operations, compared dense vs sparse vs hybrid search, and built a
> multilingual search engine that accepts queries in any of 5 languages.
> I understand what each layer of the stack is doing — not just how to call the API."
