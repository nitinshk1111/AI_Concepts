# Concepts Learned — Semantic Search Project

Everything covered while building this project, in the order it came up.
Code references point to the actual files so you can read the implementation
alongside the concept.

---

## 1. Distance Metrics — what does "similar" mean?

Before you can search for similar vectors, you need a way to measure how
far apart two vectors are. Three options, each with different tradeoffs.

### Cosine Distance
```
cosine_distance(a, b) = 1 - (a · b) / (||a|| × ||b||)
```
- Range: 0 (identical) → 2 (opposite)
- Cares about **direction only**, not magnitude
- Two sentences with different lengths but same meaning → same direction → low cosine distance
- **Default choice for text/embedding search** — used in this project
- Code: [`hnsw/distance.py`](hnsw/distance.py)

### Euclidean Distance
```
euclidean_distance(a, b) = √Σ(aᵢ - bᵢ)²
```
- Straight-line distance between two points
- Cares about both direction AND magnitude
- Common for image embeddings where scale matters

### Dot Product Distance
```
dot_product_distance(a, b) = -(a · b)   ← negated so lower = more similar
```
- Fast: no square root, no division
- When vectors are L2-normalised (magnitude = 1): dot product == cosine similarity
- Production systems normalise vectors upfront so they can use the cheaper dot product at query time
- `build_embeddings.py` uses `normalize_embeddings=True` for exactly this reason

---

## 2. Brute-Force KNN — exact but slow

**K-Nearest Neighbors (KNN):** given a query vector, find the K vectors
in the dataset that are closest to it by distance.

Brute-force approach:
```
for every stored vector:
    compute distance to query
sort all distances
return top K
```

Time complexity: **O(n × d)** per query — n vectors, d dimensions each.
No index, no structure, no shortcuts — must touch every vector every time.

### Why it doesn't scale

| Dataset size | Query time (384 dims) |
|---|---|
| 1,000 | 1.39ms |
| 5,000 | 6.74ms |
| 10,000 | 13.69ms |
| 20,000 | 27.08ms |

Query time grows linearly with data. At 200M vectors → ~5 minutes per query.
Unusable for any real application.

Code: [`hnsw/flat_index.py`](hnsw/flat_index.py) — used as ground truth for recall measurement.

---

## 3. ANN — Approximate Nearest Neighbor

**The core idea:** instead of checking every vector, organise them into
neighbourhoods upfront (indexing step, done once) so that at query time
you only check a small, relevant slice of the data.

```
Exact KNN (brute force):
  query → compare to all n vectors → top K

ANN:
  BUILD (once): organise vectors into structure
  QUERY: narrow to promising region → search locally → top K
```

You trade a tiny amount of accuracy for a massive speed gain.

**"Approximate"** means: you might occasionally miss the single mathematically
closest vector if it sits just outside the region you searched. In practice
this miss rate is very small — recall@10 of 0.99+ is achievable.

Four ANN families:

| Type | How it organises | Example |
|---|---|---|
| Tree-based | Binary space partitions | KD-tree, Ball tree |
| Hash-based | Random projections into buckets | LSH |
| Quantization | Compress vectors into codebooks | IVF, PQ, IVFPQ |
| Graph-based | Connect nearby vectors as graph nodes | **HNSW** ← used here |

---

## 4. NSW — Navigable Small World (HNSW's foundation)

Before HNSW there was NSW — a single flat graph with no layers.

**Construction:** as each new node arrives, connect it to its M nearest
existing nodes. Bidirectional edges — if A connects to B, B also connects to A.

**Search:** start from a random entry node, look at its neighbours, move to
whichever neighbour is closest to the query. Repeat until no neighbour is
an improvement. This is called **greedy search**.

**The problem with NSW:** greedy search can get stuck in a local optimum.
If you start from the wrong entry point, you might converge to a good-but-not-best
match and stop — never finding the true nearest neighbour on the other side
of the graph. HNSW was built to fix this.

---

## 5. HNSW — Hierarchical Navigable Small World

NSW + multiple layers. The hierarchy gives you fast long-range jumps at
the top, precise local search at the bottom.

```
Layer 2 (sparse):   A ─────────────────── I ──────── F
Layer 1 (medium):   A ───── C ─── I ─── F ─── J
Layer 0 (dense):    A ─ B ─ C ─ D ─ E ─ F ─ G ─ H ─ I ─ J
                    (all nodes, all connections)
```

Think: plane → train → taxi. Each layer change = zoom in one level.

### Layer assignment

When a new node is inserted, its maximum layer is sampled randomly:
```
level = floor(-ln(uniform(0,1)) × mL)    where mL = 1/ln(M)
```

Geometric distribution → most nodes land on layer 0, exponentially
fewer reach higher layers. This happens automatically — no human decides.

```
With M=16:
  P(layer 0 only) ≈ 93%
  P(reaches layer 1) ≈ 6%
  P(reaches layer 2) ≈ 0.4%
```

### Key parameters

| Parameter | What it controls | Typical value |
|---|---|---|
| `M` | Max connections per node (layers 1+) | 16 |
| `M0` | Max connections at layer 0 (= 2×M, denser) | 32 |
| `ef_construction` | Beam width during index build — higher = better graph, slower build | 200 |
| `ef_search` | Beam width during query — higher = better recall, slower query | 50 |

### Inserting a node (add)

```
Phase 1 (fast descent, ef=1):
  From max_layer → level+1
  Greedy: just find one closest node per layer to use as entry point

Phase 2 (careful insertion, ef=ef_construction):
  From min(level, max_layer) → layer 0
  Beam search: find M nearest neighbours per layer
  Add bidirectional edges
  Prune any neighbour that now exceeds M connections
```

### Querying (search)

```
1. Start at global entry point (top layer)
2. Greedy descent from top → layer 1, ef=1
3. At layer 0: beam search with ef=ef_search
4. Return top K from result set
```

### The beam search internals

Two heaps work together at each layer:

```
candidates  (min-heap)  → nodes to explore next, closest at top
W           (max-heap)  → best results so far, size = ef, farthest at top
```

Loop:
1. Pop closest unexplored node `c` from candidates
2. If `c` is farther than the worst node in W → **STOP**
   (nothing in candidates can improve W anymore)
3. Check all of `c`'s neighbours:
   - If closer than worst in W (or W not full): add to both heaps
   - If W exceeds ef: evict the farthest

`ef_search` is the size of W — it directly controls how many candidates
you evaluate before stopping. Larger W → higher recall → slower query.

Code: [`hnsw/hnsw_index.py`](hnsw/hnsw_index.py)

---

## 6. Recall@K — measuring ANN accuracy

```
recall@K = |ANN top-K ∩ FlatIndex top-K| / K
```

Example:
```
Brute-force top-5:  {A, B, C, D, E}    ← ground truth
HNSW top-5:         {A, B, C, D, F}    ← missed E, returned F

recall@5 = 4/5 = 0.80
```

In this project (20K vectors, 384 dims, M=16, ef_search=50):
```
recall@10 = 0.996   → HNSW almost never misses a true match
```

Production target: **recall@10 ≥ 0.95** with query latency < 10ms.
Tune `ef_search` up to improve recall; tune it down to reduce latency.

---

## 7. Serialization with Pickle

Building the HNSW index over 20K vectors takes ~80 seconds.
You don't want to do that on every search. Solution: pickle.

Pickle serializes any Python object to bytes and saves it to a `.pkl` file.
On the next run, `pickle.load()` restores the object exactly as it was.

```python
# Save
with open("hnsw_index.pkl", "wb") as f:
    pickle.dump(index, f)

# Load
with open("hnsw_index.pkl", "rb") as f:
    index = pickle.load(f)
```

Limitation: Python-only, version-sensitive. If you change `HNSWIndex`'s
internal structure, delete the `.pkl` and rebuild.

Code: [`search.py`](search.py) — `load_or_build_index()` function.

---

## 8. Sentence Embeddings (recap from Embedding project)

Text is not directly comparable — "fast car" and "quick automobile"
share no words but mean the same thing. A sentence embedding model
converts text into a fixed-size vector where **direction = meaning**.

Model used: `all-MiniLM-L6-v2` from sentence-transformers.
Output: 384-dimensional float32 vector per sentence.

`normalize_embeddings=True` in `build_embeddings.py` ensures every vector
has magnitude 1 (lives on a unit hypersphere), so cosine distance and
dot product give identical rankings — and dot product is cheaper.

Code: [`build_embeddings.py`](build_embeddings.py)

---

## 9. Benchmark results (this project)

Dataset: 20,000 ArXiv AI/ML paper abstracts, 384 dimensions.
M=16, ef_construction=200, ef_search=50, 50 queries averaged.

```
 Size  │ Flat (ms) │ HNSW build (s) │ HNSW (ms) │ Speedup │ Recall@10
───────┼───────────┼────────────────┼───────────┼─────────┼──────────
 1,000 │      1.39 │           2.43 │      0.61 │    2.3x │     0.998
 5,000 │      6.74 │          16.31 │      0.91 │    7.4x │     1.000
10,000 │     13.69 │          36.34 │      0.95 │   14.3x │     0.986
20,000 │     27.08 │          79.24 │      1.02 │   26.5x │     0.996
```

Key takeaway: HNSW query latency stays flat at ~1ms as data grows 20×.
Brute force scales linearly — it would be ~5 minutes per query at 200M vectors.
HNSW is what makes production vector databases possible.

---

## What's next (Phase 2)

Replace `HNSWIndex` with **Weaviate** — a production vector database that
runs HNSW internally. Same search interface, same dataset, but now:
- Weaviate manages the index (persistence, updates, deletions)
- You get filtered search, batch inserts, CRUD operations
- You can see how close your hand-built implementation gets to production performance

See the [course lessons](https://learn.deeplearning.ai/courses/vector-databases-embeddings-applications)
for the Weaviate API patterns.
