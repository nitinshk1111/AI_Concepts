# HNSW — From Scratch

This folder contains a pure-Python + NumPy implementation of the HNSW
(Hierarchical Navigable Small World) algorithm. No hnswlib, no faiss, no
Weaviate — every line of the algorithm is written here so you can see
exactly how it works.

---

## Why three files?

```
distance.py    ← what does "similar" mean mathematically?
flat_index.py  ← how does the exact (slow) answer work?
hnsw_index.py  ← how does HNSW find the approximate (fast) answer?
```

Understanding flat search first is important — HNSW is only meaningful
when you can measure how much accuracy it trades away vs. the exact baseline.

---

## distance.py — the atomic unit

Every search algorithm reduces to one question: **how far apart are two vectors?**

Three ways to measure this, each used in different contexts:

### Cosine distance
```
cosine_distance(a, b) = 1 - (a · b) / (||a|| × ||b||)
```
- Range: 0 (identical direction) to 2 (opposite direction)
- Does NOT care about vector magnitude — only direction
- Why this matters for embeddings: two sentences can have different word
  counts (different magnitudes) but the same meaning (same direction).
  Cosine distance captures meaning-similarity correctly. This is the
  default for all NLP/embedding use cases.

### Euclidean distance
```
euclidean_distance(a, b) = √Σ(aᵢ - bᵢ)²
```
- Straight-line distance between two points in vector space
- DOES care about magnitude
- Common in image embeddings and cases where scale matters

### Dot product distance
```
dot_product_distance(a, b) = -(a · b)
```
- Negated so that "closer = smaller number" (same convention as above)
- Higher dot product = more similar → negation makes it a distance
- Fast (no division, no square root) — used heavily in production when
  vectors are pre-normalized (then dot product = cosine similarity anyway)

**Which to use here?** Cosine distance — we're comparing sentence embeddings
from `sentence-transformers`, where direction = meaning.

---

## flat_index.py — brute-force KNN

### What it does
Stores all vectors in a plain Python list. At query time: compare the
query against every single stored vector, collect all distances, return
the K smallest.

### Why it exists in this project
- It is the **ground truth** — guaranteed to return the mathematically
  exact nearest neighbors, every time, no approximation
- Used in `benchmark.py` to compute **recall@K**: what fraction of the
  true top-K did HNSW actually find?
- Used to prove the scaling problem: run it on 1K, 10K, 100K vectors
  and watch the query time grow linearly

### Time complexity
- `add()`: O(1) — just append to list
- `search()`: O(n × d) — compare query to all n vectors, each d dimensions
- No index, no structure, no shortcuts — pure brute force

### When to use it in practice
Only for tiny datasets (<10K vectors) or as a correctness check. Never
in production at scale.

---

## hnsw_index.py — HNSW

### Core idea
Instead of comparing against every vector, HNSW builds a multi-layer
graph where:
- **Top layers** = sparse, long-range connections → fast navigation to
  the right region of the dataset
- **Bottom layer (layer 0)** = dense, short-range connections → precise
  local search once you're in the right region

A search starts at the top, greedy-descends layer by layer, and arrives
at layer 0 already near the answer — then does a careful local search.
Result: O(log n) comparisons instead of O(n).

### Key parameters

| Parameter | What it controls | Typical value |
|---|---|---|
| `M` | Max connections per node (layers 1+) | 16 |
| `M0` | Max connections at layer 0 (= 2×M) | 32 |
| `ef_construction` | Beam width during index build — wider = better graph quality, slower build | 200 |
| `ef_search` | Beam width during query — wider = higher recall, slower query | 50 |
| `mL` | Level multiplier = 1/ln(M) — controls probability of landing on higher layers | auto |

### Layer assignment
When a new node is inserted, its maximum layer is sampled from a
geometric distribution:

```
level = floor(-ln(uniform(0,1)) × mL)
```

- Most nodes land on layer 0 only
- Exponentially fewer nodes reach higher layers
- This is what creates the sparse-top / dense-bottom structure automatically
- No human decides which nodes go to which layer — randomness does it

### Graph construction (per node insert)
```
1. Sample level l for the new node
2. From top layer → l+1 : greedy descent, ef=1 (fast, find entry point)
3. From l → 0          : beam search, ef=ef_construction
                         select M nearest neighbors
                         add bidirectional edges
                         prune any neighbor that now exceeds M connections
4. If l > current max layer: new node becomes the global entry point
```

### Search (per query)
```
1. Start at global entry point (top layer)
2. From top layer → layer 1 : greedy descent, ef=1
3. At layer 0               : beam search with ef=ef_search
4. Return top K from the working set
```

### The beam search (greedy search) internals
Two heaps work together at each layer:

```
candidates  (min-heap)  → nodes to explore next, closest at top
W           (max-heap)  → current best results of size ef, farthest at top
```

Each step:
1. Pop the closest unexplored node `c` from `candidates`
2. If `c` is farther than the worst node currently in `W` → stop (nothing can improve)
3. Otherwise: check all of `c`'s neighbours
   - If a neighbour improves `W` (closer than worst, or `W` not full yet): add to both heaps
   - If `W` exceeds `ef`: evict the farthest

This is why `ef_search` controls recall: a larger `W` means you explore
more candidates before stopping, so you're less likely to miss a true
nearest neighbour that sits just outside the greedy path.

---

## hnsw_index.py — the full implementation

### Data structures inside HNSWIndex

```python
self.vectors    # List[np.ndarray]  — the actual embeddings, indexed by node_id
self.levels     # List[int]         — max layer each node lives on
self.graph      # List[dict]        — graph[layer][node_id] = set of neighbour node_ids
self.entry_point # int             — globally stored starting node (always on max_layer)
self.max_layer  # int              — highest layer that currently exists
```

### Constructor parameters

| Parameter | What it controls | Default |
|---|---|---|
| `M` | Max connections per node on layers 1+ | 16 |
| `M0` | Max connections on layer 0 (= 2×M) | 32 |
| `ef_construction` | Beam width during index build — wider = better graph, slower build | 200 |
| `mL` | Level multiplier = 1/ln(M) — set automatically, controls layer probability | auto |

`ef_search` is passed at query time, not at construction — so you can tune
it per-query without rebuilding the index.

---

### _sample_level() — how a node gets its layer

```python
level = int(-math.log(random.random()) * mL)
```

This samples from a geometric distribution. With M=16, mL = 1/ln(16) ≈ 0.36:

```
P(level = 0) ≈ 0.93   → ~93% of nodes live only on layer 0
P(level = 1) ≈ 0.06   → ~6% reach layer 1
P(level = 2) ≈ 0.004  → <1% reach layer 2
...
```

No human decides which nodes go where — the math naturally produces
a sparse top and dense bottom, which is exactly what creates the
highway-vs-local-roads structure.

---

### _search_layer() — beam search on one layer

This is the core routine. Called multiple times during both `add()` and `search()`.

```
Two heaps:
  candidates  (min-heap)  → nodes to explore next, closest at top
  W           (max-heap)  → working result set of size ef, farthest at top
                             stored as (-distance, node_id) to fake max-heap

Loop:
  1. Pop closest node c from candidates
  2. If dist(c) > worst node in W → STOP
     (nothing in candidates can possibly improve W)
  3. For each neighbour of c not yet visited:
       compute distance to query
       if better than worst in W (or W not full yet): add to both heaps
       if W exceeds ef: evict the farthest node
```

The stopping condition (step 2) is what makes HNSW sub-linear — once
the frontier is farther than your current best-ef results, you're done.

---

### add() — inserting a new node

Two phases:

**Phase 1: fast descent (ef=1)**
From `max_layer` down to `level+1`, run `_search_layer` with ef=1.
This is pure greedy — just find the single closest node at each layer
to use as the entry point for phase 2. No result set, no backtracking.

**Phase 2: careful insertion (ef=ef_construction)**
From `min(level, max_layer)` down to layer 0:
- Run beam search with ef=ef_construction to find best neighbours
- Connect new node to the M closest (bidirectional edges)
- If any existing neighbour now exceeds its connection limit → prune it
  (keep only its M closest connections using the same distance function)

Why bidirectional? Because search relies on graph traversal — a
one-way edge would make parts of the graph unreachable during queries.

Why prune? Unbounded connections would destroy the O(log n) guarantee
by turning the graph into something closer to brute force.

---

### search() — querying the index

```
1. Start at entry_point (lives on max_layer)
2. Greedy descent from max_layer → layer 1, ef=1 each time
   (fast: just find the closest node to use as entry into the next layer)
3. At layer 0: full beam search with ef=ef_search
   (careful: explore ef candidates, keep the best)
4. Return top K from the result set
```

The entire path from top layer to layer 0 typically visits
O(log n × M) nodes total — vs O(n) for brute force.

---

## recall@K — the metric that ties everything together

```
recall@K = |HNSW top-K ∩ FlatIndex top-K| / K
```

Example:
```
FlatIndex top-5 (ground truth):  {A, B, C, D, E}
HNSWIndex top-5 (approximate):   {A, B, C, D, F}   ← missed E, returned F instead

recall@5 = 4/5 = 0.80
```

Production target: **recall@10 ≥ 0.95** — meaning 95% of the true top-10
show up in HNSW's top-10. Tune `ef_search` upward to improve recall at
the cost of latency.

### Verified results on this implementation

Small sanity check (200 vectors, 16 dims, M=8):
```
recall@10 = 1.00  (exact match with brute force)
```

Full benchmark on 20K ArXiv paper embeddings (384 dims, M=16, ef_construction=200, ef_search=50):

```
 Size  │ Flat (ms) │ HNSW build (s) │ HNSW (ms) │ Speedup │ Recall@10
───────┼───────────┼────────────────┼───────────┼─────────┼──────────
 1,000 │      1.39 │           2.43 │      0.61 │    2.3x │     0.998
 5,000 │      6.74 │          16.31 │      0.91 │    7.4x │     1.000
10,000 │     13.69 │          36.34 │      0.95 │   14.3x │     0.986
20,000 │     27.08 │          79.24 │      1.02 │   26.5x │     0.996
```

Key observations:
- HNSW query latency stays roughly flat (~1ms) as dataset grows 20× — brute force scales linearly
- At 20K vectors: **26.5× faster** than brute force at **recall@10 = 0.996**
- Build cost is paid once; subsequent queries reuse the cached index (see `search.py`)
