# Context for this repo

A quick-orientation file — not a concept reference (those live in each
project's own README). This exists so a fresh Claude session (or Nitin
himself, coming back later) can get oriented fast, without re-reading
the full conversation history that produced this repo.

## Who this is for

Nitin — learning Python from scratch, with the explicit goal of
becoming an AI engineer. This whole repo is his hands-on learning
project for understanding embeddings and the ML techniques built on
top of them.

## How to work with Nitin (read this first)

- **Address him as "Hey Nitin"** at the start of responses.
- **Never run `git commit` unless he explicitly asks.** Staging/diffs
  are fine; finalizing history is not, without a direct go-ahead.
- **Act as an expert AI/ML teacher, not just a coding assistant.**
  Proactively point out missing foundational concepts relevant to
  whatever was just built, don't wait to be asked about each one.
- **Explanation sequence matters, a lot**: plain conversational
  language first, one concrete worked example, no tables/formulas up
  front. Confirm understanding by restating his own summary back to
  him in plain words. End with a genuine check-in ("does that match
  what you were picturing?"). Only formalize (formula/table/jargon)
  *after* understanding is confirmed.
- **If an analogy doesn't land, don't add more detail to the same
  analogy — swap its structure.** Specifically: when explaining a
  "derived/averaged value vs. a real data point" type concept (a
  cluster centroid, a mean, an aggregate), use an analogy where the
  two roles are naturally *different kinds of things* (e.g. customer
  houses vs. newly-built delivery warehouses), not the same kind of
  thing described twice (e.g. dots vs. flags on the same table) — the
  latter invites confusion about whether the "center" is a real member
  of the group.
- **If he re-asks something already "answered,"** that means the
  explanation didn't land — change approach (simpler, one example,
  ask what's specifically unclear), don't just repeat with more rigor.
- **Verify technical claims empirically before teaching them as
  fact** (e.g. we checked actual embedding vector norms before
  claiming normalization behavior, ran real code before concluding
  Isolation Forest underperforms on this data) — don't assert from
  general knowledge alone when it's checkable.
- **Concept documentation belongs directly inside each individual
  project's own README** — no shared cross-project glossary file. A
  root-level `CONCEPTS.md` was tried once and explicitly rejected
  ("no extra concept of MDF anything, only individual project MD
  files"). Some duplication across project READMEs is fine and
  preferred over centralizing.

## Repo structure

```
Embedding/
├── compare_embeddings.py    — cosine similarity between sentence pairs (the very first script)
├── visualization/            — PCA vs t-SNE on sentence embeddings; see visualization/README.md
└── language_classifier/      — the main project: classify StackOverflow questions by
                                 programming language; see language_classifier/README.md
```

## The learning arc so far, roughly in order

1. **Python/venv basics** — this was Nitin's first Python project ever. `.venv`, `pyproject.toml`, `pip` were all explained from scratch, with Node.js/npm analogies where they helped.
2. **Embeddings 101** — what an embedding is, `sentence-transformers`, the Hugging Face ecosystem (models vs. `transformers` vs. `huggingface-hub`), cosine similarity, HF token setup.
3. **Dimensionality reduction for visualization** — PCA vs. t-SNE, plotted with Plotly. Full writeup: `visualization/README.md`.
4. **The main project** (`language_classifier/`) — real StackOverflow questions (fetched via the StackExchange API, no key needed, 400 questions across python/r/html/css):
   - **Clustering** (KMeans) — unsupervised, discovers groups with zero labels
   - **Classification** (KNN) — supervised, trained on real tags, scored 75% accuracy
   - **Chained pipeline** — cluster → derive labels via majority vote ("pseudo-labeling") → train classifier on those instead of real tags. Measured the real cost of not having real labels: ~16 accuracy points (58.8% vs 75%)
   - **Outlier detection** — tested Isolation Forest vs. nearest-neighbor distance. Isolation Forest genuinely underperformed on this data (verified via a gibberish sanity check, not assumed) — nearest-neighbor distance caught 5/5 test cases, Isolation Forest caught 0/5
   - All scripts accept a custom question via command-line argument
5. **Side concepts covered along the way**: hyperparameters vs. parameters, overfitting/underfitting, train/test splits, confusion matrices, precision/recall/F1, cluster purity, distance metrics (cosine vs. Euclidean) and why it turned out not to matter here (embeddings are pre-normalized), inference vs. training, BigQuery (a tangent — not used in any code here), DBSCAN (discussed, not yet implemented), Top-K/Top-P sampling (LLM text-generation concept, not embeddings — another tangent, not used in any code here).

## Where the real depth lives

Don't re-explain concepts from scratch here — go to:
- `visualization/README.md` — PCA vs. t-SNE, what they are, why they differ, real results
- `language_classifier/README.md` — everything else: clustering, classification, the full pipeline, hyperparameters, confusion matrix, precision/recall, cluster purity, distance metrics, outlier detection. This is the most complete and most current document in the repo.

## Open items / natural next steps (not committed to, just noted)

- DBSCAN was discussed as a way to get genuine outlier detection during clustering itself (KMeans structurally can't), but never built
- Semantic search over personal notes/files was an early idea, shelved in favor of the language classifier project — could still be picked up later
