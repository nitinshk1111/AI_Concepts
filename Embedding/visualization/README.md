# PCA vs t-SNE — what's actually different

Reference notes for `visualize_embeddings.py` (PCA) and
`visualize_embeddings_tsne.py` (t-SNE). Both scripts take the exact
same 16 sentences, embed them with the same model
(`all-MiniLM-L6-v2`, 384 numbers per sentence), and then squash those
384 numbers down to 2 so they can be plotted. **The only difference
between the two scripts is which squashing algorithm they use.**

## Why we need to squash at all

A sentence embedding has 384 dimensions. Humans can only look at 2D
(flat) or 3D plots. So before anything can be drawn, 384 numbers per
sentence have to become 2 numbers per sentence — that step is called
**dimensionality reduction**, and PCA and t-SNE are two different ways
to do it.

## The core difference: one calculation vs. many small steps

**PCA does exactly one step:**
1. Look at all the data (all 384 dimensions)
2. Run one math formula to find the 2 directions along which the data
   spreads out the most
3. Project every point onto just those 2 directions
4. Done — no adjustments after this

This is what makes PCA **rigid**: one fixed rule, applied once, the
same way for every point. Whatever that single calculation produces is
final, even if it happens to place two unrelated sentences near each
other by accident.

**t-SNE does hundreds of small steps:**
1. Start with a rough, almost random 2D layout
2. Check: does this layout keep the true neighbors (from the original
   384D space) close together?
3. Nudge a few points slightly to make it a bit better
4. Repeat steps 2-3 hundreds of times
5. Stop once it stops improving much

This is what makes t-SNE **flexible** — it keeps correcting itself
over many rounds instead of committing to one answer immediately.

### Analogy

- **PCA** = measuring a room once with a tape measure and drawing the
  floor plan from that single measurement. Fast, but if the
  measurement was slightly off, that mistake is permanently baked in.
- **t-SNE** = sketching the floor plan, stepping back to check it,
  erasing and redrawing the parts that look wrong, checking again —
  repeated until it actually looks right.

## What each one preserves

| | PCA | t-SNE |
|---|---|---|
| Preserves | overall spread/variance across the whole dataset | local neighbor relationships (who's close to whom) |
| Method | one straight-line (linear) projection | iterative, non-linear rearranging |
| Deterministic? | yes — same input always gives the same output | not by default — results can vary slightly per run unless you fix `random_state` |
| Axes have real meaning? | yes — each axis is a real direction in the original space, and you can measure how much information it captures (`explained_variance_ratio_`) | no — only relative distances/neighbor groupings mean anything; the axes themselves are not interpretable |
| Distance between two far-apart clusters meaningful? | yes | **no** — only "close = similar" is meaningful, not "how much farther" |
| Speed | fast, single calculation | slower, many iterations |
| Code | `sklearn.decomposition.PCA` | `sklearn.manifold.TSNE` |

## What we actually saw in this project

Both scripts embedded the same 16 sentences across 4 topics (animals,
sports, finance, weather). The results were different:

**PCA's plot:** finance and weather separated cleanly, but **animals
and sports overlapped** — e.g. `"He trains every morning for the
marathon."` (sports) and `"The cat sat on the mat."` (animals) landed
almost on top of each other. Only ~25% of the original 384D structure
survived the squash to 2D (`explained_variance_ratio_` summed to
25.3%), and the animals-vs-sports distinction happened to live in the
part that got thrown away.

**t-SNE's plot:** all 4 topics ended up in their own separate regions,
with clear gaps between every pair. Because t-SNE optimizes directly
for "keep true neighbors close" using the full 384D neighbor
information (not just the top 2 variance directions), it recovered a
distinction that PCA's rigid 2-axis projection missed.

## Takeaway

Reach for **PCA** first — it's fast, deterministic, and good enough to
sanity-check your embeddings. Reach for **t-SNE** (or UMAP, a similar
but usually faster alternative) when you want a more visually faithful
picture of how things actually cluster, and can accept it being
slower and non-deterministic, with axes that carry no independent
meaning.
