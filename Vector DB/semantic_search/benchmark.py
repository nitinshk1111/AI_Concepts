"""
Benchmarks FlatIndex (brute force) vs HNSWIndex on recall@10 and query latency.

Reads: semantic_search/embeddings/vectors.npy
Tests across dataset sizes: 1K, 5K, 10K, 20K vectors.
Prints a summary table.

Run after build_embeddings.py.
"""

import os
import sys
import time

import numpy as np

BASE = os.path.dirname(__file__)
sys.path.insert(0, BASE)

from hnsw.distance   import cosine_distance
from hnsw.flat_index import FlatIndex
from hnsw.hnsw_index import HNSWIndex

VECS_PATH   = os.path.join(BASE, "embeddings", "vectors.npy")
SIZES       = [1_000, 5_000, 10_000, 20_000]
N_QUERIES   = 50    # queries per size
K           = 10    # recall@K
EF_SEARCH   = 50    # HNSW ef at query time
M           = 16
EF_CONSTRUCTION = 200


def compute_recall(true_results, approx_results, k):
    true_ids  = {idx for _, idx in true_results[:k]}
    approx_ids = {idx for _, idx in approx_results[:k]}
    return len(true_ids & approx_ids) / k


def benchmark_size(vectors: np.ndarray, queries: np.ndarray, size: int):
    subset  = vectors[:size]
    results = {"size": size}

    # ── Flat index ──────────────────────────────────────────────────────────
    flat = FlatIndex(distance_fn=cosine_distance)
    flat.add_batch(subset)

    flat_times = []
    flat_results_all = []
    for q in queries:
        t0 = time.perf_counter()
        r  = flat.search(q, k=K)
        flat_times.append(time.perf_counter() - t0)
        flat_results_all.append(r)

    results["flat_latency_ms"] = (sum(flat_times) / len(flat_times)) * 1000

    # ── HNSW index ──────────────────────────────────────────────────────────
    t0   = time.perf_counter()
    hnsw = HNSWIndex(M=M, ef_construction=EF_CONSTRUCTION, distance_fn=cosine_distance)
    hnsw.add_batch(subset)
    results["hnsw_build_s"] = time.perf_counter() - t0

    hnsw_times = []
    recalls    = []
    for q, true_r in zip(queries, flat_results_all):
        t0    = time.perf_counter()
        approx_r = hnsw.search(q, k=K, ef=EF_SEARCH)
        hnsw_times.append(time.perf_counter() - t0)
        recalls.append(compute_recall(true_r, approx_r, K))

    results["hnsw_latency_ms"] = (sum(hnsw_times) / len(hnsw_times)) * 1000
    results["recall_at_k"]     = sum(recalls) / len(recalls)
    results["speedup"]         = results["flat_latency_ms"] / results["hnsw_latency_ms"]

    return results


def main():
    if not os.path.exists(VECS_PATH):
        sys.exit(f"Missing: {VECS_PATH}\nRun build_embeddings.py first.")

    all_vectors = np.load(VECS_PATH)
    max_needed  = max(SIZES)

    if len(all_vectors) < max_needed:
        sys.exit(f"Need at least {max_needed} vectors, got {len(all_vectors)}.")

    # Fixed query set — drawn from the top of the dataset so they overlap
    # all test subsets; recall is still meaningful because both flat and HNSW
    # see the same corpus and we compare them against each other
    np.random.seed(42)
    query_indices = np.random.choice(len(all_vectors), size=N_QUERIES, replace=False)
    queries = all_vectors[query_indices]

    print(f"Benchmarking FlatIndex vs HNSWIndex  (recall@{K}, ef_search={EF_SEARCH})")
    print(f"Queries: {N_QUERIES}  |  M={M}  ef_construction={EF_CONSTRUCTION}\n")

    print(f"{'Size':>8} | {'Flat (ms)':>10} | {'HNSW build (s)':>15} | {'HNSW (ms)':>10} | {'Speedup':>8} | {'Recall@10':>10}")
    print("-" * 80)

    for size in SIZES:
        r = benchmark_size(all_vectors, queries, size)
        print(
            f"{r['size']:>8,} | "
            f"{r['flat_latency_ms']:>10.2f} | "
            f"{r['hnsw_build_s']:>15.2f} | "
            f"{r['hnsw_latency_ms']:>10.2f} | "
            f"{r['speedup']:>7.1f}x | "
            f"{r['recall_at_k']:>10.3f}"
        )

    print("\nDone.")


if __name__ == "__main__":
    main()
