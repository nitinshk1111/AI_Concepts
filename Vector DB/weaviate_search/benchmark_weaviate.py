"""
Step 2 — Three-way benchmark: FlatIndex vs HNSWIndex (Phase 1) vs Weaviate.

Uses the same 50 random queries and the full 20K corpus so results are
directly comparable to the Phase 1 benchmark table in semantic_search/benchmark.py.

Flat is the ground truth; recall@10 for both HNSW and Weaviate is measured
against it.

Run from the 'Vector DB/' root:
    python weaviate_search/benchmark_weaviate.py
"""

import os
import sys
import time

import certifi
os.environ.setdefault("SSL_CERT_FILE", certifi.where())

import numpy as np
import weaviate
import weaviate.classes as wvc

from paths import VECTORS_PATH, add_phase1_to_path

add_phase1_to_path()
from hnsw.distance   import cosine_distance
from hnsw.flat_index import FlatIndex
from hnsw.hnsw_index import HNSWIndex

COLLECTION    = "ArxivBenchmark"
N_QUERIES     = 50
K             = 10
M             = 16
EF_CONSTRUCTION = 200
EF_SEARCH     = 50


def compute_recall(true_results, approx_ids, k):
    true_ids = {idx for _, idx in true_results[:k]}
    return len(true_ids & set(approx_ids[:k])) / k


def run_flat(vectors, queries):
    flat = FlatIndex(distance_fn=cosine_distance)
    flat.add_batch(vectors)

    times, all_results = [], []
    for q in queries:
        t0 = time.perf_counter()
        r  = flat.search(q, k=K)
        times.append(time.perf_counter() - t0)
        all_results.append(r)

    return (sum(times) / len(times)) * 1000, all_results


def run_hnsw(vectors, queries, flat_results):
    print("  Building HNSW index...", end=" ", flush=True)
    t0   = time.perf_counter()
    hnsw = HNSWIndex(M=M, ef_construction=EF_CONSTRUCTION, distance_fn=cosine_distance)
    hnsw.add_batch(vectors)
    build_s = time.perf_counter() - t0
    print(f"{build_s:.1f}s")

    times, recalls = [], []
    for q, true_r in zip(queries, flat_results):
        t0 = time.perf_counter()
        r  = hnsw.search(q, k=K, ef=EF_SEARCH)
        times.append(time.perf_counter() - t0)
        approx_ids = [idx for _, idx in r]
        recalls.append(compute_recall(true_r, approx_ids, K))

    latency = (sum(times) / len(times)) * 1000
    recall  = sum(recalls) / len(recalls)
    return latency, recall, build_s


def run_weaviate(vectors, ids, queries, flat_results):
    client = weaviate.connect_to_embedded()
    try:
        if client.collections.exists(COLLECTION):
            client.collections.delete(COLLECTION)

        collection = client.collections.create(
            name=COLLECTION,
            vector_config=wvc.config.Configure.Vectors.self_provided(
                vector_index_config=wvc.config.Configure.VectorIndex.hnsw(
                    distance_metric=wvc.config.VectorDistances.COSINE,
                ),
            ),
            properties=[
                wvc.config.Property(name="orig_idx", data_type=wvc.config.DataType.INT),
            ],
        )

        print("  Inserting 20K vectors into Weaviate...", end=" ", flush=True)
        t0 = time.perf_counter()
        with collection.batch.dynamic() as batch:
            for i in range(len(vectors)):
                batch.add_object(
                    properties={"orig_idx": int(i)},
                    vector=vectors[i].tolist(),
                )
        insert_s = time.perf_counter() - t0
        print(f"{insert_s:.1f}s")

        times, recalls = [], []
        for q, true_r in zip(queries, flat_results):
            t0 = time.perf_counter()
            r  = collection.query.near_vector(
                near_vector=q.tolist(),
                limit=K,
                return_properties=["orig_idx"],
            )
            times.append(time.perf_counter() - t0)
            approx_ids = [obj.properties["orig_idx"] for obj in r.objects]
            recalls.append(compute_recall(true_r, approx_ids, K))

        latency = (sum(times) / len(times)) * 1000
        recall  = sum(recalls) / len(recalls)
        return latency, recall, insert_s
    finally:
        client.close()


def main():
    if not VECTORS_PATH.exists():
        sys.exit(f"Missing: {VECTORS_PATH}\nRun semantic_search/build_embeddings.py first.")

    all_vectors = np.load(VECTORS_PATH)
    ids         = list(range(len(all_vectors)))

    np.random.seed(42)
    query_indices = np.random.choice(len(all_vectors), size=N_QUERIES, replace=False)
    queries       = all_vectors[query_indices]

    print(f"Three-way benchmark  (recall@{K}, N={len(all_vectors):,}, queries={N_QUERIES})")
    print(f"M={M}  ef_construction={EF_CONSTRUCTION}  ef_search={EF_SEARCH}\n")

    print("[ Flat (ground truth) ]")
    flat_latency, flat_results = run_flat(all_vectors, queries)
    print(f"  Latency: {flat_latency:.2f}ms/query\n")

    print("[ Phase 1 HNSW (hand-built) ]")
    hnsw_latency, hnsw_recall, hnsw_build = run_hnsw(all_vectors, queries, flat_results)
    print(f"  Latency: {hnsw_latency:.2f}ms/query  |  Recall@{K}: {hnsw_recall:.3f}\n")

    print("[ Weaviate ]")
    wv_latency, wv_recall, wv_insert = run_weaviate(all_vectors, ids, queries, flat_results)
    print(f"  Latency: {wv_latency:.2f}ms/query  |  Recall@{K}: {wv_recall:.3f}\n")

    speedup_hnsw = flat_latency / hnsw_latency
    speedup_wv   = flat_latency / wv_latency

    print("─" * 72)
    print(f"{'Method':<22} {'Setup':>10} {'Latency (ms)':>14} {'Speedup':>9} {'Recall@10':>10}")
    print("─" * 72)
    print(f"{'FlatIndex (exact)':<22} {'—':>10} {flat_latency:>14.2f} {'1.0x':>9} {'1.000':>10}")
    print(f"{'HNSW (hand-built)':<22} {f'{hnsw_build:.1f}s build':>10} {hnsw_latency:>14.2f} {speedup_hnsw:>8.1f}x {hnsw_recall:>10.3f}")
    print(f"{'Weaviate':<22} {f'{wv_insert:.1f}s insert':>10} {wv_latency:>14.2f} {speedup_wv:>8.1f}x {wv_recall:>10.3f}")
    print("─" * 72)
    print("\nNote: Weaviate latency includes HTTP round-trip overhead (embedded = local).")
    print("      The recall difference reflects HNSW parameter choices inside Weaviate.")


if __name__ == "__main__":
    main()
