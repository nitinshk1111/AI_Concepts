"""
Step 3 — CRUD operations in Weaviate.

Demonstrates that a vector database is not just a search index — it supports
full Create / Read / Update / Delete like any database.

Run from the 'Vector DB/' root:
    python weaviate_search/weaviate_crud.py
"""

import os
import uuid

import certifi
os.environ.setdefault("SSL_CERT_FILE", certifi.where())

import numpy as np
import weaviate
import weaviate.classes as wvc

COLLECTION = "ArxivCRUD"
DIM = 384


def section(title):
    print(f"\n{'─' * 50}")
    print(f"  {title}")
    print(f"{'─' * 50}")


def main():
    client = weaviate.connect_to_embedded()
    try:
        # Fresh collection every run
        if client.collections.exists(COLLECTION):
            client.collections.delete(COLLECTION)

        col = client.collections.create(
            name=COLLECTION,
            vector_config=wvc.config.Configure.Vectors.self_provided(
                vector_index_config=wvc.config.Configure.VectorIndex.hnsw(
                    distance_metric=wvc.config.VectorDistances.COSINE,
                ),
            ),
            properties=[
                wvc.config.Property(name="arxiv_id", data_type=wvc.config.DataType.TEXT),
                wvc.config.Property(name="title",    data_type=wvc.config.DataType.TEXT),
                wvc.config.Property(name="year",     data_type=wvc.config.DataType.INT),
            ],
        )

        # ── CREATE ────────────────────────────────────────────────────────────
        section("CREATE — insert 3 seed papers + 1 paper we'll manipulate")

        seed_papers = [
            ("2301.00001", "Attention Is All You Need",         2017),
            ("2301.00002", "BERT: Pre-training of Deep Bidirectional Transformers", 2018),
            ("2301.00003", "GPT-3: Language Models are Few-Shot Learners", 2020),
        ]

        for arxiv_id, title, year in seed_papers:
            vec = np.random.default_rng(int(arxiv_id[-5:])).random(DIM).astype(np.float32)
            col.data.insert(
                properties={"arxiv_id": arxiv_id, "title": title, "year": year},
                vector=vec.tolist(),
            )
            print(f"  Inserted: [{arxiv_id}] {title} ({year})")

        # The paper we'll update and delete
        target_vec  = np.random.default_rng(99).random(DIM).astype(np.float32)
        target_uuid = col.data.insert(
            properties={
                "arxiv_id": "2301.00099",
                "title":    "Draft Paper: Work In Progress",
                "year":     2024,
            },
            vector=target_vec.tolist(),
        )
        print(f"  Inserted: [2301.00099] Draft Paper (uuid={target_uuid})")
        print(f"\n  Total in collection: {col.aggregate.over_all(total_count=True).total_count}")

        # ── READ ──────────────────────────────────────────────────────────────
        section("READ — fetch the draft paper by its UUID")

        obj = col.query.fetch_object_by_id(target_uuid)
        print(f"  arxiv_id : {obj.properties['arxiv_id']}")
        print(f"  title    : {obj.properties['title']}")
        print(f"  year     : {obj.properties['year']}")

        # ── UPDATE ────────────────────────────────────────────────────────────
        section("UPDATE — fix the title and year of the draft paper")

        col.data.update(
            uuid=target_uuid,
            properties={
                "title": "Weaviate CRUD: A Practical Guide",
                "year":  2025,
            },
        )
        updated = col.query.fetch_object_by_id(target_uuid)
        print(f"  title (after) : {updated.properties['title']}")
        print(f"  year  (after) : {updated.properties['year']}")

        # ── DELETE ────────────────────────────────────────────────────────────
        section("DELETE — remove the draft paper")

        print(f"  Count before delete: {col.aggregate.over_all(total_count=True).total_count}")
        col.data.delete_by_id(target_uuid)
        print(f"  Count after  delete: {col.aggregate.over_all(total_count=True).total_count}")

        # Confirm it's gone
        gone = col.query.fetch_object_by_id(target_uuid)
        print(f"  Fetch by UUID returns: {gone}  ✓ (None = deleted)")

        # ── SEARCH after CRUD ─────────────────────────────────────────────────
        section("SEARCH — near_vector over the 3 remaining papers")

        query_vec = np.random.default_rng(0).random(DIM).astype(np.float32)
        results   = col.query.near_vector(
            near_vector=query_vec.tolist(),
            limit=3,
            return_metadata=wvc.query.MetadataQuery(distance=True),
        )
        for obj in results.objects:
            print(f"  [{obj.properties['arxiv_id']}] {obj.properties['title']}  dist={obj.metadata.distance:.4f}")

    finally:
        client.close()


if __name__ == "__main__":
    main()
