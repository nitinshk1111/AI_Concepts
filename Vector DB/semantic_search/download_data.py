"""
Fetches 20K ArXiv paper abstracts from HuggingFace datasets.
Saves: semantic_search/data/arxiv_papers.csv
Columns kept: id, title, abstract, categories
"""

import csv
import os
import sys

from datasets import load_dataset
from tqdm import tqdm

OUTPUT_PATH = os.path.join(os.path.dirname(__file__), "data", "arxiv_papers.csv")
NUM_PAPERS  = 20_000

# Keep only AI/ML-relevant ArXiv categories for meaningful semantic search
AI_CATEGORIES = {
    "cs.LG", "cs.AI", "cs.CL", "cs.CV", "cs.NE",
    "cs.IR", "cs.RO", "stat.ML",
}


def _is_ai_paper(categories) -> bool:
    # categories can be a list or a space-separated string depending on dataset version
    if isinstance(categories, list):
        cats = categories
    else:
        cats = str(categories).split()
    return any(cat.strip() in AI_CATEGORIES for cat in cats)


def main():
    print("Downloading ArXiv abstracts from HuggingFace (filtering for AI/ML papers)...")
    # Stream the full dataset so we can filter without loading all 1.9M rows into RAM
    dataset = load_dataset(
        "gfissore/arxiv-abstracts-2021",
        split="train",
        streaming=True,
    )

    saved   = 0
    scanned = 0
    print(f"Saving up to {NUM_PAPERS} AI/ML papers to {OUTPUT_PATH}")
    with open(OUTPUT_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "title", "abstract", "categories"])
        writer.writeheader()

        pbar = tqdm(total=NUM_PAPERS, desc="AI/ML papers saved")
        for row in dataset:
            scanned += 1
            cats     = row.get("categories", "")
            if not _is_ai_paper(cats):
                continue
            abstract = (row.get("abstract") or "").replace("\n", " ").strip()
            title    = (row.get("title")    or "").replace("\n", " ").strip()
            if not abstract or not title:
                continue
            cats_str = " ".join(cats) if isinstance(cats, list) else str(cats)
            writer.writerow({
                "id":         row.get("id", ""),
                "title":      title,
                "abstract":   abstract,
                "categories": cats_str,
            })
            saved += 1
            pbar.update(1)
            if saved >= NUM_PAPERS:
                break
        pbar.close()

    print(f"Scanned {scanned:,} papers → saved {saved:,} AI/ML papers.")

    print("Done.")


if __name__ == "__main__":
    main()
