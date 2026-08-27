"""
Downloads ~25,000 recent AI/ML ArXiv papers (2018-2024) from HuggingFace.
Saves: RAG/data/arxiv_papers.csv
Columns: id, title, abstract, categories

Filtering for 2018+ papers means this dataset covers the modern era:
BERT (2018), GPT-2 (2019), GPT-3 (2020), LLMs, RAG, diffusion models, etc.
"""

import csv
import os

import certifi
os.environ.setdefault("SSL_CERT_FILE", certifi.where())

from datasets import load_dataset
from tqdm import tqdm

OUTPUT_DIR  = os.path.join(os.path.dirname(__file__), "data")
OUTPUT_PATH = os.path.join(OUTPUT_DIR, "arxiv_papers.csv")
NUM_PAPERS  = 25_000

AI_CATEGORIES = {
    "cs.LG", "cs.AI", "cs.CL", "cs.CV", "cs.NE",
    "cs.IR", "cs.RO", "stat.ML",
}

# ArXiv new-format IDs: YYMM.NNNNN — prefix 18xx = 2018, 19xx = 2019, etc.
RECENT_PREFIXES = {"18", "19", "20", "21"}


def _is_recent(arxiv_id: str) -> bool:
    return "." in arxiv_id and arxiv_id[:2] in RECENT_PREFIXES


def _is_ai_paper(categories) -> bool:
    cats = categories if isinstance(categories, list) else str(categories).split()
    return any(cat.strip() in AI_CATEGORIES for cat in cats)


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("Streaming ArXiv dataset — filtering for 2018-2021 AI/ML papers...")
    print("(This covers BERT, GPT-2, GPT-3, transformers, RL, computer vision, etc.)")
    dataset = load_dataset(
        "gfissore/arxiv-abstracts-2021",
        split="train",
        streaming=True,
    )

    saved   = 0
    scanned = 0

    with open(OUTPUT_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "title", "abstract", "categories"])
        writer.writeheader()

        pbar = tqdm(total=NUM_PAPERS, desc="Recent AI/ML papers saved")
        for row in dataset:
            scanned += 1
            arxiv_id = row.get("id", "")

            if not _is_recent(arxiv_id):
                continue
            if not _is_ai_paper(row.get("categories", "")):
                continue

            abstract = (row.get("abstract") or "").replace("\n", " ").strip()
            title    = (row.get("title")    or "").replace("\n", " ").strip()
            if not abstract or not title:
                continue

            cats = row.get("categories", "")
            cats_str = " ".join(cats) if isinstance(cats, list) else str(cats)

            writer.writerow({
                "id":         arxiv_id,
                "title":      title,
                "abstract":   abstract,
                "categories": cats_str,
            })
            saved += 1
            pbar.update(1)
            if saved >= NUM_PAPERS:
                break
        pbar.close()

    print(f"\nScanned {scanned:,} papers → saved {saved:,} recent AI/ML papers.")
    print(f"Saved to: {OUTPUT_PATH}")
    print("\nNext step: run build_embeddings.py")


if __name__ == "__main__":
    main()
