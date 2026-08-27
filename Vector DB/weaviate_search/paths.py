"""
Shared paths for all Phase 2 scripts.

All data and embeddings live in Phase 1's folder — Phase 2 reads them
as-is, never writes to them.
"""

from pathlib import Path
import sys

ROOT   = Path(__file__).parent.parent   # Vector DB/
PHASE1 = ROOT / "semantic_search"

VECTORS_PATH = PHASE1 / "embeddings" / "vectors.npy"
IDS_PATH     = PHASE1 / "embeddings" / "ids.npy"
TITLES_PATH  = PHASE1 / "embeddings" / "titles.npy"
DATA_PATH    = PHASE1 / "data" / "arxiv_papers.csv"


def add_phase1_to_path() -> None:
    """
    Insert Phase 1's directory into sys.path so its hnsw package
    can be imported. Call this before 'from hnsw.xxx import ...'.
    """
    p = str(PHASE1)
    if p not in sys.path:
        sys.path.insert(0, p)
