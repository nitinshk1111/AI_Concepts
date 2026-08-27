"""
Shared paths for the RAG project.
Data and embeddings live in RAG/data/ and RAG/embeddings/.
Run download_data.py then build_embeddings.py to populate them.
"""

from pathlib import Path

ROOT         = Path(__file__).parent
VECTORS_PATH = ROOT / "embeddings" / "vectors.npy"
IDS_PATH     = ROOT / "embeddings" / "ids.npy"
TITLES_PATH  = ROOT / "embeddings" / "titles.npy"
DATA_PATH    = ROOT / "data" / "arxiv_papers.csv"

OLLAMA_URL   = "http://localhost:11434"
OLLAMA_MODEL = "llama3.2"
EMBED_MODEL  = "all-MiniLM-L6-v2"
COLLECTION   = "ArxivRAG"
