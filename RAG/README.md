# Phase 3 — RAG App

Phase 3 puts everything from Phase 1 and 2 together into a working application.
It combines **Weaviate** (vector search) with **Ollama** (local LLM) to build a
Retrieval Augmented Generation system — a question-answering app grounded in real research papers.

> **Concepts reference:** [CONCEPTS.md](CONCEPTS.md) — RAG, LLMs, prompt engineering, and more.
> **Setup guide:** [SETUP.md](SETUP.md) — step-by-step instructions to run the app.

---

## What was built

| File | What it does |
|------|-------------|
| `download_data.py` | Downloads 25K recent ArXiv AI/ML papers (2018–2021) from HuggingFace |
| `build_embeddings.py` | Embeds paper abstracts using `all-MiniLM-L6-v2`, saves to `embeddings/` |
| `retriever.py` | Builds a Weaviate HNSW collection, retrieves top-K papers for a query |
| `generator.py` | Sends retrieved abstracts + query to Ollama (Llama 3.2), returns a grounded answer |
| `app.py` | Streamlit web app — dark-themed UI with search, answer card, and source papers |
| `paths.py` | Shared config — paths, model names, Ollama URL |

---

## Folder structure

```
RAG/
├── download_data.py      ← fetch 25K recent AI/ML papers
├── build_embeddings.py   ← embed abstracts → vectors.npy, ids.npy, titles.npy
├── retriever.py          ← Weaviate collection + ANN search
├── generator.py          ← Ollama prompt builder + API call
├── app.py                ← Streamlit UI
├── paths.py              ← shared paths and config
├── pyproject.toml        ← Python dependencies
├── README.md             ← this file
├── CONCEPTS.md           ← theory and key ideas
├── SETUP.md              ← step-by-step setup guide
├── data/
│   └── arxiv_papers.csv  ← 25K rows: id, title, abstract, categories
└── embeddings/
    ├── vectors.npy        ← 25K × 384 float32
    ├── ids.npy            ← ArXiv paper IDs
    └── titles.npy         ← paper titles
```

---

## How to run

> Full setup (Homebrew, Ollama, venv) is in [SETUP.md](SETUP.md). Below is the short version for returning runs.

**Step 1 — Start Ollama** (in a separate terminal tab, keep it running)
```bash
ollama serve
```

**Step 2 — Run the app**
```bash
cd "/Users/nitinshekar/Desktop/Work/RAG"
source .venv/bin/activate
streamlit run app.py
```

App opens at `http://localhost:8501`.

**First run only** — if `data/` or `embeddings/` are empty, run these first:
```bash
python download_data.py     # ~3 min — streams and filters 1.7M ArXiv papers
python build_embeddings.py  # ~1 min — embeds 25K abstracts
```

---

## Architecture

```
User types a question
        ↓
app.py embeds the question (all-MiniLM-L6-v2, 384-dim vector)
        ↓
retriever.py queries Weaviate (HNSW, cosine similarity)
        ↓
Top-K paper abstracts retrieved  ← grounding source
        ↓
generator.py builds a prompt:
  "Answer using ONLY these papers: [abstracts]
   Question: [query]"
        ↓
Ollama (Llama 3.2, running locally) generates the answer
        ↓
app.py displays the answer + source paper cards
```

Everything runs on your Mac — no internet needed after setup, no API keys, no cost per query.

---

## Dataset

**25,000 AI/ML papers from ArXiv (2018–2021)**

Filtered from the `gfissore/arxiv-abstracts-2021` HuggingFace dataset to include only
papers with ArXiv IDs from 2018 onwards (format `YYMM.NNNNN`) in categories:
`cs.LG`, `cs.AI`, `cs.CL`, `cs.CV`, `cs.NE`, `cs.IR`, `cs.RO`, `stat.ML`.

This covers the modern AI era: BERT, GPT-2, GPT-3, attention mechanisms, transformers,
diffusion models, vision transformers (ViT), CLIP, and more.

---

## Stack

| Component | Tool | Why |
|-----------|------|-----|
| Embeddings | `all-MiniLM-L6-v2` | Fast, 384-dim, good semantic quality — same as Phase 1 |
| Vector DB | Weaviate (embedded) | HNSW index, local, no server to manage — same as Phase 2 |
| LLM | Llama 3.2 via Ollama | Free, runs locally, no API key, ~2GB download |
| UI | Streamlit | Python-only, no frontend code needed |

---

## Example questions to try

- "How does BERT use bidirectional context?"
- "What is GPT-3 and what makes it different from earlier models?"
- "Explain the attention mechanism in transformers"
- "What is reinforcement learning from human feedback?"
- "How do graph neural networks work?"
- "What are diffusion models and how do they generate images?"
