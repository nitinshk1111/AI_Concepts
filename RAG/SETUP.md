# Setup Guide — RAG with Ollama

Everything you need to get the RAG app running locally, explained step by step.

---

## Why Ollama?

RAG has two parts:
- **Retrieval** — find relevant documents (done by Weaviate, built in Phase 2)
- **Generation** — read those documents and write an answer (needs a language model)

For generation we need an **LLM** (Large Language Model) — a model that can read
text and produce a coherent response. This is different from the embedding models
used in Phase 1 and 2, which only convert text to vectors.

Options for running an LLM:
| Option | Cost | Requires |
|--------|------|---------|
| OpenAI / Anthropic API | Pay per call | API key, internet |
| Ollama (local) | Free | ~4GB disk, no internet after setup |

We use **Ollama** — it runs a language model entirely on your Mac. No API key,
no cost, no data leaving your machine. Once set up, it works offline.

---

## What is Ollama?

Ollama is a tool that lets you download and run open-source language models locally.
It works like a local server — your app sends it a prompt, it returns a generated response.

Think of it as: **"Docker, but for AI models."**

You pull a model once, it's saved to disk, and Ollama serves it locally on
`http://localhost:11434`. Your Python code calls that URL — no cloud, no API key.

---

## What is Llama 3.2?

Llama 3.2 is Meta's open-source language model. "3.2" is the version.
It's capable enough to read a few research paper abstracts and synthesize a
clear answer — which is exactly what the RAG generation step needs.

Size pulled: ~2GB (the 3B parameter version — fast, fits on most Macs).

---

## Step-by-step setup

### Step 1 — Install Homebrew (macOS package manager)

Homebrew is the standard way to install developer tools on a Mac.
If you already have it (`brew --version` works), skip this step.

```bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
```

This will ask for your Mac password. Takes 1–2 minutes.

---

### Step 2 — Install Ollama

```bash
brew install ollama
```

This installs the Ollama CLI and server on your Mac.

---

### Step 3 — Start the Ollama server

Ollama runs as a background server that listens for requests.
You need it running before pulling models or running the app.

```bash
ollama serve
```

**Leave this terminal tab open.** The server must stay running.
Open a new terminal tab for all following commands.

---

### Step 4 — Pull the Llama 3.2 model

In a new terminal tab:

```bash
ollama pull llama3.2
```

This downloads the Llama 3.2 model (~2GB). One-time download — it's cached locally
after this. Takes 2–5 minutes depending on your internet speed.

---

### Step 5 — Verify it works

```bash
ollama run llama3.2 "What is a vector database?"
```

You should see a generated response in your terminal. If it answers, everything is working.

---

### Step 6 — Install Python dependencies

```bash
cd "/Users/nitinshekar/Desktop/Work/RAG"
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

---

### Step 7 — Download the dataset

This downloads ~25,000 recent AI/ML papers from ArXiv (2018–2021).
This era covers BERT, GPT-2, GPT-3, transformers, reinforcement learning, and more.

```bash
cd "/Users/nitinshekar/Desktop/Work/RAG"
source .venv/bin/activate
python download_data.py
```

Takes 2–5 minutes (streams and filters ~1.7M ArXiv papers).
Output: `RAG/data/arxiv_papers.csv`

---

### Step 8 — Build embeddings

Converts paper abstracts into vectors using `all-MiniLM-L6-v2`.

```bash
python build_embeddings.py
```

Takes 3–5 minutes for 25K papers.
Output: `RAG/embeddings/vectors.npy`, `ids.npy`, `titles.npy`

---

### Step 9 — Run the app

Make sure the Ollama server is still running (Step 3), then:

```bash
cd "/Users/nitinshekar/Desktop/Work/RAG"
source .venv/bin/activate
streamlit run app.py
```

> The first run loads 25K papers into Weaviate (~40s). Subsequent queries are fast.

This opens the RAG app in your browser at `http://localhost:8501`.

---

## How it all fits together

```
You type a question in the browser
            ↓
app.py embeds the question (sentence-transformers)
            ↓
retriever.py searches Weaviate for relevant papers
            ↓
Top-K paper abstracts retrieved
            ↓
generator.py sends [question + abstracts] to Ollama
            ↓
Ollama (Llama 3.2) generates an answer grounded in those papers
            ↓
app.py displays the answer + source papers in the browser
```

Everything runs locally on your Mac — no internet needed after setup,
no API keys, no cost per query.

---

## Troubleshooting

**"could not connect to ollama server"**
→ You need to run `ollama serve` first (Step 3) and keep that tab open.

**"model not found"**
→ Run `ollama pull llama3.2` (Step 4) first.

**Slow responses**
→ Normal on first query — Llama 3.2 loads into memory. Subsequent queries are faster.

**Weaviate insert takes ~40 seconds on app start**
→ Expected — 25K papers are inserted fresh each run (no persistence in embedded mode).
