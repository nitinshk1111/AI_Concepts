# Phase 3 Concepts — RAG

---

## What is RAG?

**Retrieval Augmented Generation** (RAG) is a technique that makes a language model answer
questions using a specific set of documents — instead of relying only on what it learned
during training.

Plain LLM without RAG:
```
User: "How does BERT work?"
LLM: [generates answer from training memory — may be outdated, vague, or hallucinated]
```

LLM with RAG:
```
User: "How does BERT work?"
→ Search a paper database for the most relevant abstracts
→ Give those abstracts to the LLM as context
LLM: [generates answer grounded in those specific papers]
```

The key insight: **language models are good at reading and summarizing text**.
RAG gives them the right text to work from, so they don't have to guess.

---

## Why RAG beats a plain LLM for this use case

| Problem | Without RAG | With RAG |
|---------|-------------|----------|
| Hallucination | LLM invents plausible-sounding facts | LLM is constrained to retrieved text |
| Outdated info | Model frozen at training cutoff | You control the document set |
| Source tracing | "Trust me" — no citations | You see which papers were used |
| Domain focus | General knowledge, diluted | Only papers in your database |

This is why production systems (legal research, medical Q&A, enterprise knowledge bases)
almost always use RAG rather than relying on a general-purpose LLM alone.

---

## The two steps of RAG

### Step 1 — Retrieval

The question is converted to a vector (same embedding model used to embed the documents).
Then an ANN search finds the most semantically similar paper abstracts.

```
"How does BERT work?"
        ↓  (all-MiniLM-L6-v2)
[0.13, -0.42, 0.07, ...]  ← 384-dim query vector
        ↓  (Weaviate HNSW, cosine similarity)
Top 5 paper abstracts  ← the retrieval result
```

This is exactly what was built in Phase 1 (flat index) and Phase 2 (Weaviate HNSW).
RAG adds the generation step on top.

### Step 2 — Generation

The retrieved abstracts are inserted into a **prompt** that instructs the LLM to answer
using only those documents. The LLM then generates a natural-language answer.

```python
prompt = """You are a research assistant. Answer the question below using ONLY
the provided research paper abstracts. Be concise and accurate.
If the papers don't contain enough information to answer, say so clearly.

Research papers:
Paper 1: BERT: Pre-training of Deep Bidirectional Transformers...
Abstract: We introduce BERT...

Question: How does BERT use bidirectional context?

Answer:"""
```

The LLM reads the papers and synthesizes an answer. If no relevant papers were retrieved,
the LLM correctly says so — it won't hallucinate an answer from nowhere.

---

## What is a Large Language Model (LLM)?

An LLM is a neural network trained on massive amounts of text to predict the next word.
After enough training, it learns to:

- Follow instructions
- Summarize and explain text
- Answer questions
- Write code and prose

LLMs are **not** a database — they don't look things up. They generate text based on
patterns learned during training. This is why they can hallucinate: they produce text
that sounds right, even if it's wrong.

RAG fixes this by making the LLM work from provided text rather than memory.

---

## What is Ollama?

Ollama is a tool that lets you run open-source LLMs locally on your Mac.
It acts as a local server:

```
Your Python code → POST to http://localhost:11434/api/generate → Ollama → Llama 3.2
```

No API key, no cost, no data leaving your machine. You pull a model once, it's cached
on disk, and Ollama serves it locally for every query.

Think of it as: **Docker, but for AI models.**

---

## What is Llama 3.2?

Llama 3.2 is Meta's open-source language model. The version used here is the 3B parameter
variant — it's capable enough to read and synthesize research abstracts, while being
small enough to run on a MacBook (~2GB download, fits in RAM).

Parameters are the learned weights of the model — more parameters generally means more
knowledge and capability, but also more memory and slower inference.

---

## Prompt engineering

The way you write the prompt to the LLM heavily affects the quality of the answer.
In this project, the prompt has three key design decisions:

1. **Role instruction** — "You are a research assistant" sets the context and tone
2. **Constraint** — "Answer using ONLY the provided abstracts" prevents hallucination
3. **Fallback** — "If the papers don't contain enough information, say so" gives the LLM
   permission to admit ignorance rather than guess

Without the constraint, the LLM would blend the retrieved text with its own training
knowledge, making it impossible to trace which parts came from your documents.

---

## What is Streamlit?

Streamlit is a Python library that turns a Python script into a web app.
You write Python, and it renders interactive UI elements in the browser —
text inputs, buttons, sliders, cards — without writing any HTML or JavaScript.

```python
query = st.text_input("Your question")
if st.button("Ask"):
    answer = generate(query)
    st.write(answer)
```

This makes it ideal for ML demos and data apps where the logic is in Python and
you don't want to build a full frontend.

---

## How the pieces connect

```
arxiv_papers.csv           25K recent AI/ML paper abstracts (2018–2021)
        ↓
build_embeddings.py        all-MiniLM-L6-v2 → vectors.npy (25K × 384 floats)
        ↓
retriever.py               Weaviate HNSW collection, ANN search on query vector
        ↓
generator.py               Prompt builder + Ollama API call (Llama 3.2)
        ↓
app.py                     Streamlit UI — search box, answer card, source papers
```

Phase 3 is not a new set of techniques — it's the same embeddings from Phase 1,
the same HNSW from Phase 2, plus an LLM generation layer on top. The pipeline
concept is what's new: chaining retrieval and generation into one coherent product.

---

## Key terms

| Term | Meaning |
|------|---------|
| RAG | Retrieval Augmented Generation — grounding LLM answers in retrieved documents |
| LLM | Large Language Model — generates text based on patterns from training |
| Prompt | The full text sent to the LLM (instructions + context + question) |
| Context window | How much text an LLM can read in one call |
| Hallucination | When an LLM generates confident but incorrect information |
| Ollama | Tool to run open-source LLMs locally |
| Streamlit | Python library for building web UIs without frontend code |
| Grounding | Anchoring LLM output to specific source documents |
