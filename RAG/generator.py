"""
Generator — sends retrieved paper abstracts + user query to Ollama (Llama 3.2)
and returns a grounded answer.

Used by app.py. Not meant to be run directly.
"""

import requests
from paths import OLLAMA_URL, OLLAMA_MODEL


def build_prompt(query: str, papers: list[dict]) -> str:
    context = ""
    for i, paper in enumerate(papers, 1):
        context += f"Paper {i}: {paper['title']}\n"
        context += f"Abstract: {paper['abstract']}\n\n"

    return f"""You are a research assistant. Answer the question below using ONLY
the provided research paper abstracts. Be concise and accurate.
If the papers don't contain enough information to answer, say so clearly.

Research papers:
{context}
Question: {query}

Answer:"""


def generate(query: str, papers: list[dict]) -> str:
    prompt = build_prompt(query, papers)

    response = requests.post(
        f"{OLLAMA_URL}/api/generate",
        json={
            "model":  OLLAMA_MODEL,
            "prompt": prompt,
            "stream": False,
        },
        timeout=120,
    )
    response.raise_for_status()
    return response.json()["response"].strip()
