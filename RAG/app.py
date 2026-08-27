"""
RAG App — Retrieval Augmented Generation over 20K ArXiv AI/ML papers.

Run:
    streamlit run app.py
"""

import os
import certifi
os.environ.setdefault("SSL_CERT_FILE", certifi.where())

import streamlit as st
import weaviate
from sentence_transformers import SentenceTransformer

from retriever import load_data, build_weaviate_collection, retrieve
from generator import generate
from paths import EMBED_MODEL

# ── Page config ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="ArXiv Research Assistant",
    page_icon="📚",
    layout="wide",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────

st.markdown("""
<style>
    /* Hide Streamlit branding */
    #MainMenu, footer, header { visibility: hidden; }

    /* Page background */
    .stApp { background-color: #0f1117; }

    /* Hero header */
    .hero {
        background: linear-gradient(135deg, #1a1f2e 0%, #16213e 50%, #1a1f2e 100%);
        border: 1px solid #2d3548;
        border-radius: 16px;
        padding: 48px 40px;
        margin-bottom: 32px;
        text-align: center;
    }
    .hero h1 {
        font-size: 2.4rem;
        font-weight: 700;
        color: #ffffff;
        margin: 0 0 8px 0;
    }
    .hero p {
        font-size: 1.05rem;
        color: #8892a4;
        margin: 0;
    }
    .hero .badge {
        display: inline-block;
        background: #1e3a5f;
        color: #60a5fa;
        font-size: 0.75rem;
        font-weight: 600;
        padding: 4px 12px;
        border-radius: 20px;
        margin: 16px 4px 0 4px;
        border: 1px solid #2d5a8e;
    }

    /* Search box */
    .stTextInput > div > div > input {
        background-color: #1a1f2e !important;
        border: 1.5px solid #2d3548 !important;
        border-radius: 10px !important;
        color: #e2e8f0 !important;
        font-size: 1rem !important;
        padding: 14px 16px !important;
        transition: border-color 0.2s;
    }
    .stTextInput > div > div > input:focus {
        border-color: #3b82f6 !important;
        box-shadow: 0 0 0 3px rgba(59, 130, 246, 0.15) !important;
    }
    .stTextInput > div > div > input::placeholder {
        color: #4a5568 !important;
    }

    /* Ask button */
    .stButton > button {
        background: linear-gradient(135deg, #3b82f6, #2563eb) !important;
        color: white !important;
        border: none !important;
        border-radius: 10px !important;
        padding: 14px 32px !important;
        font-size: 1rem !important;
        font-weight: 600 !important;
        width: 100% !important;
        transition: opacity 0.2s !important;
    }
    .stButton > button:hover { opacity: 0.9 !important; }
    .stButton > button:disabled { opacity: 0.4 !important; }

    /* Answer card */
    .answer-card {
        background: #1a1f2e;
        border: 1px solid #2d3548;
        border-left: 4px solid #3b82f6;
        border-radius: 12px;
        padding: 28px 32px;
        margin: 24px 0;
        color: #e2e8f0;
        font-size: 1.02rem;
        line-height: 1.75;
    }
    .answer-label {
        font-size: 0.75rem;
        font-weight: 700;
        color: #3b82f6;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        margin-bottom: 12px;
    }

    /* Source cards */
    .source-card {
        background: #1a1f2e;
        border: 1px solid #2d3548;
        border-radius: 10px;
        padding: 18px 20px;
        margin-bottom: 10px;
        transition: border-color 0.2s;
    }
    .source-card:hover { border-color: #3b82f6; }
    .source-title {
        font-size: 0.95rem;
        font-weight: 600;
        color: #e2e8f0;
        margin-bottom: 6px;
    }
    .source-meta {
        font-size: 0.78rem;
        color: #60a5fa;
        margin-bottom: 8px;
    }
    .source-abstract {
        font-size: 0.83rem;
        color: #8892a4;
        line-height: 1.6;
        display: -webkit-box;
        -webkit-line-clamp: 3;
        -webkit-box-orient: vertical;
        overflow: hidden;
    }

    /* Section label */
    .section-label {
        font-size: 0.75rem;
        font-weight: 700;
        color: #4a5568;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        margin: 28px 0 12px 0;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background-color: #12151f !important;
        border-right: 1px solid #1e2433 !important;
    }
    .sidebar-stat {
        background: #1a1f2e;
        border: 1px solid #2d3548;
        border-radius: 8px;
        padding: 12px 16px;
        margin-bottom: 8px;
        font-size: 0.83rem;
        color: #8892a4;
    }
    .sidebar-stat span { color: #e2e8f0; font-weight: 600; }

    /* Slider */
    .stSlider { padding: 0 4px; }
</style>
""", unsafe_allow_html=True)

# ── Load resources ─────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner="Loading embedding model...")
def load_model():
    return SentenceTransformer(EMBED_MODEL)


@st.cache_resource(show_spinner="Indexing 25K papers into Weaviate (one-time, ~40s)...")
def load_weaviate():
    vectors, ids, titles, abstracts = load_data()
    try:
        client = weaviate.connect_to_embedded()
    except Exception:
        # Another Weaviate process already holds the ports — connect to it
        client = weaviate.connect_to_local(port=8079, grpc_port=50050)
    collection = build_weaviate_collection(client, vectors, ids, titles, abstracts)
    return client, collection


model                = load_model()
client, collection   = load_weaviate()

# ── Sidebar ────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("### ⚙️ Settings")
    top_k = st.slider("Source papers", min_value=1, max_value=10, value=5,
                      help="How many papers to retrieve and pass to the LLM")
    st.caption("More papers = richer context, slower response.")

    st.markdown("<div style='margin-top:24px'></div>", unsafe_allow_html=True)
    st.markdown("### 🔧 Stack")

    for label, value in [
        ("LLM",        "Llama 3.2 (local)"),
        ("Vector DB",  "Weaviate embedded"),
        ("Embeddings", "all-MiniLM-L6-v2"),
        ("Dataset",    "25K ArXiv (2018–2021)"),
    ]:
        st.markdown(
            f"<div class='sidebar-stat'>{label}<br><span>{value}</span></div>",
            unsafe_allow_html=True,
        )

    st.markdown("<div style='margin-top:24px'></div>", unsafe_allow_html=True)
    st.markdown("### 💡 Try asking")
    examples = [
        "How does attention work in transformers?",
        "What is reinforcement learning from human feedback?",
        "Explain graph neural networks",
        "How does BERT differ from GPT?",
    ]
    for ex in examples:
        st.markdown(f"<div style='font-size:0.8rem;color:#6b7280;padding:4px 0'>→ {ex}</div>",
                    unsafe_allow_html=True)

# ── Hero ───────────────────────────────────────────────────────────────────────

st.markdown("""
<div class="hero">
    <h1>📚 ArXiv Research Assistant</h1>
    <p>Ask any question about AI/ML. Answers are generated by Llama 3.2<br>
    and grounded in 25,000 real research paper abstracts (2018–2021).</p>
    <span class="badge">RAG</span>
    <span class="badge">Weaviate</span>
    <span class="badge">Llama 3.2</span>
    <span class="badge">25K Papers</span>
</div>
""", unsafe_allow_html=True)

# ── Search ─────────────────────────────────────────────────────────────────────

col1, col2 = st.columns([5, 1])
with col1:
    query = st.text_input(
        "question",
        placeholder="e.g. How do graph neural networks learn from structured data?",
        label_visibility="collapsed",
    )
with col2:
    ask = st.button("Ask", disabled=not query, type="primary")

# ── Results ────────────────────────────────────────────────────────────────────

if ask and query:
    with st.spinner("🔍 Searching 25K papers..."):
        papers = retrieve(query, collection, model, top_k=top_k)

    with st.spinner("🤖 Generating answer with Llama 3.2..."):
        answer = generate(query, papers)

    # Answer
    st.markdown(f"""
    <div class="answer-card">
        <div class="answer-label">Answer</div>
        {answer}
    </div>
    """, unsafe_allow_html=True)

    # Sources
    st.markdown(f"<div class='section-label'>Sources — {len(papers)} papers retrieved</div>",
                unsafe_allow_html=True)

    for i, paper in enumerate(papers, 1):
        st.markdown(f"""
        <div class="source-card">
            <div class="source-title">{i}. {paper['title']}</div>
            <div class="source-meta">arxiv:{paper['arxiv_id']} &nbsp;·&nbsp; distance: {paper['distance']}</div>
            <div class="source-abstract">{paper['abstract']}</div>
        </div>
        """, unsafe_allow_html=True)
