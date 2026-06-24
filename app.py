"""
app.py — DocuMind  ·  Streamlit UI
────────────────────────────────────
Groq  ×  Pinecone  ×  LangChain  ×  HuggingFace Embeddings
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import List

import streamlit as st
from dotenv import load_dotenv

from document_processor import (
    SUPPORTED_EXTENSIONS,
    format_source_citation,
    process_uploaded_file,
)
from rag_engine import RAGEngine

# ── Bootstrap ─────────────────────────────────────────────────────────────────
load_dotenv()

st.set_page_config(
    page_title="DocuMind",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ── Fonts ── */
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600&family=DM+Sans:wght@300;400;500;600&family=Space+Grotesk:wght@400;600;700&display=swap');

/* ── Root palette ── */
:root {
    --bg-0:   #0a0c10;
    --bg-1:   #10131a;
    --bg-2:   #161b26;
    --bg-3:   #1e2536;
    --accent: #4f9cf9;
    --accent2:#7c5cfc;
    --green:  #2dd98f;
    --amber:  #f5a623;
    --red:    #f95f5f;
    --text-1: #e8ecf4;
    --text-2: #8a94a8;
    --text-3: #545e72;
    --border: #232a3b;
    --radius: 12px;
}

/* ── Global ── */
html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    background-color: var(--bg-0) !important;
    color: var(--text-1) !important;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: var(--bg-1) !important;
    border-right: 1px solid var(--border) !important;
}
[data-testid="stSidebar"] * { color: var(--text-1) !important; }

/* ── Inputs ── */
input, textarea, [data-testid="stTextInput"] input {
    background: var(--bg-2) !important;
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
    color: var(--text-1) !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.82rem !important;
}
input:focus, textarea:focus { border-color: var(--accent) !important; box-shadow: 0 0 0 2px rgba(79,156,249,.15) !important; }

/* ── Buttons ── */
[data-testid="stButton"] button {
    background: linear-gradient(135deg, var(--accent), var(--accent2)) !important;
    color: #fff !important;
    border: none !important;
    border-radius: 8px !important;
    font-family: 'Space Grotesk', sans-serif !important;
    font-weight: 600 !important;
    letter-spacing: .03em !important;
    transition: opacity .2s, transform .15s !important;
}
[data-testid="stButton"] button:hover { opacity: .9 !important; transform: translateY(-1px) !important; }

/* ── File uploader ── */
[data-testid="stFileUploader"] {
    background: var(--bg-2) !important;
    border: 2px dashed var(--border) !important;
    border-radius: var(--radius) !important;
    transition: border-color .2s !important;
}
[data-testid="stFileUploader"]:hover { border-color: var(--accent) !important; }

/* ── Chat messages ── */
.user-bubble {
    background: linear-gradient(135deg, #1a2a4a, #1d253d);
    border: 1px solid #2a3a5a;
    border-radius: 16px 16px 4px 16px;
    padding: 14px 18px;
    margin: 8px 0 8px 48px;
    color: var(--text-1);
    font-size: .93rem;
    line-height: 1.6;
}
.assistant-bubble {
    background: var(--bg-2);
    border: 1px solid var(--border);
    border-radius: 16px 16px 16px 4px;
    padding: 14px 18px;
    margin: 8px 48px 8px 0;
    color: var(--text-1);
    font-size: .93rem;
    line-height: 1.6;
}
.bubble-label {
    font-family: 'Space Grotesk', sans-serif;
    font-size: .72rem;
    font-weight: 600;
    letter-spacing: .06em;
    text-transform: uppercase;
    margin-bottom: 6px;
}
.user-label   { color: var(--accent); }
.assist-label { color: var(--green); }

/* ── Source chips ── */
.source-chip {
    display: inline-block;
    background: var(--bg-3);
    border: 1px solid var(--border);
    border-radius: 20px;
    padding: 3px 10px;
    font-size: .75rem;
    font-family: 'JetBrains Mono', monospace;
    color: var(--text-2);
    margin: 3px 2px;
    transition: border-color .15s;
}
.source-chip:hover { border-color: var(--accent); color: var(--accent); }

/* ── Status badges ── */
.badge {
    display: inline-flex;
    align-items: center;
    gap: 5px;
    padding: 4px 10px;
    border-radius: 20px;
    font-size: .76rem;
    font-weight: 600;
    font-family: 'Space Grotesk', sans-serif;
}
.badge-green  { background: rgba(45,217,143,.12); color: var(--green);  border: 1px solid rgba(45,217,143,.3); }
.badge-amber  { background: rgba(245,166,35,.12);  color: var(--amber);  border: 1px solid rgba(245,166,35,.3); }
.badge-blue   { background: rgba(79,156,249,.12);  color: var(--accent); border: 1px solid rgba(79,156,249,.3); }
.badge-red    { background: rgba(249,95,95,.12);   color: var(--red);    border: 1px solid rgba(249,95,95,.3); }

/* ── Metrics row ── */
.metric-card {
    background: var(--bg-2);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 12px 16px;
    text-align: center;
}
.metric-val {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 1.6rem;
    font-weight: 700;
    color: var(--accent);
}
.metric-lbl { font-size: .75rem; color: var(--text-2); margin-top: 2px; }

/* ── Chat input ── */
[data-testid="stChatInput"] textarea {
    background: var(--bg-2) !important;
    border: 1px solid var(--border) !important;
    border-radius: 12px !important;
    color: var(--text-1) !important;
    font-size: .9rem !important;
}

/* ── Expander ── */
[data-testid="stExpander"] {
    background: var(--bg-2) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius) !important;
}

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: var(--bg-1); }
::-webkit-scrollbar-thumb { background: var(--bg-3); border-radius: 3px; }

/* ── Divider ── */
hr { border-color: var(--border) !important; }

/* ── Code blocks ── */
code {
    background: var(--bg-3) !important;
    color: var(--accent) !important;
    border-radius: 4px !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: .83em !important;
    padding: 1px 5px !important;
}

/* ── Select boxes ── */
[data-testid="stSelectbox"] > div {
    background: var(--bg-2) !important;
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
}
</style>
""", unsafe_allow_html=True)


# ── Session state defaults ─────────────────────────────────────────────────────
def _init_state():
    defaults = {
        "engine":           None,
        "engine_ready":     False,
        "messages":         [],
        "indexed_files":    [],
        "total_chunks":     0,
        "init_status":      "",
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

_init_state()


# ── Header ────────────────────────────────────────────────────────────────────
col_h1, col_h2 = st.columns([3, 1])
with col_h1:
    st.markdown("""
    <h1 style="font-family:'Space Grotesk',sans-serif; font-size:2rem;
               font-weight:700; letter-spacing:-.02em; margin:0;
               background:linear-gradient(135deg,#4f9cf9,#7c5cfc); -webkit-background-clip:text;
               -webkit-text-fill-color:transparent; background-clip:text;">
        🧠 DocuMind
    </h1>
    <p style="color:#8a94a8; font-size:.9rem; margin-top:4px;">
        Groq · Pinecone · LangChain · Semantic Search · Conversational Memory
    </p>
    """, unsafe_allow_html=True)
with col_h2:
    status_html = (
        '<span class="badge badge-green">● Engine Ready</span>'
        if st.session_state.engine_ready
        else '<span class="badge badge-amber">● Not Initialised</span>'
    )
    st.markdown(f"<div style='text-align:right;padding-top:16px'>{status_html}</div>",
                unsafe_allow_html=True)

st.markdown("---")


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <p style="font-family:'Space Grotesk',sans-serif; font-weight:700;
              font-size:1rem; letter-spacing:.05em; text-transform:uppercase;
              color:#4f9cf9; margin-bottom:16px;">⚙ Configuration</p>
    """, unsafe_allow_html=True)

    # ── API Keys ──
    with st.expander("🔑 API Keys", expanded=not st.session_state.engine_ready):
        groq_key = st.text_input(
            "Groq API Key",
            value=os.getenv("GROQ_API_KEY", ""),
            type="password",
            placeholder="gsk_…",
            help="Get yours at console.groq.com",
        )
        pinecone_key = st.text_input(
            "Pinecone API Key",
            value=os.getenv("PINECONE_API_KEY", ""),
            type="password",
            placeholder="xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx",
            help="Get yours at app.pinecone.io",
        )

    # ── Model settings ──
    with st.expander("🤖 Model Settings"):
        groq_model = st.selectbox(
            "Groq Model",
            options=[
                "llama-3.3-70b-versatile",
                "llama-3.1-8b-instant",
                "meta-llama/llama-4-scout-17b-16e-instruct",
                "qwen/qwen3-32b",
            ],
            index=0,
        )
        temperature = st.slider("Temperature", 0.0, 1.0, 0.1, 0.05)
        top_k = st.slider("Top-K chunks retrieved", 1, 10, 4)

    # ── Index settings ──
    with st.expander("🌲 Pinecone Index"):
        index_name = st.text_input(
            "Index Name",
            value=os.getenv("PINECONE_INDEX_NAME", "documind-index"),
        )
        pinecone_region = st.selectbox(
            "Region",
            ["us-east-1", "us-west-2", "eu-west-1", "ap-southeast-1"],
        )

    # ── Chunking settings ──
    with st.expander("✂️ Chunking"):
        chunk_size    = st.slider("Chunk Size (chars)", 200, 2000, 1000, 50)
        chunk_overlap = st.slider("Chunk Overlap (chars)", 0, 500, 200, 25)

    st.markdown("---")

    # ── Init button ──
    if st.button("🚀  Initialise Engine", use_container_width=True):
        if not groq_key or not pinecone_key:
            st.error("Both API keys are required.")
        else:
            status_area = st.empty()
            try:
                def cb(msg):
                    status_area.info(msg)

                engine = RAGEngine(
                    groq_api_key=groq_key,
                    pinecone_api_key=pinecone_key,
                    index_name=index_name,
                    groq_model=groq_model,
                    pinecone_region=pinecone_region,
                    top_k=top_k,
                    temperature=temperature,
                )
                engine.initialize(status_callback=cb)
                st.session_state.engine       = engine
                st.session_state.engine_ready = True
                status_area.success("✅ Engine initialised!")
            except Exception as e:
                status_area.error(f"Initialisation failed: {e}")

    st.markdown("---")

    # ── Memory controls ──
    st.markdown("""
    <p style="font-family:'Space Grotesk',sans-serif; font-weight:700;
              font-size:.8rem; letter-spacing:.05em; text-transform:uppercase;
              color:#8a94a8;">Memory & Index</p>
    """, unsafe_allow_html=True)

    col_a, col_b = st.columns(2)
    with col_a:
        if st.button("🗑  Clear Chat", use_container_width=True):
            st.session_state.messages = []
            if st.session_state.engine:
                st.session_state.engine.clear_memory()
            st.rerun()
    with col_b:
        if st.button("🔄  Clear Index", use_container_width=True):
            if st.session_state.engine:
                st.session_state.engine.clear_index()
                st.session_state.indexed_files = []
                st.session_state.total_chunks  = 0
                st.success("Index cleared.")
            else:
                st.warning("Engine not ready.")

    st.markdown("---")

    # ── Stats ──
    st.markdown("""
    <p style="font-family:'Space Grotesk',sans-serif; font-weight:700;
              font-size:.8rem; letter-spacing:.05em; text-transform:uppercase;
              color:#8a94a8;">Session Stats</p>
    """, unsafe_allow_html=True)

    st.markdown(f"""
    <div class="metric-card" style="margin-bottom:8px">
        <div class="metric-val">{len(st.session_state.indexed_files)}</div>
        <div class="metric-lbl">Files Indexed</div>
    </div>
    <div class="metric-card" style="margin-bottom:8px">
        <div class="metric-val">{st.session_state.total_chunks}</div>
        <div class="metric-lbl">Total Chunks</div>
    </div>
    <div class="metric-card">
        <div class="metric-val">{len([m for m in st.session_state.messages if m["role"]=="user"])}</div>
        <div class="metric-lbl">Questions Asked</div>
    </div>
    """, unsafe_allow_html=True)


# ── Main area: two-column layout ─────────────────────────────────────────────
left_col, right_col = st.columns([1.8, 1], gap="large")

# ── Document upload panel ─────────────────────────────────────────────────────
with right_col:
    st.markdown("""
    <p style="font-family:'Space Grotesk',sans-serif; font-weight:700;
              font-size:.95rem; letter-spacing:.02em; color:#e8ecf4;">
        📁 Document Ingestion
    </p>
    """, unsafe_allow_html=True)

    uploaded_files = st.file_uploader(
        "Drop files here — PDF, DOCX, TXT, MD",
        type=["pdf", "docx", "doc", "txt", "md", "markdown"],
        accept_multiple_files=True,
        label_visibility="collapsed",
    )

    if uploaded_files:
        if st.button("📥  Index Documents", use_container_width=True):
            if not st.session_state.engine_ready:
                st.error("Please initialise the engine first.")
            else:
                progress = st.progress(0)
                status   = st.empty()
                total    = len(uploaded_files)
                for i, f in enumerate(uploaded_files):
                    if f.name in st.session_state.indexed_files:
                        status.info(f"⏭  Skipping already-indexed: {f.name}")
                        continue
                    status.info(f"⚙️  Processing {f.name}…")
                    try:
                        chunks = process_uploaded_file(
                            f,
                            chunk_size=chunk_size,
                            chunk_overlap=chunk_overlap,
                        )

                        def _cb(msg):
                            status.info(msg)

                        ingested = st.session_state.engine.add_documents(chunks, status_callback=_cb)
                        st.session_state.indexed_files.append(f.name)
                        st.session_state.total_chunks += ingested
                        status.success(f"✅  {f.name} → {ingested} chunks indexed")
                    except Exception as e:
                        st.error(f"Error processing {f.name}: {e}")
                    progress.progress((i + 1) / total)
                progress.empty()
                status.success("🎉 All documents indexed!")
                st.rerun()

    # ── Indexed file list ──
    if st.session_state.indexed_files:
        st.markdown("""
        <p style="font-size:.78rem; font-weight:600; color:#8a94a8;
                  text-transform:uppercase; letter-spacing:.05em; margin-top:14px;">
            Indexed Files
        </p>
        """, unsafe_allow_html=True)
        for fname in st.session_state.indexed_files:
            ext = Path(fname).suffix.lower()
            icon = {"pdf":"📕","docx":"📘","doc":"📘","txt":"📄","md":"📝","markdown":"📝"}.get(ext[1:], "📎")
            st.markdown(
                f'<div style="background:#161b26;border:1px solid #232a3b;border-radius:8px;'
                f'padding:7px 12px;margin:4px 0;font-size:.82rem;font-family:\'JetBrains Mono\',monospace;">'
                f'{icon} {fname}</div>',
                unsafe_allow_html=True,
            )

    st.markdown("---")

    # ── Quick-start hints ──
    st.markdown("""
    <p style="font-family:'Space Grotesk',sans-serif; font-weight:700;
              font-size:.9rem; color:#e8ecf4; margin-bottom:8px;">💡 Suggested Questions</p>
    """, unsafe_allow_html=True)

    suggestions = [
        "Summarise the key points of all documents",
        "What are the main findings or conclusions?",
        "List all mentioned dates and events",
        "Who are the key people or organisations?",
        "What recommendations are made?",
    ]
    for s in suggestions:
        if st.button(f"→ {s}", use_container_width=True, key=f"sug_{s}"):
            st.session_state["_pending_question"] = s
            st.rerun()


# ── Chat panel ────────────────────────────────────────────────────────────────
with left_col:
    st.markdown("""
    <p style="font-family:'Space Grotesk',sans-serif; font-weight:700;
              font-size:.95rem; letter-spacing:.02em; color:#e8ecf4;">
        💬 Chat
    </p>
    """, unsafe_allow_html=True)

    # ── Render conversation history ──
    chat_container = st.container()
    with chat_container:
        if not st.session_state.messages:
            st.markdown("""
            <div style="text-align:center; padding:40px 20px; color:#545e72;">
                <div style="font-size:2.5rem; margin-bottom:12px;">🧠</div>
                <div style="font-family:'Space Grotesk',sans-serif; font-size:1rem;
                            font-weight:600; color:#8a94a8;">DocuMind is ready — start asking</div>
                <div style="font-size:.85rem; margin-top:6px;">
                    Initialise the engine → Upload documents → Start asking
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            for msg in st.session_state.messages:
                if msg["role"] == "user":
                    st.markdown(f"""
                    <div class="user-bubble">
                        <div class="bubble-label user-label">You</div>
                        {msg["content"]}
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    answer_html  = msg["content"].replace("\n", "<br>")
                    sources_html = ""
                    for src in msg.get("sources", []):
                        chip = format_source_citation(src)
                        # strip markdown bold for HTML chip
                        chip_plain = chip.replace("**", "")
                        sources_html += f'<span class="source-chip">{chip_plain}</span>'

                    source_section = (
                        f'<div style="margin-top:10px;border-top:1px solid #232a3b;'
                        f'padding-top:8px;"><span style="font-size:.72rem;color:#545e72;'
                        f'font-family:\'Space Grotesk\',sans-serif;text-transform:uppercase;'
                        f'letter-spacing:.05em;">Sources</span><br>{sources_html}</div>'
                        if sources_html else ""
                    )

                    st.markdown(f"""
                    <div class="assistant-bubble">
                        <div class="bubble-label assist-label">Assistant</div>
                        {answer_html}
                        {source_section}
                    </div>
                    """, unsafe_allow_html=True)

    # ── Handle pending suggestion click ──
    if "_pending_question" in st.session_state:
        pending = st.session_state.pop("_pending_question")
        if st.session_state.engine_ready and st.session_state.indexed_files:
            st.session_state.messages.append({"role": "user", "content": pending})
            with st.spinner("Searching your documents…"):
                result = st.session_state.engine.query(pending)
            st.session_state.messages.append({
                "role":    "assistant",
                "content": result["answer"],
                "sources": result["sources"],
            })
            st.rerun()

    # ── Chat input ──
    user_input = st.chat_input(
        "Ask anything about your documents…",
        disabled=not st.session_state.engine_ready,
    )

    if user_input:
        if not st.session_state.indexed_files:
            st.warning("⚠️  Please index at least one document first.")
        else:
            st.session_state.messages.append({"role": "user", "content": user_input})
            with st.spinner("🔍 Searching & generating response…"):
                try:
                    result = st.session_state.engine.query(user_input)
                    st.session_state.messages.append({
                        "role":    "assistant",
                        "content": result["answer"],
                        "sources": result["sources"],
                    })
                except Exception as e:
                    st.session_state.messages.append({
                        "role":    "assistant",
                        "content": f"⚠️ Error: {e}",
                        "sources": [],
                    })
            st.rerun()
