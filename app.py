"""
app.py — RAG Chatbot UI (Streamlit)
Beautiful dark-mode interface with:
  • Drag-and-drop document upload
  • Streaming responses
  • Source citations
  • Document management
"""

import streamlit as st
from engine import (
    DATA_DIR,
    ingest_documents,
    build_chain,
    get_sources_for_query,
    get_doc_count,
    clear_vector_store,
)

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="RAG Chatbot · phi4-mini",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Injected CSS ───────────────────────────────────────────────────────────────
st.markdown(
    """
<style>
/* ── Google Font ──────────────────────────────────────── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

/* ── Global ───────────────────────────────────────────── */
html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}
.stApp {
    background: linear-gradient(135deg, #0d0d1a 0%, #111128 50%, #0d1a2e 100%);
    min-height: 100vh;
}

/* ── Sidebar ──────────────────────────────────────────── */
[data-testid="stSidebar"] {
    background: rgba(15, 15, 35, 0.95) !important;
    border-right: 1px solid rgba(99, 102, 241, 0.2);
}
[data-testid="stSidebar"] .block-container {
    padding-top: 1.5rem;
}

/* ── Header ───────────────────────────────────────────── */
.rag-header {
    background: linear-gradient(135deg, rgba(99,102,241,0.15) 0%, rgba(139,92,246,0.10) 100%);
    border: 1px solid rgba(99,102,241,0.3);
    border-radius: 16px;
    padding: 1.4rem 1.8rem;
    margin-bottom: 1.5rem;
    backdrop-filter: blur(10px);
}
.rag-header h1 {
    font-size: 1.6rem;
    font-weight: 700;
    background: linear-gradient(135deg, #a78bfa, #60a5fa);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin: 0;
    padding: 0;
}
.rag-header p {
    color: #94a3b8;
    font-size: 0.85rem;
    margin: 0.3rem 0 0;
}

/* ── Status badges ────────────────────────────────────── */
.badge {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 4px 12px;
    border-radius: 99px;
    font-size: 0.75rem;
    font-weight: 600;
    letter-spacing: 0.03em;
}
.badge-green  { background: rgba(34,197,94,0.15); color: #4ade80; border: 1px solid rgba(34,197,94,0.3); }
.badge-yellow { background: rgba(234,179,8,0.15);  color: #facc15; border: 1px solid rgba(234,179,8,0.3); }
.badge-blue   { background: rgba(99,102,241,0.15); color: #818cf8; border: 1px solid rgba(99,102,241,0.3); }
.badge-red    { background: rgba(239,68,68,0.12);  color: #f87171; border: 1px solid rgba(239,68,68,0.3); }

/* ── Chat messages ────────────────────────────────────── */
.chat-bubble-user {
    background: linear-gradient(135deg, rgba(99,102,241,0.22), rgba(139,92,246,0.18));
    border: 1px solid rgba(99,102,241,0.35);
    border-radius: 14px 14px 4px 14px;
    padding: 0.85rem 1.1rem;
    margin: 0.5rem 0;
    color: #e2e8f0;
    line-height: 1.6;
    max-width: 88%;
    margin-left: auto;
}
.chat-bubble-ai {
    background: rgba(30, 30, 55, 0.85);
    border: 1px solid rgba(99,102,241,0.18);
    border-radius: 14px 14px 14px 4px;
    padding: 0.85rem 1.1rem;
    margin: 0.5rem 0;
    color: #cbd5e1;
    line-height: 1.6;
    max-width: 92%;
}
.chat-label {
    font-size: 0.7rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-bottom: 0.3rem;
}
.label-user { color: #a78bfa; }
.label-ai   { color: #60a5fa; }

/* ── Sources panel ────────────────────────────────────── */
.source-card {
    background: rgba(15,23,42,0.7);
    border: 1px solid rgba(99,102,241,0.2);
    border-left: 3px solid #6366f1;
    border-radius: 8px;
    padding: 0.6rem 0.9rem;
    margin: 0.35rem 0;
    font-size: 0.8rem;
    color: #94a3b8;
}
.source-card strong { color: #a5b4fc; }

/* ── Upload area ──────────────────────────────────────── */
[data-testid="stFileUploader"] {
    border: 2px dashed rgba(99,102,241,0.4) !important;
    border-radius: 12px;
    background: rgba(99,102,241,0.05);
    transition: border-color 0.2s;
}
[data-testid="stFileUploader"]:hover {
    border-color: rgba(99,102,241,0.7) !important;
}

/* ── Buttons ──────────────────────────────────────────── */
.stButton > button {
    background: linear-gradient(135deg, #6366f1, #8b5cf6) !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    font-weight: 600 !important;
    transition: all 0.2s ease !important;
}
.stButton > button:hover {
    transform: translateY(-1px);
    box-shadow: 0 6px 20px rgba(99,102,241,0.4) !important;
}

/* ── Chat input ───────────────────────────────────────── */
[data-testid="stChatInput"] > div {
    background: rgba(20,20,45,0.9) !important;
    border: 1px solid rgba(99,102,241,0.35) !important;
    border-radius: 14px !important;
}

/* ── Dividers & misc ─────────────────────────────────── */
hr { border-color: rgba(99,102,241,0.15) !important; }
.stMarkdown p { color: #94a3b8; }
[data-testid="metric-container"] {
    background: rgba(15,23,42,0.6);
    border: 1px solid rgba(99,102,241,0.2);
    border-radius: 10px;
    padding: 0.5rem;
}

/* hide streamlit branding */
#MainMenu, footer, header { visibility: hidden; }
</style>
""",
    unsafe_allow_html=True,
)


# ── Session state defaults ─────────────────────────────────────────────────────
def _init_state():
    defaults = {
        "messages": [],
        "doc_count": 0,
        "chain": None,
        "retriever": None,
        "ready": False,
        "last_sources": [],
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


_init_state()


# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(
        """
        <div style="text-align:center; margin-bottom:1.2rem;">
          <div style="font-size:2.5rem; margin-bottom:0.3rem;">🧠</div>
          <div style="font-weight:700; font-size:1.1rem; color:#a78bfa;">RAG Chatbot</div>
          <div style="font-size:0.72rem; color:#64748b; margin-top:2px;">phi4-mini · nomic-embed · ChromaDB</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ── Model / GPU Status ────────────────────────────────────────────────────
    st.markdown("#### ⚡ System Status")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(
            '<div class="badge badge-green">🟢 GPU ROCm</div>', unsafe_allow_html=True
        )
    with col2:
        st.markdown(
            '<div class="badge badge-blue">phi4-mini</div>', unsafe_allow_html=True
        )

    st.markdown(
        """
        <div style="font-size:0.72rem; color:#475569; margin: 0.5rem 0 1rem;">
          AMD RX 6500M · gfx1034 · ROCm
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.divider()

    # ── Document Upload ───────────────────────────────────────────────────────
    st.markdown("#### 📂 Upload Documents")
    st.markdown(
        '<div style="font-size:0.78rem; color:#64748b; margin-bottom:0.5rem;">Supports PDF, TXT, MD</div>',
        unsafe_allow_html=True,
    )

    uploaded_files = st.file_uploader(
        "Drop files here",
        type=["pdf", "txt", "md", "markdown"],
        accept_multiple_files=True,
        label_visibility="collapsed",
    )

    if uploaded_files:
        if st.button("📥 Ingest Documents", use_container_width=True):
            DATA_DIR.mkdir(parents=True, exist_ok=True)
            temp_paths = []

            with st.spinner("Saving files…"):
                for uf in uploaded_files:
                    dest = DATA_DIR / uf.name
                    dest.write_bytes(uf.getbuffer())
                    temp_paths.append(dest)

            progress_bar = st.progress(0, text="Chunking & embedding…")

            def _cb(i, total, name):
                progress_bar.progress(
                    int((i + 1) / total * 100),
                    text=f"Processing: {name}",
                )

            n_files, n_chunks = ingest_documents(
                directory=DATA_DIR, progress_callback=_cb
            )
            progress_bar.empty()

            if n_chunks > 0:
                st.success(f"✅ {n_files} file(s) → {n_chunks} chunks indexed!")
                st.session_state.chain, st.session_state.retriever = build_chain(
                    streaming=True
                )
                st.session_state.doc_count = get_doc_count()
                st.session_state.ready = st.session_state.chain is not None
                st.rerun()
            else:
                st.error("No content could be extracted. Check file format.")

    st.divider()

    # ── Knowledge base stats ──────────────────────────────────────────────────
    st.markdown("#### 📊 Knowledge Base")
    doc_count = st.session_state.doc_count or get_doc_count()
    st.session_state.doc_count = doc_count

    if doc_count > 0:
        st.markdown(
            f'<div class="badge badge-green">✅ {doc_count} chunks indexed</div>',
            unsafe_allow_html=True,
        )
        # Auto-build chain if not built yet
        if st.session_state.chain is None:
            st.session_state.chain, st.session_state.retriever = build_chain(
                streaming=True
            )
            st.session_state.ready = st.session_state.chain is not None
    else:
        st.markdown(
            '<div class="badge badge-yellow">⚠ No documents yet</div>',
            unsafe_allow_html=True,
        )

    st.markdown("")

    col_clr, col_chat = st.columns(2)
    with col_clr:
        if st.button("🗑 Clear DB", use_container_width=True):
            clear_vector_store()
            st.session_state.chain = None
            st.session_state.retriever = None
            st.session_state.ready = False
            st.session_state.doc_count = 0
            st.success("Vector store cleared.")
            st.rerun()
    with col_chat:
        if st.button("💬 New Chat", use_container_width=True):
            st.session_state.messages = []
            st.session_state.last_sources = []
            st.rerun()

    st.divider()

    # ── Settings ──────────────────────────────────────────────────────────────
    with st.expander("⚙️ Settings", expanded=False):
        show_sources = st.toggle("Show source citations", value=True)
        st.caption("GPU: AMD RX 6500M (RDNA2)")
        st.caption("HSA_GFX: 10.3.0 (gfx1034)")
        st.caption("ctx window: 4096 tokens")

    st.markdown(
        '<div style="font-size:0.68rem; color:#334155; margin-top:1rem; text-align:center;">'
        "Powered by Ollama · LangChain · ChromaDB</div>",
        unsafe_allow_html=True,
    )


# ── Main area ──────────────────────────────────────────────────────────────────
st.markdown(
    """
    <div class="rag-header">
      <h1>🧠 RAG Chatbot</h1>
      <p>Upload documents in the sidebar, then chat with your data — 100% local, GPU-accelerated.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ── Guard: no documents ────────────────────────────────────────────────────────
if not st.session_state.ready and get_doc_count() == 0:
    st.markdown(
        """
        <div style="
          text-align:center;
          padding: 3.5rem 2rem;
          background: rgba(99,102,241,0.06);
          border: 1px dashed rgba(99,102,241,0.3);
          border-radius: 18px;
          margin-top: 2rem;
        ">
          <div style="font-size:3.5rem; margin-bottom:1rem;">📄</div>
          <div style="font-size:1.15rem; font-weight:600; color:#a78bfa; margin-bottom:0.5rem;">
            No Documents Yet
          </div>
          <div style="color:#64748b; font-size:0.88rem; max-width:420px; margin:0 auto;">
            Upload PDFs, TXT or Markdown files using the sidebar to build your knowledge base.
            The chatbot will answer questions based on your documents only.
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.stop()

# ── Ensure chain is ready ─────────────────────────────────────────────────────
if st.session_state.chain is None:
    with st.spinner("Loading model…"):
        st.session_state.chain, st.session_state.retriever = build_chain(streaming=True)
        st.session_state.ready = st.session_state.chain is not None

# ── Chat history display ───────────────────────────────────────────────────────
for msg in st.session_state.messages:
    role = msg["role"]
    if role == "user":
        st.markdown(
            f'<div class="chat-label label-user">You</div>'
            f'<div class="chat-bubble-user">{msg["content"]}</div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f'<div class="chat-label label-ai">🧠 phi4-mini</div>'
            f'<div class="chat-bubble-ai">{msg["content"]}</div>',
            unsafe_allow_html=True,
        )
        # Show sources if stored
        if show_sources and msg.get("sources"):
            with st.expander("📎 Sources", expanded=False):
                for s in msg["sources"]:
                    page_txt = f" · page {s['page']}" if s.get("page") else ""
                    st.markdown(
                        f'<div class="source-card">'
                        f'<strong>📄 {s["file"]}{page_txt}</strong><br>'
                        f'<span style="font-size:0.75rem;">{s["snippet"]}…</span>'
                        f"</div>",
                        unsafe_allow_html=True,
                    )

# ── Chat input ─────────────────────────────────────────────────────────────────
if prompt := st.chat_input("Ask something about your documents…"):
    # Store user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.markdown(
        f'<div class="chat-label label-user">You</div>'
        f'<div class="chat-bubble-user">{prompt}</div>',
        unsafe_allow_html=True,
    )

    # Retrieve sources for citation (parallel, non-streaming)
    sources = []
    if show_sources:
        sources = get_sources_for_query(prompt)

    # Stream the LLM response
    st.markdown(
        '<div class="chat-label label-ai">🧠 phi4-mini</div>', unsafe_allow_html=True
    )
    response_box = st.empty()
    full_response = ""

    try:
        with st.spinner(""):
            for chunk in st.session_state.chain.stream(prompt):
                full_response += chunk
                response_box.markdown(
                    f'<div class="chat-bubble-ai">{full_response}▌</div>',
                    unsafe_allow_html=True,
                )
        # Final (remove cursor)
        response_box.markdown(
            f'<div class="chat-bubble-ai">{full_response}</div>',
            unsafe_allow_html=True,
        )
    except Exception as e:
        full_response = f"⚠️ Error: {e}"
        response_box.error(full_response)

    # Show sources inline
    if show_sources and sources:
        with st.expander("📎 Sources", expanded=True):
            for s in sources:
                page_txt = f" · page {s['page']}" if s.get("page") else ""
                st.markdown(
                    f'<div class="source-card">'
                    f'<strong>📄 {s["file"]}{page_txt}</strong><br>'
                    f'<span style="font-size:0.75rem;">{s["snippet"]}…</span>'
                    f"</div>",
                    unsafe_allow_html=True,
                )

    # Persist to history
    st.session_state.messages.append(
        {"role": "assistant", "content": full_response, "sources": sources}
    )
