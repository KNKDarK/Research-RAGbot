"""
engine.py — Optimized RAG Engine
- LLM  : phi4-mini via Ollama (ROCm/AMD GPU accelerated) or API-based (OpenAI/Groq)
- Embed: nomic-embed-text via Ollama or all-MiniLM-L6-v2 via HuggingFace
- Store: ChromaDB (persistent, MMR retrieval)
- AMD RX 6500M (RDNA2 / gfx1034) → HSA_OVERRIDE_GFX_VERSION=10.3.0
"""
# ruff: noqa: E402

import os
import shutil
from functools import cache
from pathlib import Path

# ── ROCm / AMD GPU hints (set before importing anything GPU-related) ──────────
os.environ.setdefault("HSA_OVERRIDE_GFX_VERSION", "10.3.0")  # RDNA2 gfx1034
os.environ.setdefault("OLLAMA_NUM_GPU", "1")
os.environ.setdefault("OLLAMA_GPU_OVERHEAD", "256MiB")  # keep headroom

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain_core.documents import Document

# ── Config ────────────────────────────────────────────────────────────────────
DATA_DIR = Path("./data")  # pre-downloaded / bundled papers
UPLOAD_DIR = Path("./uploads")  # user-uploaded papers
CHROMA_DIR = Path("./chroma_db")

# Embedding provider: "ollama" or "huggingface" (None → auto-detect)
EMBED_PROVIDER = os.environ.get("EMBED_PROVIDER")  # default: auto-detect
EMBED_MODEL = os.environ.get("EMBED_MODEL", "nomic-embed-text")  # Ollama model
HF_EMBED_MODEL = os.environ.get(
    "HF_EMBED_MODEL", "all-MiniLM-L6-v2"
)  # HuggingFace model

OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")

# LLM provider: "ollama", "openai", or "groq" (None → auto-detect)
LLM_PROVIDER = os.environ.get("LLM_PROVIDER")  # default: auto-detect
LLM_MODEL = os.environ.get("LLM_MODEL", "phi4-mini")  # Ollama model
GOOGLE_MODEL = os.environ.get("GOOGLE_MODEL", "gemini-3.5-flash")
GROQ_MODEL = os.environ.get("GROQ_MODEL", "llama-3.1-8b-instant")

CHUNK_SIZE = 512
CHUNK_OVERLAP = 64
RETRIEVER_K = 4
FETCH_K = 12  # MMR candidate pool
MMR_LAMBDA = 0.65  # diversity vs relevance (0=max diversity, 1=max relevance)

# ── System prompt ─────────────────────────────────────────────────────────────
RAG_PROMPT = ChatPromptTemplate.from_template(
    """You are an expert Scientific Research Assistant. Use the following pieces of \
retrieved context to answer the question at the end.
If you don't know the answer, say that you don't know based on the documents—do not \
try to make up an answer.
Always structure your answer clearly with bullet points and reference the paper or \
author if mentioned in the context.

Context:
{context}

Question: {question}
Helpful Answer:"""
)


# ── Auto-detect (cached) ───────────────────────────────────────────────────────
@cache
def _ollama_available() -> bool:
    """Check whether Ollama is reachable.  Result is cached after first call."""
    import requests  # pylint: disable=import-outside-toplevel

    try:
        response = requests.get(f"{OLLAMA_HOST}/api/version", timeout=2)
        return response.ok
    except Exception:
        return False


# ── Embeddings (lazy singleton) ────────────────────────────────────────────────
_embeddings_cache = None


def _get_embeddings():
    global _embeddings_cache
    if _embeddings_cache is not None:
        return _embeddings_cache
    provider = (
        EMBED_PROVIDER
        if EMBED_PROVIDER is not None
        else ("ollama" if _ollama_available() else "huggingface")
    )
    if provider == "ollama":
        emb = OllamaEmbeddings(model=EMBED_MODEL)
    else:
        emb = HuggingFaceEmbeddings(model_name=HF_EMBED_MODEL)
    _embeddings_cache = emb
    return emb


# ── LLM (lazy) ────────────────────────────────────────────────────────────────
def _get_llm():
    provider = (
        LLM_PROVIDER
        if LLM_PROVIDER is not None
        else ("ollama" if _ollama_available() else "google")
    )
    if provider == "google":
        return ChatGoogleGenerativeAI(
            model=GOOGLE_MODEL,
            temperature=0.1,
            max_output_tokens=512,
            max_retries=3,
        )
    if provider == "groq":
        from langchain_groq import ChatGroq  # type: ignore[import-untyped]  # noqa: PLC0415  # pylint: disable=import-outside-toplevel

        return ChatGroq(
            model=GROQ_MODEL,
            temperature=0.1,
            max_tokens=512,
        )
    return ChatOllama(
        model=LLM_MODEL,
        temperature=0.1,
        num_ctx=4096,
        num_predict=512,
    )


# ── ChromaDB cache (avoid reloading from disk) ───────────────────────────────
_db_cache: Chroma | None = None


def _load_store() -> Chroma | None:
    """Load existing ChromaDB if it has data (cached after first load)."""
    global _db_cache
    if _db_cache is not None:
        return _db_cache
    if CHROMA_DIR.exists():
        db = Chroma(
            persist_directory=str(CHROMA_DIR), embedding_function=_get_embeddings()
        )
        try:
            # limit=1 avoids fetching all docs just to check emptiness
            if len(db.get(limit=1)["ids"]) > 0:
                _db_cache = db
                return db
        except Exception:
            pass
    return None


def _invalidate_cache():
    """Drop the cached ChromaDB handle (call after writes)."""
    global _db_cache
    _db_cache = None


def get_doc_count() -> int:
    db = _load_store()
    if db is None:
        return 0
    try:
        return db._collection.count()
    except Exception:
        return 0


def get_collection_stats() -> dict:
    """Return per-source_type chunk counts."""
    db = _load_store()
    if db is None:
        return {"preloaded": 0, "uploaded": 0, "total": 0}
    try:
        total = db._collection.count()
        pre = len(db.get(where={"source_type": "preloaded"})["ids"])
        up = len(db.get(where={"source_type": "uploaded"})["ids"])
        return {"preloaded": pre, "uploaded": up, "total": total}
    except Exception:
        return {"preloaded": 0, "uploaded": 0, "total": 0}


def retrieve_for_query(
    query: str, source_type: str | None = None
) -> tuple[list[dict], list[dict]]:
    """Single retrieval returning (context_chunks, sources) for a query.

    Eliminates redundant ChromaDB queries — use this instead of calling
    get_context_for_query + get_sources_for_query separately.
    """
    db = _load_store()
    if db is None:
        return [], []
    search_kwargs: dict = {
        "k": RETRIEVER_K,
        "fetch_k": FETCH_K,
        "lambda_mult": MMR_LAMBDA,
    }
    if source_type and source_type != "all":
        search_kwargs["filter"] = {"source_type": source_type}
    retriever = db.as_retriever(search_type="mmr", search_kwargs=search_kwargs)
    docs = retriever.invoke(query)
    seen, context_chunks, sources = set(), [], []
    for d in docs:
        src = d.metadata.get("source_file", d.metadata.get("source", "?"))
        raw_page = d.metadata.get("page")
        page = int(raw_page) if raw_page is not None else None
        key = f"{src}:{raw_page}"
        if key not in seen:
            seen.add(key)
            page_display = page + 1 if page is not None else None
            context_chunks.append(
                {
                    "file": src,
                    "page": page_display,
                    "content": d.page_content.strip(),
                    "source_type": d.metadata.get("source_type", ""),
                }
            )
            sources.append(
                {
                    "file": src,
                    "page": page_display,
                    "snippet": d.page_content[:180].replace("\n", " "),
                    "source_type": d.metadata.get("source_type", ""),
                }
            )
    return context_chunks, sources


def get_context_for_query(query: str, source_type: str | None = None) -> list[dict]:
    """Return full context chunks for a query — delegates to retrieve_for_query."""
    ctx, _ = retrieve_for_query(query, source_type)
    return ctx


def get_sources_for_query(query: str, source_type: str | None = None) -> list[dict]:
    """Return source metadata for a given query — delegates to retrieve_for_query."""
    _, src = retrieve_for_query(query, source_type)
    return src


def clear_vector_store():
    """Wipe ChromaDB to start fresh."""
    _invalidate_cache()
    if CHROMA_DIR.exists():
        shutil.rmtree(CHROMA_DIR)
    CHROMA_DIR.mkdir(parents=True, exist_ok=True)


def clear_uploads():
    """Wipe only the uploaded chunks and the uploads directory."""
    _invalidate_cache()
    db = _load_store()
    if db:
        try:
            existing = db.get(where={"source_type": "uploaded"})
            if existing and existing["ids"]:
                db.delete(existing["ids"])
        except Exception:
            pass
    if UPLOAD_DIR.exists():
        shutil.rmtree(UPLOAD_DIR)
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


# ── Document loaders ──────────────────────────────────────────────────────────
_SUPPORTED = {".pdf", ".txt", ".md", ".markdown"}


def _load_pdf(path: Path) -> list[Document]:
    from pypdf import PdfReader  # pylint: disable=import-outside-toplevel

    reader = PdfReader(str(path))
    docs: list[Document] = []
    for i, page in enumerate(reader.pages):
        text = page.extract_text()
        if text.strip():
            docs.append(
                Document(page_content=text, metadata={"page": i, "source": path.name})
            )
    return docs


def load_file(path: Path) -> list[Document]:
    ext = path.suffix.lower()
    try:
        if ext == ".pdf":
            return _load_pdf(path)
        elif ext in {".md", ".markdown", ".txt"}:
            text = path.read_text(encoding="utf-8")
            if text.strip():
                return [Document(page_content=text, metadata={"source": path.name})]
    except Exception as e:
        print(f"[WARN] Could not load {path.name}: {e}")
    return []


def ingest_documents(
    source_type: str = "preloaded",
    progress_callback=None,
) -> tuple[int, int]:
    """
    Ingest documents from the source_type-specific directory.
    source_type: "preloaded" → DATA_DIR, "uploaded" → UPLOAD_DIR.
    Old chunks of the same source_type are replaced to avoid duplicates.
    Returns (num_files_processed, num_chunks_added).
    """
    directory = DATA_DIR if source_type == "preloaded" else UPLOAD_DIR

    if not directory.exists():
        return 0, 0

    files: list[Path] = [
        f for f in directory.rglob("*") if f.suffix.lower() in _SUPPORTED
    ]
    if not files:
        return 0, 0

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""],
    )

    all_chunks: list[Document] = []
    processed = 0

    for i, f in enumerate(files):
        if progress_callback:
            progress_callback(i, len(files), f.name)
        docs = load_file(f)
        if docs:
            chunks = splitter.split_documents(docs)
            for c in chunks:
                c.metadata["source_file"] = f.name
                c.metadata["source_type"] = source_type
            all_chunks.extend(chunks)
            processed += 1

    if not all_chunks:
        return 0, 0

    _invalidate_cache()
    db = _load_store()
    if db:
        try:
            existing = db.get(where={"source_type": source_type})
            if existing and existing["ids"]:
                db.delete(existing["ids"])
        except Exception:
            pass
        db.add_documents(all_chunks)
    else:
        Chroma.from_documents(
            all_chunks,
            _get_embeddings(),
            persist_directory=str(CHROMA_DIR),
        )

    return processed, len(all_chunks)


# ── RAG Chain builder ─────────────────────────────────────────────────────────
def _format_docs(docs: list[Document]) -> str:
    parts = []
    for d in docs:
        src = d.metadata.get("source_file", d.metadata.get("source", "unknown"))
        raw_page = d.metadata.get("page")
        page = int(raw_page) if raw_page is not None else None
        header = f"[{src}" + (f" · p.{page + 1}" if page is not None else "") + "]"
        parts.append(f"{header}\n{d.page_content.strip()}")
    return "\n\n".join(parts)


def build_chain(source_type: str | None = None):
    """
    Returns (chain, retriever) or (None, None) if no documents indexed.
    Pass source_type="preloaded" / "uploaded" to filter retrieval.
    """
    db = _load_store()
    if db is None:
        return None, None

    search_kwargs: dict = {
        "k": RETRIEVER_K,
        "fetch_k": FETCH_K,
        "lambda_mult": MMR_LAMBDA,
    }
    if source_type:
        search_kwargs["filter"] = {"source_type": source_type}

    retriever = db.as_retriever(
        search_type="mmr",
        search_kwargs=search_kwargs,
    )

    llm = _get_llm()

    chain = (
        RunnableParallel(
            context=retriever | _format_docs,
            question=RunnablePassthrough(),
        )
        | RAG_PROMPT
        | llm
        | StrOutputParser()
    )

    return chain, retriever
