"""
engine.py — Optimized RAG Engine
- LLM  : phi4-mini via Ollama (ROCm/AMD GPU accelerated)
- Embed: nomic-embed-text via Ollama
- Store: ChromaDB (persistent, MMR retrieval)
- AMD RX 6500M (RDNA2 / gfx1034) → HSA_OVERRIDE_GFX_VERSION=10.3.0
"""
# ruff: noqa: E402

import os
import shutil
from pathlib import Path
from typing import List, Tuple, Optional

# ── ROCm / AMD GPU hints (set before importing anything GPU-related) ──────────
os.environ.setdefault("HSA_OVERRIDE_GFX_VERSION", "10.3.0")  # RDNA2 gfx1034
os.environ.setdefault("OLLAMA_NUM_GPU", "1")
os.environ.setdefault("OLLAMA_GPU_OVERHEAD", "256MiB")  # keep headroom

from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    UnstructuredMarkdownLoader,
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain_core.documents import Document

# ── Config ────────────────────────────────────────────────────────────────────
DATA_DIR = Path("./data")
CHROMA_DIR = Path("./chroma_db")
EMBED_MODEL = "nomic-embed-text"
LLM_MODEL = "phi4-mini"

CHUNK_SIZE = 512
CHUNK_OVERLAP = 64
RETRIEVER_K = 4
FETCH_K = 12  # MMR candidate pool
MMR_LAMBDA = 0.65  # diversity vs relevance (0=max diversity, 1=max relevance)

# ── System prompt ─────────────────────────────────────────────────────────────
RAG_PROMPT = ChatPromptTemplate.from_template(
    """You are a knowledgeable assistant. Use ONLY the context below to answer.
If the answer is not in the context, say "I don't have enough information in the \
uploaded documents to answer that."

--- CONTEXT ---
{context}
--------------

Question: {question}

Provide a clear, concise, accurate answer. Cite the document source when relevant."""
)

# ── Embeddings (singleton) ────────────────────────────────────────────────────
_embeddings = OllamaEmbeddings(model=EMBED_MODEL)


# ── Vector store helpers ──────────────────────────────────────────────────────
def _load_store() -> Optional[Chroma]:
    """Load existing ChromaDB if it has data."""
    if CHROMA_DIR.exists():
        db = Chroma(persist_directory=str(CHROMA_DIR), embedding_function=_embeddings)
        try:
            count = db._collection.count()
            if count > 0:
                return db
        except Exception:
            pass
    return None


def get_doc_count() -> int:
    db = _load_store()
    if db is None:
        return 0
    try:
        return db._collection.count()
    except Exception:
        return 0


def clear_vector_store():
    """Wipe ChromaDB to start fresh."""
    if CHROMA_DIR.exists():
        shutil.rmtree(CHROMA_DIR)
    CHROMA_DIR.mkdir(parents=True, exist_ok=True)


# ── Document loaders ──────────────────────────────────────────────────────────
_SUPPORTED = {".pdf", ".txt", ".md", ".markdown"}


def load_file(path: Path) -> List[Document]:
    ext = path.suffix.lower()
    try:
        if ext == ".pdf":
            return PyPDFLoader(str(path)).load()
        elif ext in {".md", ".markdown"}:
            try:
                return UnstructuredMarkdownLoader(str(path)).load()
            except Exception:
                return TextLoader(str(path), encoding="utf-8").load()
        elif ext == ".txt":
            return TextLoader(str(path), encoding="utf-8").load()
    except Exception as e:
        print(f"[WARN] Could not load {path.name}: {e}")
    return []


def ingest_documents(
    directory: Optional[Path] = None,
    extra_files: Optional[List[Path]] = None,
    reset: bool = False,
    progress_callback=None,
) -> Tuple[int, int]:
    """
    Ingest documents from `directory` and/or `extra_files`.
    Returns (num_files_processed, num_chunks_added).
    """
    if reset:
        clear_vector_store()

    directory = directory or DATA_DIR
    files: List[Path] = []

    if directory.exists():
        files += [f for f in directory.rglob("*") if f.suffix.lower() in _SUPPORTED]

    if extra_files:
        files += extra_files

    if not files:
        return 0, 0

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""],
    )

    all_chunks: List[Document] = []
    processed = 0

    for i, f in enumerate(files):
        if progress_callback:
            progress_callback(i, len(files), f.name)
        docs = load_file(f)
        if docs:
            chunks = splitter.split_documents(docs)
            # Tag each chunk with source filename
            for c in chunks:
                c.metadata["source_file"] = f.name
            all_chunks.extend(chunks)
            processed += 1

    if not all_chunks:
        return 0, 0

    # Upsert into ChromaDB
    Chroma.from_documents(
        all_chunks,
        _embeddings,
        persist_directory=str(CHROMA_DIR),
    )

    return processed, len(all_chunks)


# ── RAG Chain builder ─────────────────────────────────────────────────────────
def _format_docs(docs: List[Document]) -> str:
    parts = []
    for d in docs:
        src = d.metadata.get("source_file", d.metadata.get("source", "unknown"))
        page = d.metadata.get("page", "")
        header = f"[{src}" + (f" · p.{page+1}" if page != "" else "") + "]"
        parts.append(f"{header}\n{d.page_content.strip()}")
    return "\n\n".join(parts)


def build_chain(streaming: bool = False):
    """
    Returns (chain, retriever) or (None, None) if no documents indexed.
    chain accepts a plain string question and returns str (or stream).
    """
    db = _load_store()
    if db is None:
        return None, None

    retriever = db.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": RETRIEVER_K,
            "fetch_k": FETCH_K,
            "lambda_mult": MMR_LAMBDA,
        },
    )

    llm = ChatOllama(
        model=LLM_MODEL,
        temperature=0.1,
        num_ctx=4096,  # fits RX 6500M 4 GB VRAM with phi4-mini
        num_predict=512,
    )

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


def get_sources_for_query(query: str) -> List[dict]:
    """Return source metadata for a given query (for display)."""
    db = _load_store()
    if db is None:
        return []
    retriever = db.as_retriever(
        search_type="mmr",
        search_kwargs={"k": RETRIEVER_K, "fetch_k": FETCH_K, "lambda_mult": MMR_LAMBDA},
    )
    docs = retriever.invoke(query)
    seen, sources = set(), []
    for d in docs:
        src = d.metadata.get("source_file", d.metadata.get("source", "?"))
        page = d.metadata.get("page", "")
        key = f"{src}:{page}"
        if key not in seen:
            seen.add(key)
            sources.append(
                {
                    "file": src,
                    "page": int(page) + 1 if page != "" else None,
                    "snippet": d.page_content[:180].replace("\n", " "),
                }
            )
    return sources
