from unittest.mock import patch
from pathlib import Path

# Patch OllamaEmbeddings at the module level to avoid calling Ollama during import
with patch("langchain_ollama.OllamaEmbeddings") as mock_embed:
    import engine


def test_get_doc_count_empty():
    with patch("engine._load_store", return_value=None):
        assert engine.get_doc_count() == 0


def test_load_file_unsupported_ext():
    assert engine.load_file(Path("test.invalid_ext")) == []


def test_format_docs_basic():
    from langchain_core.documents import Document

    docs = [
        Document(page_content="Hello world", metadata={"source_file": "doc1.txt"}),
        Document(
            page_content="Second chunk", metadata={"source_file": "doc2.pdf", "page": 1}
        ),
    ]
    formatted = engine._format_docs(docs)
    assert "[doc1.txt]\nHello world" in formatted
    assert "[doc2.pdf · p.2]\nSecond chunk" in formatted


def test_format_docs_page_none():
    from langchain_core.documents import Document

    docs = [
        Document(
            page_content="Content here",
            metadata={"source_file": "doc.pdf", "page": None},
        ),
    ]
    formatted = engine._format_docs(docs)
    assert "[doc.pdf]\nContent here" in formatted


def test_format_docs_page_missing():
    from langchain_core.documents import Document

    docs = [
        Document(page_content="No page key", metadata={"source_file": "doc.txt"}),
    ]
    formatted = engine._format_docs(docs)
    assert "[doc.txt]\nNo page key" in formatted


def test_get_context_for_query_no_db():
    with patch("engine._load_store", return_value=None):
        assert engine.get_context_for_query("test") == []


def test_get_sources_for_query_no_db():
    with patch("engine._load_store", return_value=None):
        assert engine.get_sources_for_query("test") == []


def test_get_collection_stats_no_db():
    with patch("engine._load_store", return_value=None):
        assert engine.get_collection_stats() == {
            "preloaded": 0,
            "uploaded": 0,
            "total": 0,
        }


def test_clear_vector_store_no_db():
    with patch("engine._load_store", return_value=None):
        with patch("engine.CHROMA_DIR") as mock_dir:
            mock_dir.exists.return_value = False
            engine.clear_vector_store()


def test_clear_uploads_no_db():
    with patch("engine._load_store", return_value=None):
        with patch("engine.UPLOAD_DIR") as mock_dir:
            mock_dir.exists.return_value = False
            engine.clear_uploads()
