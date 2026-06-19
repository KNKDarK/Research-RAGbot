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


def test_format_docs():
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
