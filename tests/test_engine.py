from unittest.mock import patch, MagicMock
from pathlib import Path

with patch("langchain_ollama.OllamaEmbeddings"):
    import engine


# ── Helpers ─────────────────────────────────────────────────────────────────


def _make_mock_db(ids=None, docs=None, metadatas=None):
    mock_db = MagicMock()
    if ids is None:
        ids = []
    mock_db.get.return_value = {
        "ids": ids,
        "documents": docs or [],
        "metadatas": metadatas or [{} for _ in ids],
    }
    mock_db._collection.count.return_value = len(ids)
    mock_db.as_retriever.return_value = MagicMock()
    mock_db.as_retriever.return_value.invoke.return_value = []
    return mock_db


# ── _load_store ─────────────────────────────────────────────────────────────


def test_load_store_cached():
    engine._db_cache = "cached"
    assert engine._load_store() == "cached"
    engine._db_cache = None


def test_load_store_dir_not_exists():
    with patch("engine.CHROMA_DIR") as mock_dir:
        mock_dir.exists.return_value = False
        assert engine._load_store() is None


def test_load_store_empty():
    with patch("engine.CHROMA_DIR") as mock_dir:
        mock_dir.exists.return_value = True
        with patch("engine.Chroma") as MockChroma:
            mock_db = MockChroma.return_value
            mock_db.get.return_value = {"ids": []}
            mock_db._collection.count.return_value = 0
            assert engine._load_store() is None


def test_load_store_with_data():
    with patch("engine.CHROMA_DIR") as mock_dir:
        mock_dir.exists.return_value = True
        with patch("engine.Chroma") as MockChroma:
            mock_db = MockChroma.return_value
            mock_db.get.return_value = {"ids": ["1", "2"]}
            mock_db._collection.count.return_value = 2
            result = engine._load_store()
            assert result is mock_db
            assert engine._db_cache is mock_db
    engine._db_cache = None


def test_load_store_exception():
    with patch("engine.CHROMA_DIR") as mock_dir:
        mock_dir.exists.return_value = True
        with patch("engine.Chroma") as MockChroma:
            mock_db = MockChroma.return_value
            mock_db.get.side_effect = Exception("db error")
            assert engine._load_store() is None


# ── _invalidate_cache ───────────────────────────────────────────────────────


def test_invalidate_cache():
    engine._db_cache = "something"
    engine._invalidate_cache()
    assert engine._db_cache is None


# ── get_doc_count ────────────────────────────────────────────────────────────


def test_get_doc_count_no_db():
    with patch("engine._load_store", return_value=None):
        assert engine.get_doc_count() == 0


def test_get_doc_count_with_db():
    mock_db = _make_mock_db(ids=["a", "b", "c"])
    with patch("engine._load_store", return_value=mock_db):
        assert engine.get_doc_count() == 3


def test_get_doc_count_exception():
    mock_db = MagicMock()
    mock_db._collection.count.side_effect = Exception("error")
    with patch("engine._load_store", return_value=mock_db):
        assert engine.get_doc_count() == 0


# ── get_collection_stats ────────────────────────────────────────────────────


def test_get_collection_stats_no_db():
    with patch("engine._load_store", return_value=None):
        assert engine.get_collection_stats() == {
            "preloaded": 0,
            "uploaded": 0,
            "total": 0,
        }


def test_get_collection_stats_with_data():
    mock_db = MagicMock()
    mock_db._collection.count.return_value = 3

    def get_side_effect(**kwargs):
        where = kwargs.get("where", {})
        source_type = where.get("source_type") if where else None
        if source_type is None:
            return {"ids": ["1", "2", "3"]}
        elif source_type == "preloaded":
            return {"ids": ["1", "2"]}
        elif source_type == "uploaded":
            return {"ids": ["3"]}
        return {"ids": []}

    mock_db.get.side_effect = get_side_effect
    with patch("engine._load_store", return_value=mock_db):
        assert engine.get_collection_stats() == {
            "preloaded": 2,
            "uploaded": 1,
            "total": 3,
        }


def test_get_collection_stats_exception():
    mock_db = MagicMock()
    mock_db.get.side_effect = Exception("error")
    with patch("engine._load_store", return_value=mock_db):
        assert engine.get_collection_stats() == {
            "preloaded": 0,
            "uploaded": 0,
            "total": 0,
        }


# ── get_context_for_query ────────────────────────────────────────────────────


def test_get_context_for_query_no_db():
    with patch("engine._load_store", return_value=None):
        assert engine.get_context_for_query("test") == []


def test_get_context_for_query_with_db():
    from langchain_core.documents import Document

    mock_db = MagicMock()
    mock_retriever = MagicMock()
    mock_db.as_retriever.return_value = mock_retriever
    mock_retriever.invoke.return_value = [
        Document(
            page_content="Some content",
            metadata={
                "source_file": "paper.pdf",
                "page": 0,
                "source_type": "preloaded",
            },
        ),
    ]
    with patch("engine._load_store", return_value=mock_db):
        result = engine.get_context_for_query("query")
        assert len(result) == 1
        assert result[0]["file"] == "paper.pdf"
        assert result[0]["page"] == 1
        assert result[0]["source_type"] == "preloaded"
        assert result[0]["content"] == "Some content"


def test_get_context_for_query_with_filter():
    from langchain_core.documents import Document

    mock_db = MagicMock()
    mock_retriever = MagicMock()
    mock_db.as_retriever.return_value = mock_retriever
    mock_retriever.invoke.return_value = [
        Document(
            page_content="Data",
            metadata={"source_file": "doc.txt", "source_type": "uploaded"},
        ),
    ]
    with patch("engine._load_store", return_value=mock_db):
        engine.get_context_for_query("query", source_type="uploaded")
        call_kwargs = mock_db.as_retriever.call_args[1]
        assert call_kwargs["search_kwargs"]["filter"] == {"source_type": "uploaded"}


def test_get_context_for_query_dedup():
    from langchain_core.documents import Document

    mock_db = MagicMock()
    mock_retriever = MagicMock()
    mock_db.as_retriever.return_value = mock_retriever
    mock_retriever.invoke.return_value = [
        Document(page_content="A", metadata={"source_file": "f.pdf", "page": 0}),
        Document(page_content="B", metadata={"source_file": "f.pdf", "page": 0}),
    ]
    with patch("engine._load_store", return_value=mock_db):
        result = engine.get_context_for_query("query")
        assert len(result) == 1


def test_get_context_for_query_no_page():
    from langchain_core.documents import Document

    mock_db = MagicMock()
    mock_retriever = MagicMock()
    mock_db.as_retriever.return_value = mock_retriever
    mock_retriever.invoke.return_value = [
        Document(
            page_content="C",
            metadata={"source_file": "doc.txt", "source_type": "preloaded"},
        ),
    ]
    with patch("engine._load_store", return_value=mock_db):
        result = engine.get_context_for_query("query")
        assert result[0]["page"] is None


# ── get_sources_for_query ───────────────────────────────────────────────────


def test_get_sources_for_query_no_db():
    with patch("engine._load_store", return_value=None):
        assert engine.get_sources_for_query("test") == []


def test_get_sources_for_query_with_db():
    from langchain_core.documents import Document

    mock_db = MagicMock()
    mock_retriever = MagicMock()
    mock_db.as_retriever.return_value = mock_retriever
    mock_retriever.invoke.return_value = [
        Document(
            page_content="Source snippet longer than 180 chars " + "x" * 200,
            metadata={"source_file": "src.pdf", "page": 2, "source_type": "preloaded"},
        ),
    ]
    with patch("engine._load_store", return_value=mock_db):
        result = engine.get_sources_for_query("query")
        assert len(result) == 1
        assert result[0]["file"] == "src.pdf"
        assert result[0]["page"] == 3
        assert "snippet" in result[0]


def test_get_sources_for_query_with_filter():
    mock_db = MagicMock()
    mock_retriever = MagicMock()
    mock_db.as_retriever.return_value = mock_retriever
    with patch("engine._load_store", return_value=mock_db):
        engine.get_sources_for_query("query", source_type="uploaded")
        call_kwargs = mock_db.as_retriever.call_args[1]
        assert call_kwargs["search_kwargs"]["filter"] == {"source_type": "uploaded"}


# ── clear_vector_store ───────────────────────────────────────────────────────


def test_clear_vector_store_dir_exists():
    with patch("engine.CHROMA_DIR") as mock_dir:
        mock_dir.exists.return_value = True
        with patch("shutil.rmtree") as mock_rm:
            engine.clear_vector_store()
            mock_rm.assert_called_once_with(mock_dir)
            mock_dir.mkdir.assert_called_once_with(parents=True, exist_ok=True)


def test_clear_vector_store_dir_not_exists():
    with patch("engine.CHROMA_DIR") as mock_dir:
        mock_dir.exists.return_value = False
        with patch("shutil.rmtree") as mock_rm:
            engine.clear_vector_store()
            mock_rm.assert_not_called()
            mock_dir.mkdir.assert_called_once_with(parents=True, exist_ok=True)


# ── clear_uploads ────────────────────────────────────────────────────────────


def test_clear_uploads_with_uploaded_docs():
    mock_db = MagicMock()
    mock_db.get.return_value = {"ids": ["1", "2"]}
    with patch("engine._load_store", return_value=mock_db):
        with patch("engine.UPLOAD_DIR") as mock_dir:
            mock_dir.exists.return_value = False
            engine.clear_uploads()
            mock_db.delete.assert_called_once_with(["1", "2"])


def test_clear_uploads_no_uploaded_docs():
    mock_db = MagicMock()
    mock_db.get.return_value = {"ids": []}
    with patch("engine._load_store", return_value=mock_db):
        with patch("engine.UPLOAD_DIR") as mock_dir:
            mock_dir.exists.return_value = False
            engine.clear_uploads()
            mock_db.delete.assert_not_called()


def test_clear_uploads_db_error():
    mock_db = MagicMock()
    mock_db.get.side_effect = Exception("error")
    with patch("engine._load_store", return_value=mock_db):
        with patch("engine.UPLOAD_DIR") as mock_dir:
            mock_dir.exists.return_value = True
            with patch("shutil.rmtree") as mock_rm:
                engine.clear_uploads()
                mock_rm.assert_called_once_with(mock_dir)


def test_clear_uploads_remove_dir():
    mock_db = MagicMock()
    mock_db.get.return_value = {"ids": []}
    with patch("engine._load_store", return_value=mock_db):
        with patch("engine.UPLOAD_DIR") as mock_dir:
            mock_dir.exists.return_value = True
            with patch("shutil.rmtree") as mock_rm:
                engine.clear_uploads()
                mock_rm.assert_called_once_with(mock_dir)


# ── load_file ────────────────────────────────────────────────────────────────


def test_load_file_pdf():
    with patch("engine.PyPDFLoader") as MockLoader:
        mock_loader = MockLoader.return_value
        mock_loader.load.return_value = ["doc1", "doc2"]
        result = engine.load_file(Path("test.pdf"))
        assert result == ["doc1", "doc2"]
        MockLoader.assert_called_once_with("test.pdf")


def test_load_file_txt():
    with patch("engine.TextLoader") as MockLoader:
        mock_loader = MockLoader.return_value
        mock_loader.load.return_value = ["content"]
        result = engine.load_file(Path("test.txt"))
        assert result == ["content"]
        MockLoader.assert_called_once_with("test.txt", encoding="utf-8")


def test_load_file_md():
    with patch("engine.UnstructuredMarkdownLoader") as MockMdLoader:
        mock_loader = MockMdLoader.return_value
        mock_loader.load.return_value = ["md content"]
        result = engine.load_file(Path("test.md"))
        assert result == ["md content"]
        MockMdLoader.assert_called_once_with("test.md")


def test_load_file_markdown():
    with patch("engine.UnstructuredMarkdownLoader") as MockMdLoader:
        mock_loader = MockMdLoader.return_value
        mock_loader.load.return_value = ["md content"]
        result = engine.load_file(Path("test.markdown"))
        assert result == ["md content"]
        MockMdLoader.assert_called_once_with("test.markdown")


def test_load_file_md_fallback():
    with patch("engine.UnstructuredMarkdownLoader") as MockMdLoader:
        MockMdLoader.return_value.load.side_effect = Exception("parser error")
        with patch("engine.TextLoader") as MockTxtLoader:
            mock_txt = MockTxtLoader.return_value
            mock_txt.load.return_value = ["fallback content"]
            result = engine.load_file(Path("test.md"))
            assert result == ["fallback content"]


def test_load_file_error():
    with patch("engine.PyPDFLoader") as MockLoader:
        MockLoader.return_value.load.side_effect = Exception("load error")
        result = engine.load_file(Path("test.pdf"))
        assert result == []


def test_load_file_unsupported_ext():
    assert engine.load_file(Path("test.invalid_ext")) == []


# ── ingest_documents ─────────────────────────────────────────────────────────


def test_ingest_dir_not_exists():
    with patch("engine.DATA_DIR") as mock_dir:
        mock_dir.exists.return_value = False
        assert engine.ingest_documents("preloaded") == (0, 0)


def test_ingest_no_files():
    with patch("engine.DATA_DIR") as mock_dir:
        mock_dir.exists.return_value = True
        mock_dir.rglob.return_value = []
        assert engine.ingest_documents("preloaded") == (0, 0)


def test_ingest_with_files():
    from langchain_core.documents import Document

    mock_file = MagicMock(spec=Path)
    mock_file.suffix = ".txt"
    mock_file.name = "test.txt"

    with patch("engine.DATA_DIR") as mock_dir:
        mock_dir.exists.return_value = True
        mock_dir.rglob.return_value = [mock_file]
        with patch("engine.load_file", return_value=[Document(page_content="hello")]):
            with patch("engine.RecursiveCharacterTextSplitter") as MockSplitter:
                splitter = MockSplitter.return_value
                splitter.split_documents.return_value = [
                    Document(page_content="chunk1"),
                ]
                with patch("engine._load_store", return_value=None):
                    with patch("engine.Chroma") as MockChroma:
                        result = engine.ingest_documents("preloaded")
                        assert result == (1, 1)
                        MockChroma.from_documents.assert_called_once()


def test_ingest_with_existing_db():
    from langchain_core.documents import Document

    mock_file = MagicMock(spec=Path)
    mock_file.suffix = ".txt"
    mock_file.name = "test.txt"

    mock_db = MagicMock()
    mock_db.get.return_value = {"ids": ["old_id"]}

    with patch("engine.DATA_DIR") as mock_dir:
        mock_dir.exists.return_value = True
        mock_dir.rglob.return_value = [mock_file]
        with patch("engine.load_file", return_value=[Document(page_content="hello")]):
            with patch("engine.RecursiveCharacterTextSplitter") as MockSplitter:
                splitter = MockSplitter.return_value
                splitter.split_documents.return_value = [
                    Document(page_content="new_chunk"),
                ]
                with patch("engine._load_store", return_value=mock_db):
                    result = engine.ingest_documents("preloaded")
                    assert result == (1, 1)
                    mock_db.delete.assert_called_once_with(["old_id"])
                    mock_db.add_documents.assert_called_once()


def test_ingest_no_chunks():
    mock_file = MagicMock(spec=Path)
    mock_file.suffix = ".txt"
    mock_file.name = "test.txt"

    with patch("engine.DATA_DIR") as mock_dir:
        mock_dir.exists.return_value = True
        mock_dir.rglob.return_value = [mock_file]
        with patch("engine.load_file", return_value=[]):
            result = engine.ingest_documents("preloaded")
            assert result == (0, 0)


def test_ingest_with_progress():
    from langchain_core.documents import Document

    mock_file = MagicMock(spec=Path)
    mock_file.suffix = ".txt"
    mock_file.name = "test.txt"

    callback = MagicMock()

    with patch("engine.DATA_DIR") as mock_dir:
        mock_dir.exists.return_value = True
        mock_dir.rglob.return_value = [mock_file]
        with patch("engine.load_file", return_value=[Document(page_content="data")]):
            with patch("engine.RecursiveCharacterTextSplitter") as MockSplitter:
                splitter = MockSplitter.return_value
                splitter.split_documents.return_value = [
                    Document(page_content="chunk"),
                ]
                with patch("engine._load_store", return_value=None):
                    with patch("engine.Chroma"):
                        engine.ingest_documents("preloaded", progress_callback=callback)
                        callback.assert_called_once_with(0, 1, "test.txt")


def test_ingest_delete_error():
    from langchain_core.documents import Document

    mock_file = MagicMock(spec=Path)
    mock_file.suffix = ".txt"
    mock_file.name = "test.txt"

    mock_db = MagicMock()
    mock_db.get.side_effect = Exception("get error")

    with patch("engine.DATA_DIR") as mock_dir:
        mock_dir.exists.return_value = True
        mock_dir.rglob.return_value = [mock_file]
        with patch("engine.load_file", return_value=[Document(page_content="data")]):
            with patch("engine.RecursiveCharacterTextSplitter") as MockSplitter:
                splitter = MockSplitter.return_value
                splitter.split_documents.return_value = [
                    Document(page_content="chunk"),
                ]
                with patch("engine._load_store", return_value=mock_db):
                    engine.ingest_documents("preloaded")
                    mock_db.add_documents.assert_called_once()


# ── _format_docs ─────────────────────────────────────────────────────────────


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


# ── build_chain ──────────────────────────────────────────────────────────────


def test_build_chain_no_db():
    with patch("engine._load_store", return_value=None):
        chain, retriever = engine.build_chain()
        assert chain is None
        assert retriever is None


def test_build_chain_with_db():
    mock_db = _make_mock_db(ids=["1"])
    with patch("engine._load_store", return_value=mock_db):
        with patch("engine.ChatOllama"):
            chain, retriever = engine.build_chain()
            assert chain is not None
            assert retriever is not None


def test_build_chain_with_filter():
    mock_db = _make_mock_db(ids=["1"])
    with patch("engine._load_store", return_value=mock_db):
        with patch("engine.ChatOllama"):
            engine.build_chain(source_type="preloaded")
            call_kwargs = mock_db.as_retriever.call_args[1]
            assert call_kwargs["search_kwargs"]["filter"] == {
                "source_type": "preloaded"
            }
