import sys
import types
from unittest.mock import MagicMock
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

_ORIGINAL_ENGINE = None


@pytest.fixture(autouse=True)
def restore_engine():
    global _ORIGINAL_ENGINE
    if _ORIGINAL_ENGINE is None:
        _ORIGINAL_ENGINE = sys.modules.get("engine")
    yield
    if _ORIGINAL_ENGINE is not None:
        sys.modules["engine"] = _ORIGINAL_ENGINE
    elif "engine" in sys.modules:
        del sys.modules["engine"]


def _make_mock_engine(**overrides):
    mod = types.ModuleType("engine")
    defaults = {
        "DATA_DIR": Path("./data"),
        "UPLOAD_DIR": Path("./uploads"),
        "EMBED_PROVIDER": "ollama",
        "LLM_PROVIDER": "ollama",
        "_ollama_available": MagicMock(return_value=True),
        "EMBED_MODEL": "nomic-embed-text",
        "LLM_MODEL": "phi4-mini",
        "HF_EMBED_MODEL": "all-MiniLM-L6-v2",
        "GOOGLE_MODEL": "gemini-3.5-flash",
        "GROQ_MODEL": "llama-3.1-8b-instant",
        "ingest_documents": MagicMock(return_value=(0, 0)),
        "build_chain": MagicMock(return_value=(None, None)),
        "retrieve_for_query": MagicMock(return_value=([], [])),
        "get_sources_for_query": MagicMock(return_value=[]),
        "get_context_for_query": MagicMock(return_value=[]),
        "get_doc_count": MagicMock(return_value=0),
        "get_collection_stats": MagicMock(
            return_value={"preloaded": 0, "uploaded": 0, "total": 0}
        ),
        "clear_uploads": MagicMock(),
        "clear_vector_store": MagicMock(),
    }
    defaults.update(overrides)
    for k, v in defaults.items():
        setattr(mod, k, v)
    return mod


def _make_ready_engine():
    mock_chain = MagicMock()
    mock_chain.stream.return_value = ["Hello", " world"]
    mock_retriever = MagicMock()
    return _make_mock_engine(
        get_doc_count=MagicMock(return_value=5),
        get_collection_stats=MagicMock(
            return_value={"preloaded": 5, "uploaded": 0, "total": 5}
        ),
        build_chain=MagicMock(return_value=(mock_chain, mock_retriever)),
    )


@pytest.fixture
def app(request):
    marker = request.node.get_closest_marker("engine_overrides")
    overrides = marker.kwargs if marker else {}
    mock_engine = _make_mock_engine(**overrides)
    sys.modules["engine"] = mock_engine
    at = AppTest.from_file("app.py", default_timeout=10)
    at.run()
    return at, mock_engine


def test_app_runs_without_errors(app):
    at, _ = app
    assert at.status == []
    assert len(at.error) == 0


def test_app_shows_empty_state(app):
    at, _ = app
    assert at.session_state["ready"] is False
    assert at.session_state["messages"] == []
    assert at.session_state["chain"] is None


def test_app_has_three_buttons(app):
    at, _ = app
    assert len(at.button) == 3


def test_app_has_source_radio(app):
    at, _ = app
    assert len(at.radio) == 1


def test_app_has_markdown_elements(app):
    at, _ = app
    assert len(at.markdown) >= 10


def test_app_new_chat_button_clears_messages(app):
    at, _ = app
    at.session_state["messages"] = [{"role": "user", "content": "hello"}]
    at.button[2].click().run()
    assert at.session_state["messages"] == []


def test_app_reingest_button_calls_ingest(app):
    at, mock_engine = app
    at.button[0].click().run()
    mock_engine.ingest_documents.assert_called()


def test_app_clear_uploads_button_calls_clear_uploads(app):
    at, mock_engine = app
    at.button[1].click().run()
    mock_engine.clear_uploads.assert_called()


@pytest.mark.engine_overrides(
    get_doc_count=MagicMock(return_value=5),
    get_collection_stats=MagicMock(
        return_value={"preloaded": 5, "uploaded": 0, "total": 5}
    ),
    build_chain=MagicMock(return_value=(MagicMock(), MagicMock())),
)
def test_app_ready_with_documents(app):
    at, _ = app
    assert at.session_state["ready"] is True
    assert at.session_state["chain"] is not None


def test_app_source_radio_changes_state(app):
    at, _ = app
    at.radio[0].set_value("uploaded").run()
    assert at.session_state["source_type"] == "uploaded"


def test_app_show_context_toggle(app):
    at, _ = app
    assert at.session_state["show_context"] is True


def test_app_show_sources_toggle(app):
    at, _ = app
    assert at.session_state["show_sources"] is True


def test_app_chat_message_appears(app):
    mock_engine = _make_ready_engine()
    sys.modules["engine"] = mock_engine
    at = AppTest.from_file("app.py", default_timeout=10)
    at.run()
    at.chat_input[0].set_value("What is RAG?").run()
    messages = at.session_state["messages"]
    assert len(messages) == 2
    assert messages[0]["role"] == "user"
    assert messages[1]["role"] == "assistant"


def test_app_chat_saves_context_and_sources(app):
    mock_chain = MagicMock()
    mock_chain.stream.return_value = ["Answer text"]
    mock_engine = _make_mock_engine(
        get_doc_count=MagicMock(return_value=5),
        get_collection_stats=MagicMock(
            return_value={"preloaded": 5, "uploaded": 0, "total": 5}
        ),
        build_chain=MagicMock(return_value=(mock_chain, MagicMock())),
        retrieve_for_query=MagicMock(
            return_value=(
                [
                    {
                        "file": "doc.pdf",
                        "page": 1,
                        "content": "ctx content",
                        "source_type": "preloaded",
                    }
                ],
                [
                    {
                        "file": "doc.pdf",
                        "page": 1,
                        "snippet": "snip",
                        "source_type": "preloaded",
                    }
                ],
            )
        ),
    )
    sys.modules["engine"] = mock_engine
    at = AppTest.from_file("app.py", default_timeout=10)
    at.run()
    at.chat_input[0].set_value("test").run()
    msgs = at.session_state["messages"]
    assert len(msgs[1].get("sources", [])) >= 1
    assert len(msgs[1].get("context", [])) >= 1


def test_app_chat_without_sources(app):
    mock_chain = MagicMock()
    mock_chain.stream.return_value = ["Answer"]
    mock_engine = _make_mock_engine(
        get_doc_count=MagicMock(return_value=5),
        build_chain=MagicMock(return_value=(mock_chain, MagicMock())),
    )
    sys.modules["engine"] = mock_engine
    at = AppTest.from_file("app.py", default_timeout=10)
    at.run()
    at.chat_input[0].set_value("query").run()
    assert len(at.session_state["messages"]) == 2


def test_app_chat_error_handling(app):
    mock_chain = MagicMock()
    mock_chain.stream.side_effect = Exception("LLM error")
    mock_engine = _make_mock_engine(
        get_doc_count=MagicMock(return_value=5),
        build_chain=MagicMock(return_value=(mock_chain, MagicMock())),
    )
    sys.modules["engine"] = mock_engine
    at = AppTest.from_file("app.py", default_timeout=10)
    at.run()
    at.chat_input[0].set_value("query").run()
    msgs = at.session_state["messages"]
    assert "⚠️ Error:" in msgs[-1]["content"]


def test_app_reingest_with_progress(app):
    mock_engine = _make_mock_engine(
        get_doc_count=MagicMock(return_value=5),
        ingest_documents=MagicMock(return_value=(2, 20)),
        build_chain=MagicMock(return_value=(MagicMock(), MagicMock())),
    )
    sys.modules["engine"] = mock_engine
    at = AppTest.from_file("app.py", default_timeout=10)
    at.run()
    at.button[0].click().run()
    assert at.session_state["ready"] is True


def test_app_clear_uploads_then_rebuilds_chain(app):
    mock_engine = _make_mock_engine(
        get_doc_count=MagicMock(return_value=5),
        build_chain=MagicMock(return_value=(MagicMock(), MagicMock())),
    )
    sys.modules["engine"] = mock_engine
    at = AppTest.from_file("app.py", default_timeout=10)
    at.run()
    at.button[1].click().run()
    assert at.session_state["ready"] is True


def test_app_file_upload_and_ingest(app):
    mock_engine = _make_mock_engine(
        get_doc_count=MagicMock(return_value=0),
        ingest_documents=MagicMock(return_value=(1, 10)),
        build_chain=MagicMock(return_value=(MagicMock(), MagicMock())),
    )
    sys.modules["engine"] = mock_engine
    at = AppTest.from_file("app.py", default_timeout=10)
    at.run()
    at.file_uploader[0].upload("test.pdf", b"fake pdf data")
    at.run()
    at.button[0].click().run()
    mock_engine.ingest_documents.assert_called()


def test_app_file_upload_no_content_error(app):
    mock_engine = _make_mock_engine(
        get_doc_count=MagicMock(return_value=0),
        ingest_documents=MagicMock(return_value=(0, 0)),
    )
    sys.modules["engine"] = mock_engine
    at = AppTest.from_file("app.py", default_timeout=10)
    at.run()
    at.file_uploader[0].upload("empty.pdf", b"")
    at.run()
    at.button[0].click().run()
    mock_engine.ingest_documents.assert_called()


def test_app_file_upload_triggers_ingest(app):
    mock_engine = _make_mock_engine(
        get_doc_count=MagicMock(return_value=0),
        ingest_documents=MagicMock(return_value=(1, 10)),
        build_chain=MagicMock(return_value=(MagicMock(), MagicMock())),
    )
    sys.modules["engine"] = mock_engine
    at = AppTest.from_file("app.py", default_timeout=10)
    at.run()
    at.file_uploader[0].upload("test.pdf", b"fake pdf data")
    at.run()
    at.button[0].click().run()
    mock_engine.ingest_documents.assert_called()
    assert "doc_count" in at.session_state


def test_app_chat_displays_context(app):
    mock_chain = MagicMock()
    mock_chain.stream.return_value = ["Answer"]
    mock_engine = _make_mock_engine(
        get_doc_count=MagicMock(return_value=5),
        build_chain=MagicMock(return_value=(mock_chain, MagicMock())),
        retrieve_for_query=MagicMock(
            return_value=(
                [
                    {
                        "file": "doc.pdf",
                        "page": 1,
                        "content": "ctx",
                        "source_type": "preloaded",
                    }
                ],
                [],
            )
        ),
    )
    sys.modules["engine"] = mock_engine
    at = AppTest.from_file("app.py", default_timeout=10)
    at.run()
    at.chat_input[0].set_value("test").run()
    msgs = at.session_state["messages"]
    assert len(msgs[-1].get("context", [])) >= 1


def test_app_chat_with_sources_disabled(app):
    mock_chain = MagicMock()
    mock_chain.stream.return_value = ["Answer"]
    mock_engine = _make_mock_engine(
        get_doc_count=MagicMock(return_value=5),
        build_chain=MagicMock(return_value=(mock_chain, MagicMock())),
        retrieve_for_query=MagicMock(
            return_value=(
                [],
                [
                    {
                        "file": "doc.pdf",
                        "page": 1,
                        "snippet": "snip",
                        "source_type": "preloaded",
                    }
                ],
            )
        ),
    )
    sys.modules["engine"] = mock_engine
    at = AppTest.from_file("app.py", default_timeout=10)
    at.run()
    at.session_state["show_sources"] = False
    at.chat_input[0].set_value("test").run()
    msgs = at.session_state["messages"]
    assert len(msgs) == 2


def test_app_chat_with_context_disabled(app):
    mock_chain = MagicMock()
    mock_chain.stream.return_value = ["Answer"]
    mock_engine = _make_mock_engine(
        get_doc_count=MagicMock(return_value=5),
        build_chain=MagicMock(return_value=(mock_chain, MagicMock())),
        retrieve_for_query=MagicMock(
            return_value=(
                [
                    {
                        "file": "doc.pdf",
                        "page": 1,
                        "content": "ctx data",
                        "source_type": "preloaded",
                    }
                ],
                [],
            )
        ),
    )
    sys.modules["engine"] = mock_engine
    at = AppTest.from_file("app.py", default_timeout=10)
    at.run()
    at.session_state["show_context"] = False
    at.chat_input[0].set_value("test").run()
    msgs = at.session_state["messages"]
    assert len(msgs) == 2


def test_app_chat_context_display_on_rerun(app):
    mock_chain = MagicMock()
    mock_chain.stream.return_value = ["Answer"]
    mock_engine = _make_mock_engine(
        get_doc_count=MagicMock(return_value=5),
        build_chain=MagicMock(return_value=(mock_chain, MagicMock())),
        retrieve_for_query=MagicMock(
            return_value=(
                [
                    {
                        "file": "doc.pdf",
                        "page": 1,
                        "content": "display me",
                        "source_type": "preloaded",
                    }
                ],
                [],
            )
        ),
    )
    sys.modules["engine"] = mock_engine
    at = AppTest.from_file("app.py", default_timeout=10)
    at.run()
    at.chat_input[0].set_value("test").run()
    at.run()
    assert len(at.session_state["messages"]) >= 1
    assert at.session_state["messages"][-1].get("context") is not None
