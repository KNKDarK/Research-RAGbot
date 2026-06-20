# Implementation Plan: RAG Chat Query Processing

## Summary
The feature is implemented in `engine.py` with Chroma-backed retrieval, Ollama embeddings, an Ollama chat model, and Streamlit presentation in `app.py`.

## Technical Context
- Runtime: Python 3.11 in CI.
- Dependencies: LangChain integrations, ChromaDB, Streamlit, Ollama, Pytest, Ruff, Mypy, Bandit.
- Data: Local `data/`, `uploads/`, and persistent `chroma_db/` directories.
- Interfaces: Streamlit UI and engine helper functions.

## Steps
1. Keep ingestion isolated by source type and invalidate cached vector stores after writes.
2. Build retrieval chains only when Chroma contains indexed chunks.
3. Expose source and context helpers for UI citations.
4. Verify formatting, linting, typing, tests, coverage, and security checks in CI.

## Validation
- `ruff check .`
- `ruff format --check .`
- `mypy . --ignore-missing-imports`
- `pytest --cov=. --cov-report=term-missing --cov-fail-under=20`
