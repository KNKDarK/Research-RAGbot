# Tasks: RAG Chat Query Processing

## Implementation
- [x] Load PDF, text, and Markdown documents.
- [x] Split documents into chunks with source metadata.
- [x] Store and retrieve chunks from ChromaDB.
- [x] Build a LangChain RAG chain for query answering.
- [x] Return source and context metadata for UI display.

## Tests
- [x] Cover empty document count behavior.
- [x] Cover unsupported file loading.
- [x] Cover retrieved document formatting.
- [ ] Add tests for source de-duplication.
- [ ] Add tests for source type filtering.

## Documentation
- [x] Add feature specification.
- [x] Add implementation plan.
- [x] Add task tracking.

## Verification
- [x] `ruff check .`
- [x] `ruff format --check .`
- [x] `mypy . --ignore-missing-imports`
- [x] `pytest --cov=. --cov-report=term-missing --cov-fail-under=20`
