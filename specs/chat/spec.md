# Feature Specification: RAG Chat Query Processing

## Overview
Users can ask questions against indexed research documents and receive grounded answers with source metadata.

## User Story
As a research assistant user, I want to submit natural-language questions, so that I can get answers based on uploaded or preloaded documents.

## Acceptance Criteria
- Given an indexed document collection, when a user submits a query, then the app retrieves relevant chunks and returns an answer based on those chunks.
- Given no indexed documents are available, when the user attempts to build a chat chain, then the system reports that no chain or retriever is available.
- Given retrieved chunks include source metadata, when sources are displayed, then duplicate file/page pairs are collapsed.

## Requirements
- Answers must be generated only after a vector store is available.
- Retrieval must support filtering by preloaded or uploaded source type.
- Source and context helpers must return display-ready file, page, snippet, content, and source type data.

## Out of Scope
- Multi-user document permissions.
- Remote model hosting configuration.
- Evaluation of answer factuality beyond retrieved-source grounding.

## Open Questions
- What minimum coverage threshold should replace the current initial 20 percent gate as test depth improves?
