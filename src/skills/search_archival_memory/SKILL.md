---
name: search_archival_memory
description: >
  Queries the ChromaDB vector storage for semantically similar past
  conversations or experiences using two-stage retrieval: broad cosine
  recall followed by LLM reranking for better relevance. Supports
  optional session_id and epic_id scoping. Use session_id from active
  session context when available for project-scoped search.
---

# search_archival_memory

Two-stage memory retrieval:
  Stage 1: cosine similarity recall (n_results * MEMORY_RECALL_MULTIPLIER)
  Stage 2: System 1 batch relevance scoring and reranking

Falls back to cosine-only when reranker is disabled or unavailable.

```json
{
  "name": "search_archival_memory",
  "description": "Queries ChromaDB vector storage using two-stage retrieval: broad cosine recall + LLM reranking. Returns semantically relevant memories with relevance scores. Supports session and epic scoping.",
  "parameters": {
    "type": "object",
    "properties": {
      "query": {
        "type": "string",
        "description": "The search phrase or concept to find in memory."
      },
      "n_results": {
        "type": "integer",
        "description": "Number of results to return after reranking. Default 3, max 10.",
        "default": 3
      },
      "session_id": {
        "type": "integer",
        "description": "Optional session id to scope search to current project memories."
      },
      "epic_id": {
        "type": "integer",
        "description": "Optional epic id to scope search to a project epic's memories."
      },
      "skip_reranking": {
        "type": "boolean",
        "description": "Set true to use cosine-only retrieval (faster, less accurate). Default false.",
        "default": false
      }
    },
    "required": ["query"]
  }
}
```
