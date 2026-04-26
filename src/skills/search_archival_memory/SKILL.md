---
name: search_archival_memory
description: Searches the long-term vector database (Hippocampus) for semantic context, with optional session and epic scoping.
---

# search_archival_memory

Searches the long-term vector database (Hippocampus) for semantic context.

## Schema

```json
{
  "name": "search_archival_memory",
  "description": "Queries the ChromaDB vector storage for semantically similar past conversations or experiences. Supports optional session_id and epic_id scoping to retrieve project-relevant memories first. Use session_id from the active session context when available.",
  "parameters": {
    "type": "object",
    "properties": {
      "query": {
        "type": "string",
        "description": "The search phrase or concept to find in memory."
      },
      "n_results": {
        "type": "integer",
        "description": "The number of results to return. Default 3, max 10.",
        "default": 3
      },
      "session_id": {
        "type": "integer",
        "description": "Optional session id to scope search to current project memories. Pass the id from active_session context when available."
      },
      "epic_id": {
        "type": "integer",
        "description": "Optional epic id to scope search to a project epic's memories."
      }
    },
    "required": ["query"]
  }
}
```
