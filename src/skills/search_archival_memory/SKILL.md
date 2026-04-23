# search_archival_memory

Searches the long-term vector database (Hippocampus) for semantic context.

## Schema

```json
{
  "name": "search_archival_memory",
  "description": "Queries the ChromaDB vector storage for semantically similar past conversations or experiences. Use this to recall historical context, past preferences, or earlier data. Returns a JSON list of matching documents with their metadata.",
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
      }
    },
    "required": ["query"]
  }
}
```
