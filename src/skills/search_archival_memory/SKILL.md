# search_archival_memory

Searches the Archival Memory (ChromaDB vector store) for semantically relevant
historical context. Use this when the user asks about past conversations,
prior decisions, or anything that might have been stored in long-term memory.

## Schema

```json
{
  "name": "search_archival_memory",
  "description": "Searches Archival Memory (ChromaDB) for relevant historical context.",
  "parameters": {
    "type": "object",
    "properties": {
      "query": {
        "type": "string",
        "description": "Search query."
      }
    },
    "required": ["query"]
  }
}
```
