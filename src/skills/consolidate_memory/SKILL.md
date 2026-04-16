# consolidate_memory

Performs long-term cognitive compression on recent chat history.
Retrieves the last 50 chat turns (excluding heartbeat tasks), uses
System 2 (Groq `llama-3.1-8b-instant`) to extract persistent user
preferences, facts, and patterns, then writes the distilled insights
to `core_memory.json` under the `consolidated_insights` key.

Intended for use by the heartbeat loop on the scheduled
"Run memory consolidation on recent chat history" Task.

## Schema

```json
{
  "name": "consolidate_memory",
  "description": "Performs long-term cognitive compression on recent chat history, extracting persistent user preferences and facts into core memory.",
  "parameters": {
    "type": "object",
    "properties": {},
    "required": []
  }
}
```
