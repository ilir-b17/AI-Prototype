# update_core_memory

Updates a key in Core Working Memory (the agent's short-term RAM, stored in
`data/core_memory.json`). Use this to record the current focus, user preferences,
or any short-term context that should persist across the session.

## Schema

```json
{
  "name": "update_core_memory",
  "description": "Updates a key in Core Working Memory (short-term RAM) like 'current_focus'.",
  "parameters": {
    "type": "object",
    "properties": {
      "key": {
        "type": "string",
        "description": "The key to update."
      },
      "value": {
        "type": "string",
        "description": "The new value."
      }
    },
    "required": ["key", "value"]
  }
}
```
