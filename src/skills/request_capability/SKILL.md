# request_capability

Call this when you cannot fulfill the user's request because a required tool
or capability does not exist in your current toolkit.
**Triggers System 2 tool synthesis** — the orchestrator intercepts this call,
asks System 2 (Groq/Gemini) to draft a new Python tool, then presents it to
the Admin for approval. Upon approval the tool is deployed into the runtime
and persisted to the Tool Registry.

## Schema

```json
{
  "name": "request_capability",
  "description": "Call this tool when you cannot fulfill the user's request because a required tool or capability does not exist. Triggers System 2 to synthesise a new tool for Admin approval.",
  "parameters": {
    "type": "object",
    "properties": {
      "gap_description": {
        "type": "string",
        "description": "Exact description of the missing capability (e.g. 'read a local file by path')."
      },
      "suggested_tool_name": {
        "type": "string",
        "description": "snake_case name for the tool that would fill this gap (e.g. 'read_file')."
      }
    },
    "required": ["gap_description", "suggested_tool_name"]
  }
}
```
