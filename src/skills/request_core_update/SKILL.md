# request_core_update

Request authorization to modify the system charter or delete core memory.
**Triggers MFA verification** — the orchestrator intercepts this call and
challenges the Admin with a passphrase before the function body executes.
Only write/delete/modify operations require MFA; reads are always permitted.

## Schema

```json
{
  "name": "request_core_update",
  "description": "Request authorization to modify the system charter or delete core memory. Triggers MFA.",
  "parameters": {
    "type": "object",
    "properties": {
      "component": {
        "type": "string",
        "description": "System component to update, e.g. 'charter'."
      },
      "proposed_change": {
        "type": "string",
        "description": "Details of the requested update."
      }
    },
    "required": ["component", "proposed_change"]
  }
}
```
