---
name: query_decision_log
description: >
  Queries AIDEN's supervisor decision log and moral audit history to
  explain past decisions. Use when the admin asks why a specific agent
  was chosen, what happened in the last turn, why a response was
  rejected, or wants to review recent decision history.
---

# query_decision_log

Reads from supervisor_decisions and moral_audit_log tables to provide
introspective explanations of AIDEN's recent reasoning.

```json
{
  "name": "query_decision_log",
  "description": "Queries AIDEN's supervisor decision log and moral audit history. Explains why agents were chosen, what plans were executed, and whether outputs passed moral review.",
  "parameters": {
    "type": "object",
    "properties": {
      "scope": {
        "type": "string",
        "description": "What to query: 'last_turn' (most recent decision), 'recent' (last N decisions), 'rejections' (moral audit failures only), 'all' (decisions + moral audit combined).",
        "enum": ["last_turn", "recent", "rejections", "all"],
        "default": "last_turn"
      },
      "limit": {
        "type": "integer",
        "description": "Number of records to return. Default 5, max 20.",
        "default": 5
      },
      "user_id": {
        "type": "string",
        "description": "Filter to a specific user_id. Omit for all users."
      }
    },
    "required": []
  }
}
```