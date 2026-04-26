---
name: query_system_health
description: >
  Returns a system health report covering recent errors and warnings
  from the system log, and synthesis run history. Use when the admin
  asks about errors, failed tool synthesis attempts, why a capability
  was not created, or whether any system components are failing.
---

# query_system_health

Reads from system_logs (WARNING/ERROR/CRITICAL entries) and
synthesis_runs to produce a health summary.

```json
{
  "name": "query_system_health",
  "description": "Returns a system health report with recent errors, warnings, and tool synthesis history. Explains why capabilities may have failed to synthesise.",
  "parameters": {
    "type": "object",
    "properties": {
      "hours": {
        "type": "integer",
        "description": "Lookback window in hours for error logs. Default 24, max 168 (7 days).",
        "default": 24
      },
      "include_synthesis_history": {
        "type": "boolean",
        "description": "Include tool synthesis run history. Default true.",
        "default": true
      },
      "include_error_log": {
        "type": "boolean",
        "description": "Include system error/warning log. Default true.",
        "default": true
      },
      "error_limit": {
        "type": "integer",
        "description": "Maximum error log entries to return. Default 20, max 50.",
        "default": 20
      }
    },
    "required": []
  }
}
```