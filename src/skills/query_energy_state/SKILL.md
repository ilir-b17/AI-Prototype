---
name: query_energy_state
description: >
  Returns AIDEN's current cognitive energy budget and a breakdown
  of recently deferred tasks with their deferral reasons. Use when
  the admin asks about the energy budget, why tasks were skipped by
  the heartbeat, or whether there is enough energy to run a task.
---

# query_energy_state

Reads the live predictive energy budget from the orchestrator runtime
and queries recently deferred tasks from the objective backlog.

```json
{
  "name": "query_energy_state",
  "description": "Returns AIDEN's current cognitive energy budget and deferred task analysis. Shows why tasks were skipped by the heartbeat loop.",
  "parameters": {
    "type": "object",
    "properties": {
      "include_deferred_tasks": {
        "type": "boolean",
        "description": "Include list of currently deferred tasks. Default true.",
        "default": true
      },
      "include_blocked_tasks": {
        "type": "boolean",
        "description": "Include list of currently blocked tasks. Default true.",
        "default": true
      }
    },
    "required": []
  }
}
```