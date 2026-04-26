---
name: query_objective_status
description: >
  Queries the objective backlog to show task status, progress, and
  energy deferral details. Use when the admin asks what tasks are
  pending, active, deferred, or blocked, or wants a project progress
  summary. Supports filtering by status and tier.
---

# query_objective_status

Reads from objective_backlog with flexible filters. For deferred
tasks, includes energy evaluation context explaining why the task
was postponed.

```json
{
  "name": "query_objective_status",
  "description": "Queries the objective backlog with status and tier filters. Returns task lists with energy deferral context where applicable.",
  "parameters": {
    "type": "object",
    "properties": {
      "status_filter": {
        "type": "string",
        "description": "Filter by status: 'pending', 'active', 'deferred', 'blocked', 'completed', 'all'. Default 'all'.",
        "default": "all"
      },
      "tier": {
        "type": "string",
        "description": "Filter by tier: 'Epic', 'Story', 'Task'. Omit for all tiers."
      },
      "include_energy_context": {
        "type": "boolean",
        "description": "For deferred tasks, include the energy evaluation that caused deferral. Default true.",
        "default": true
      },
      "limit": {
        "type": "integer",
        "description": "Maximum tasks to return. Default 20, max 50.",
        "default": 20
      }
    },
    "required": []
  }
}
```