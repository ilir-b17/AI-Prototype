# spawn_new_objective

Adds a new Epic, Story, or Task to the Objective Backlog (3-tier hierarchy).
Use this when you identify a sub-task or new goal that should be tracked and
potentially executed autonomously by the heartbeat loop.

## Schema

```json
{
  "name": "spawn_new_objective",
  "description": "Adds a new Epic, Story, or Task to the Objective Backlog. Use when you identify a sub-task needed to complete a broader goal.",
  "parameters": {
    "type": "object",
    "properties": {
      "tier": {
        "type": "string",
        "description": "Hierarchy level: Epic, Story, or Task."
      },
      "title": {
        "type": "string",
        "description": "Clear, actionable title for the objective."
      },
      "estimated_energy": {
        "type": "integer",
        "description": "Estimated cognitive energy cost (5-50)."
      }
    },
    "required": ["tier", "title", "estimated_energy"]
  }
}
```
