# spawn_new_objective

Adds a new Epic, Story, or Task to the Objective Backlog (3-tier hierarchy).
Use this when you identify a sub-task or new goal that should be tracked and
potentially executed autonomously by the heartbeat loop.

## Schema

```json
{
  "name": "spawn_new_objective",
  "description": "Adds a new goal, objective, Epic, Story, or Task to the Objective Backlog. Use when the Admin asks to add, create, define, or track a goal/objective, or when the system identifies new work that should be tracked.",
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
      },
      "agent_domain": {
        "type": "string",
        "description": "Optional execution domain. Use 'aiden' (or omit) for heartbeat/AIDEN execution, or a domain like 'google' for Blackboard domain agents."
      }
    },
    "required": ["tier", "title", "estimated_energy"]
  }
}
```
