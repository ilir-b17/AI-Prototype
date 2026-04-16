# update_objective_status

Updates the status of an objective in the Backlog.
Valid statuses: `pending`, `active`, `completed`, `suspended`.
Use to mark tasks as active when starting work, completed when done,
or suspended when blocked.

## Schema

```json
{
  "name": "update_objective_status",
  "description": "Updates the status of an objective in the backlog. Use to mark tasks active, completed, or suspended.",
  "parameters": {
    "type": "object",
    "properties": {
      "task_id": {
        "type": "integer",
        "description": "The numeric ID of the objective."
      },
      "new_status": {
        "type": "string",
        "description": "New status: pending, active, completed, or suspended."
      }
    },
    "required": ["task_id", "new_status"]
  }
}
```
