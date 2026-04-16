# update_ledger

Writes a new task entry to the system ledger (SQLite task queue).
Use this to log work items, scheduled tasks, or anything that needs to be tracked
in the operational task backlog.

## Schema

```json
{
  "name": "update_ledger",
  "description": "Writes a new task to the system ledger/task queue.",
  "parameters": {
    "type": "object",
    "properties": {
      "task_description": {
        "type": "string",
        "description": "The detailed description of the task to be logged."
      },
      "priority": {
        "type": "integer",
        "description": "Priority 1 (highest) to 10 (lowest). Default 5."
      }
    },
    "required": ["task_description"]
  }
}
```
