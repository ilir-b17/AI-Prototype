# query_highest_priority_task

Returns the single highest-priority pending Task from the Objective Backlog
(SQLite `objective_backlog` table). Use this at the start of a heartbeat cycle
to find the most urgent autonomous work item.

## Schema

```json
{
  "name": "query_highest_priority_task",
  "description": "Returns the single highest-priority pending Task from the Objective Backlog. Use this at the start of a heartbeat cycle to find work.",
  "parameters": {
    "type": "object",
    "properties": {},
    "required": []
  }
}
```
