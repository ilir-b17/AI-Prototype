# get_system_info

Returns the current date, time, timezone, and host platform hardware details (CPU, RAM, Disk).
Use this whenever the user asks about the current time, date, or the machine's capabilities, environment, or current load.

## Schema

```json
{
  "name": "get_system_info",
  "description": "Returns a JSON string containing the current date, time, timezone, OS, and host hardware capabilities (CPU cores and usage, RAM total/available, Disk total/free). Use whenever the user asks about the current time, date, the machine environment, or the system's hardware resources and load.",
  "parameters": {
    "type": "object",
    "properties": {},
    "required": []
  }
}
```
