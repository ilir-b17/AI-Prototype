# manage_file_system

Manage the file system through 'read', 'list', 'write', and 'delete' operations.

Use this to read system logs, inspect project data, or review your own python code. If you discover a bug in your logic, read the file here first, then use `request_core_update` to propose the fix.

## Schema

```json
{
  "name": "manage_file_system",
  "description": "Manage the file system through read, list, write, and delete operations. Use this to read system logs, inspect project data, or review your own python code. If you discover a bug in your logic, read the file here first, then use `request_core_update` to propose the fix.",
  "parameters": {
    "type": "object",
    "properties": {
      "action": {
        "type": "string",
        "description": "The file system action to perform: 'read', 'write', 'list', or 'delete'."
      },
      "file_path": {
        "type": "string",
        "description": "The path to the file or directory."
      },
      "content": {
        "type": "string",
        "description": "The content to write to the file. Only used when action is 'write'."
      }
    },
    "required": ["action", "file_path"]
  }
}
```
