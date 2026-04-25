# manage_file_system

Provides basic file system operations: `read` and `list`.
The `write` and `delete` operations will immediately return a SECURITY LOCK requiring the use of `request_core_update` to get admin approval.

## Schema

```json
{
  "name": "manage_file_system",
  "description": "Manages file system operations and resolves and confines paths to the configured allowed roots. Can 'read' files or 'list' directories. 'write' and 'delete' operations trigger a SECURITY LOCK. Returns file content, directory listing JSON, or detailed JSON error messages if the path is invalid.",
  "parameters": {
    "type": "object",
    "properties": {
      "action": {
        "type": "string",
        "description": "The action to perform: 'read', 'list', 'write', or 'delete'.",
        "enum": ["read", "list", "write", "delete"]
      },
      "file_path": {
        "type": "string",
        "description": "The absolute or relative path to the file or directory."
      },
      "content": {
        "type": "string",
        "description": "The content to write (only used if action is 'write')."
      }
    },
    "required": ["action", "file_path"]
  }
}
```
