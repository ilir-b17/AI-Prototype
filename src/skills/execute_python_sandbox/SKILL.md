# execute_python_sandbox

Executes isolated Python code locally via a temporary subprocess.

## Schema

```json
{
  "name": "execute_python_sandbox",
  "description": "Executes Python code by dispatching to the dynamic tool worker builtin (__exec_user_python__). Execution inherits worker sandbox controls: AST validation, blocked-module/import checks, POSIX resource limits where supported, and worker-confined /tmp file access. Returns script output on success or structured JSON errors when blocked.",
  "parameters": {
    "type": "object",
    "properties": {
      "code_string": {
        "type": "string",
        "description": "The raw string representation of the Python script to execute. Ensure it contains a print statement if you expect output."
      }
    },
    "required": ["code_string"]
  }
}
```
