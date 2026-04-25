# run_terminal_command

Executes a command in the underlying shell (bash/cmd).
Outputs standard STDOUT and STDERR or returns detailed JSON error messages upon failure.

## Schema

```json
{
  "name": "run_terminal_command",
  "description": "Executes terminal commands without a shell (create_subprocess_exec). Rejects shell metacharacters outside quotes and enforces a first-token allowlist from AIDEN_TERMINAL_ALLOWLIST (default: git status, ls, dir, wc, echo, cat, type, pwd, hostname). Returns STDOUT/STDERR on success, or structured JSON errors when blocked or invalid.",
  "parameters": {
    "type": "object",
    "properties": {
      "command": {
        "type": "string",
        "description": "The shell command string to execute."
      }
    },
    "required": ["command"]
  }
}
```
