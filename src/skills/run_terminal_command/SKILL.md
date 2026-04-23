# run_terminal_command

Executes a command in the underlying shell (bash/cmd).
Outputs standard STDOUT and STDERR or returns detailed JSON error messages upon failure.

## Schema

```json
{
  "name": "run_terminal_command",
  "description": "Executes a string command in the underlying system terminal. Useful for running shell utilities, git commands, checking processes, etc. Destructive commands (rm, mv, shutdown) will be blocked. Includes a configurable execution timeout to prevent hanging. Returns standard output/error, or JSON error details if execution fails.",
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
