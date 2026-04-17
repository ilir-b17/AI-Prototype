# run_terminal_command

Execute shell commands on the system. Security controls block inherently destructive commands, and a strict 10-second timeout applies.

Use this to inspect system resources, run `git status`, or run `pip install` when a newly proposed skill requires new dependencies.

## Schema

```json
{
  "name": "run_terminal_command",
  "description": "Execute shell commands safely. Use this to inspect system resources, run `git status`, or run `pip install` when a newly proposed skill requires new dependencies.",
  "parameters": {
    "type": "object",
    "properties": {
      "command": {
        "type": "string",
        "description": "The shell command to execute."
      }
    },
    "required": ["command"]
  }
}
```
