# escalate_to_system_2

Escalate a complex reasoning task or a stuck state to System 2.

## Schema

```json
{
  "name": "escalate_to_system_2",
  "description": "Escalates a complex task or problem to System 2 when System 1 is stuck or requires deep reasoning. Use this when facing a complex multi-step logic problem, advanced formatting, or if previous attempts have failed.",
  "parameters": {
    "type": "object",
    "properties": {
      "problem_description": {
        "type": "string",
        "description": "A clear description of the complex problem to be solved."
      },
      "context_scratchpad": {
        "type": "string",
        "description": "Any relevant context, variables, or partial work done so far."
      }
    },
    "required": ["problem_description", "context_scratchpad"]
  }
}
```
