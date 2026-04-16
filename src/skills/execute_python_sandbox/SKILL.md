# execute_python_sandbox

Execute arbitrary Python code in an isolated sandbox with a strict 10-second timeout.

Use this as your internal calculator and data processor. Use it to perform complex math, parse data, or test newly drafted Python logic safely before requesting it be made into a permanent skill.

## Schema

```json
{
  "name": "execute_python_sandbox",
  "description": "Execute arbitrary Python code in an isolated sandbox with a strict 10-second timeout. Use this as your internal calculator and data processor. Use it to perform complex math, parse data, or test newly drafted Python logic safely before requesting it be made into a permanent skill.",
  "parameters": {
    "type": "object",
    "properties": {
      "code_string": {
        "type": "string",
        "description": "The Python code to execute."
      }
    },
    "required": ["code_string"]
  }
}
```
