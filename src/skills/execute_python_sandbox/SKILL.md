# execute_python_sandbox

Executes isolated Python code locally via a temporary subprocess.

## Schema

```json
{
  "name": "execute_python_sandbox",
  "description": "Executes Python code locally in a secure subprocess. Excellent for data processing, complex calculations, analyzing local repository code, or testing API calls. Will timeout if it hangs. The python environment has access to the installed packages in the main bot environment.",
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
