# analyze_table_file

Loads a local CSV, TSV, or Excel file and returns a concise summary.

## Schema

```json
{
  "name": "analyze_table_file",
  "description": "Analyzes a local tabular data file (CSV, TSV, XLSX), resolves and confines paths to the configured allowed roots, and resolves relative paths to AIDEN_DOWNLOADS_DIR. Returns a JSON summary including shape, column types, missing values, and a brief data preview.",
  "parameters": {
    "type": "object",
    "properties": {
      "file_path": {
        "type": "string",
        "description": "The path to the file. If relative, it assumes the downloads directory."
      },
      "sheet_name": {
        "type": "string",
        "description": "The specific sheet name to read for Excel files (optional)."
      },
      "max_rows": {
        "type": "integer",
        "description": "Number of preview rows to include. Default 20.",
        "default": 20
      },
      "max_chars": {
        "type": "integer",
        "description": "Maximum characters in the output string to prevent context bloat. Default 12000.",
        "default": 12000
      }
    },
    "required": ["file_path"]
  }
}
```
