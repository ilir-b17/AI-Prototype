# analyze_table_file

Load a CSV/TSV/Excel file and return a concise summary of its contents using pandas.

Use this for quick inspection of data files before deeper analysis.

## Schema

```json
{
  "name": "analyze_table_file",
  "description": "Load a CSV/TSV/Excel file and return a concise summary of its contents using pandas.",
  "parameters": {
    "type": "object",
    "properties": {
      "file_path": {
        "type": "string",
        "description": "Path to the data file. Relative paths are resolved under the downloads directory."
      },
      "sheet_name": {
        "type": "string",
        "description": "Excel sheet name to load (optional)."
      },
      "max_rows": {
        "type": "integer",
        "description": "Maximum number of rows to display in the preview (default 20)."
      },
      "max_chars": {
        "type": "integer",
        "description": "Maximum number of characters to return (default 12000)."
      }
    },
    "required": ["file_path"]
  }
}
```
