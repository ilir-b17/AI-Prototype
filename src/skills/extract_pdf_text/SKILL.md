# extract_pdf_text

Extract text from a PDF file and return a plain-text string for summarization.

If a relative file name is provided, the tool will look inside the downloads folder.

## Schema

```json
{
  "name": "extract_pdf_text",
  "description": "Extract plain text from a PDF file for summarization. If the path is relative, the downloads folder is used.",
  "parameters": {
    "type": "object",
    "properties": {
      "file_path": {
        "type": "string",
        "description": "Path to the PDF file. Relative paths are resolved under the downloads directory."
      },
      "max_pages": {
        "type": "integer",
        "description": "Maximum number of pages to read from the PDF (default 20)."
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
