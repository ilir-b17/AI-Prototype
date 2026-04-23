# extract_pdf_text

Reads and extracts text from a PDF file.

## Schema

```json
{
  "name": "extract_pdf_text",
  "description": "Reads text from a local PDF file. Resolves relative paths to the AIDEN_DOWNLOADS_DIR. Useful for analyzing reports or documents. Returns JSON with the extracted text (up to max_chars and max_pages limits) or a specific error message if the PDF is scanned or image-only.",
  "parameters": {
    "type": "object",
    "properties": {
      "file_path": {
        "type": "string",
        "description": "The path to the PDF file. If relative, assumes the downloads directory."
      },
      "max_pages": {
        "type": "integer",
        "description": "The maximum number of pages to read. Default 20.",
        "default": 20
      },
      "max_chars": {
        "type": "integer",
        "description": "The maximum number of characters to return. Default 12000.",
        "default": 12000
      }
    },
    "required": ["file_path"]
  }
}
```
