# extract_pdf_text

Reads and extracts text from a PDF file.

## Schema

```json
{
  "name": "extract_pdf_text",
  "description": "Reads text from a local PDF file, resolves and confines paths to the configured allowed roots, and resolves relative paths to AIDEN_DOWNLOADS_DIR. By default, legacy limits apply (max_pages/max_chars). Set full_context=true to request unchunked extraction; in that mode max_chars can be provided as a dynamic safety cap.",
  "parameters": {
    "type": "object",
    "properties": {
      "file_path": {
        "type": "string",
        "description": "The path to the PDF file. If relative, assumes the downloads directory."
      },
      "full_context": {
        "type": "boolean",
        "description": "If true, request full-document extraction mode. Legacy hard caps are bypassed unless max_pages/max_chars are explicitly provided.",
        "default": false
      },
      "max_pages": {
        "type": "integer",
        "description": "The maximum number of pages to read. In legacy mode default is 20. In full_context mode this is optional and 0/omitted means all pages.",
        "default": 20
      },
      "max_chars": {
        "type": "integer",
        "description": "The maximum number of characters to return. In legacy mode default is 12000. In full_context mode this can be passed as a dynamic context safety cap.",
        "default": 12000
      }
    },
    "required": ["file_path"]
  }
}
```
