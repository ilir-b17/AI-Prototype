---
name: generate_docx_report
description: Generates a professionally formatted Microsoft Word (.docx) document.
---

```json
{
  "name": "generate_docx_report",
  "description": "Generates a professionally formatted Microsoft Word (.docx) document. Use this when the user requests a formal report, memo, or document. It saves the file locally and returns the absolute file path.",
  "parameters": {
    "type": "object",
    "properties": {
      "title": {
        "type": "string",
        "description": "The title of the report. This will be the main header."
      },
      "content": {
        "type": "string",
        "description": "The main body of the report. Use clear paragraphs."
      },
      "filename": {
        "type": "string",
        "description": "The desired filename (e.g., 'Organ_Trafficking_Report.docx'). Must end in .docx."
      }
    },
    "required": ["title", "content", "filename"]
  }
}
```