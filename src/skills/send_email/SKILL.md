---
name: send_email
description: Sends an email with an optional file attachment via SMTP.
---

```json
{
  "name": "send_email",
  "description": "Sends an email to a specified recipient. Can optionally attach a file if the absolute file path is provided.",
  "parameters": {
    "type": "object",
    "properties": {
      "recipient": {
        "type": "string",
        "description": "The email address to send to."
      },
      "subject": {
        "type": "string",
        "description": "The subject line of the email."
      },
      "body": {
        "type": "string",
        "description": "The plain text body of the email."
      },
      "attachment_path": {
        "type": "string",
        "description": "Optional. The absolute file path of a document to attach (e.g., returned from generate_docx_report)."
      }
    },
    "required": ["recipient", "subject", "body"]
  }
}
```