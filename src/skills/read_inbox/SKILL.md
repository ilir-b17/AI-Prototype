---
name: read_inbox
description: Reads unread emails from an IMAP inbox, filters by trusted senders, and downloads attachments into confined storage.
---

## Schema

```json
{
  "name": "read_inbox",
  "description": "Reads unread emails from an IMAP folder, filters to trusted senders from AIDEN_TRUSTED_SENDERS, extracts sender/subject/body/message_id/date, and downloads attachments to AIDEN_DOWNLOADS_DIR under a per-message subdirectory.",
  "parameters": {
    "type": "object",
    "properties": {
      "max_emails": {
        "type": "integer",
        "description": "Maximum number of unread emails to inspect.",
        "default": 5
      },
      "folder": {
        "type": "string",
        "description": "IMAP folder/mailbox to read.",
        "default": "INBOX"
      },
      "mark_as_read": {
        "type": "boolean",
        "description": "Whether qualifying trusted emails should be marked as read after processing.",
        "default": true
      }
    }
  }
}
```
