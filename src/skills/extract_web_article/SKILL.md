# extract_web_article

Downloads and extracts the main text content from a web URL.

## Schema

```json
{
  "name": "extract_web_article",
  "description": "Fetches a URL and extracts the main article text, stripping out ads, navigation, and boilerplate. Legacy mode caps output by max_chars. Set full_context=true to request unchunked extraction; in that mode max_chars can be supplied as a dynamic context safety cap.",
  "parameters": {
    "type": "object",
    "properties": {
      "url": {
        "type": "string",
        "description": "The full HTTP/HTTPS URL of the article to extract."
      },
      "full_context": {
        "type": "boolean",
        "description": "If true, request full-context extraction mode. Legacy hard caps are bypassed unless max_chars is explicitly provided.",
        "default": false
      },
      "max_chars": {
        "type": "integer",
        "description": "The maximum number of characters to return. In legacy mode default is 12000. In full_context mode this can be passed as a dynamic context safety cap.",
        "default": 12000
      }
    },
    "required": ["url"]
  }
}
```
