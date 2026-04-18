# extract_web_article

Fetch a URL and extract the main article text (boilerplate removed) using trafilatura.

Use this after `web_search` to read the content of a page before summarizing.

## Schema

```json
{
  "name": "extract_web_article",
  "description": "Fetch a URL and extract the main article text with trafilatura (boilerplate removed). Use after web_search to read page content.",
  "parameters": {
    "type": "object",
    "properties": {
      "url": {
        "type": "string",
        "description": "The URL to fetch and extract text from."
      },
      "max_chars": {
        "type": "integer",
        "description": "Maximum number of characters to return (default 12000)."
      }
    },
    "required": ["url"]
  }
}
```
