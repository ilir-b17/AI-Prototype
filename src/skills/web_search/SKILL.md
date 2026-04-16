# web_search

Search the web using DuckDuckGo. Returns up to `max_results` results, each with
a title, URL, and a short snippet. Zero-cost — no API key required.
Use this whenever the user asks for current information, news, prices, or anything
that requires looking up external data.

## Schema

```json
{
  "name": "web_search",
  "description": "Search the web using DuckDuckGo. Returns top results with title, URL, and snippet. Use for current information, news, or anything requiring external data.",
  "parameters": {
    "type": "object",
    "properties": {
      "query": {
        "type": "string",
        "description": "The search query string."
      },
      "max_results": {
        "type": "integer",
        "description": "Maximum number of results to return (default 3, max 10)."
      }
    },
    "required": ["query"]
  }
}
```
