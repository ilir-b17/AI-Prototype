# web_search

Performs a web search utilizing DuckDuckGo to return relevant internet results.

## Schema

```json
{
  "name": "web_search",
  "description": "Performs an internet search to retrieve the most up-to-date information including current news, weather forecasts, local business listings, events, prices, and general web context. Returns a structured JSON list of results containing the title, URL, and a brief text snippet. Use this whenever you need real-time or location-specific data that is not available in training knowledge.",
  "parameters": {
    "type": "object",
    "properties": {
      "query": {
        "type": "string",
        "description": "The search keywords or query string."
      },
      "max_results": {
        "type": "integer",
        "description": "The maximum number of results to return. Default is 3, maximum is 10.",
        "default": 3
      }
    },
    "required": ["query"]
  }
}
```
