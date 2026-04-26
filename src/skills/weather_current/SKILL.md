---
name: weather_current
description: Returns current weather conditions for a location query.
---

# weather_current

Fetches current weather for a location string.

```json
{
  "name": "weather_current",
  "description": "Get current weather for a city or location query (temperature, conditions).",
  "parameters": {
    "type": "object",
    "properties": {
      "query": {
        "type": "string",
        "description": "Location query, e.g. 'Vienna' or 'Tokyo, JP'."
      }
    },
    "required": ["query"]
  }
}
```
