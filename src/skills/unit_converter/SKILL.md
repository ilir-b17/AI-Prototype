---
name: unit_converter
description: Converts common units from a natural-language query.
---

# unit_converter

Converts common units from a query string.

```json
{
  "name": "unit_converter",
  "description": "Convert values across common units such as km/miles, kg/pounds, celsius/fahrenheit, and data sizes.",
  "parameters": {
    "type": "object",
    "properties": {
      "query": {
        "type": "string",
        "description": "Conversion request, e.g. '100 km to miles'."
      }
    },
    "required": ["query"]
  }
}
```
