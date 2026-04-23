# get_stock_price

Retrieves current market data for a given stock ticker symbol.

## Schema

```json
{
  "name": "get_stock_price",
  "description": "Fetches the latest market price, daily change, and previous close for a given stock ticker symbol (e.g., 'AAPL', 'MSFT') using Yahoo Finance. Returns structured JSON containing the financial data or detailed error messages if the ticker is invalid.",
  "parameters": {
    "type": "object",
    "properties": {
      "ticker": {
        "type": "string",
        "description": "The stock ticker symbol to look up (e.g., 'TSLA')."
      }
    },
    "required": ["ticker"]
  }
}
```
