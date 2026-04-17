# get_stock_price

Fetches the latest market price for a stock ticker using `yfinance` (Yahoo Finance).
Returns the current price, previous close, and percentage change.

Use this whenever the user asks about a stock price, share price, or market quote.

## Schema

```json
{
  "name": "get_stock_price",
  "description": "Fetch the latest stock market price for a given ticker symbol (e.g. GOOGL, AAPL, TSLA). Returns current price, previous close, and daily change. Use whenever the user asks about a stock price or market quote.",
  "parameters": {
    "type": "object",
    "properties": {
      "ticker": {
        "type": "string",
        "description": "The stock ticker symbol, e.g. 'GOOGL', 'AAPL', 'TSLA', 'MSFT'."
      }
    },
    "required": ["ticker"]
  }
}
```
