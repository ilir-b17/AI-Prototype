import asyncio
import logging

logger = logging.getLogger(__name__)


import json
import os

def _sync_get_stock_price(ticker: str) -> str:
    try:
        import yfinance as yf

        t = yf.Ticker(ticker.upper())
        fi = t.fast_info

        # fast_info throws KeyError or AttributeError if the ticker is totally invalid in newer yfinance versions
        try:
            price = fi.last_price
        except Exception:
            price = None

        if price is None:
            return json.dumps({
                "status": "error",
                "message": f"Could not retrieve price for ticker '{ticker.upper()}'",
                "suggestion": "Check if the stock symbol is correct or if it has been delisted."
            })

        prev_close = fi.previous_close
        currency = getattr(fi, "currency", "USD")
        exchange = getattr(fi, "exchange", "N/A")

        change = price - prev_close if prev_close else 0.0
        pct = (change / prev_close * 100) if prev_close else 0.0
        direction = "up" if change >= 0 else "down"

        # Try to get a human-readable name, fall back to ticker
        try:
            name = t.info.get("shortName") or ticker.upper()
        except Exception:
            name = ticker.upper()

        return json.dumps({
            "status": "success",
            "data": {
                "name": name,
                "ticker": ticker.upper(),
                "exchange": exchange,
                "currency": currency,
                "current_price": round(price, 4),
                "previous_close": round(prev_close, 4) if prev_close else None,
                "change_amount": round(change, 4),
                "change_percent": round(pct, 2),
                "direction": direction
            }
        }, indent=2)

    except Exception as exc:
        return json.dumps({
            "status": "error",
            "message": f"Error fetching stock price for '{ticker}'",
            "details": str(exc),
            "suggestion": "yfinance might be rate-limited, or the ticker is invalid. Wait a bit or try another tool."
        })


async def get_stock_price(ticker: str) -> str:
    """Fetch the latest market price for a stock ticker symbol using yfinance."""
    logger.info(f"get_stock_price: {ticker!r}")

    if not isinstance(ticker, str) or not ticker.strip():
         return json.dumps({
             "status": "error",
             "message": "Invalid ticker",
             "details": "Ticker must be a non-empty string."
         })

    try:
        timeout_seconds = float(os.getenv("TOOL_EXEC_TIMEOUT_SECONDS", "10.0"))
    except ValueError:
        timeout_seconds = 10.0

    try:
        return await asyncio.wait_for(
            asyncio.to_thread(_sync_get_stock_price, ticker),
            timeout=timeout_seconds
        )
    except asyncio.TimeoutError:
        return json.dumps({
            "status": "error",
            "message": f"Stock fetch timed out after {timeout_seconds}s",
            "details": "Yahoo Finance API is taking too long to respond."
        })
