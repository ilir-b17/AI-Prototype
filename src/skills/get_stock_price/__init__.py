import asyncio
import logging

logger = logging.getLogger(__name__)


def _sync_get_stock_price(ticker: str) -> str:
    try:
        import yfinance as yf

        t = yf.Ticker(ticker.upper())
        fi = t.fast_info

        price = fi.last_price
        prev_close = fi.previous_close
        currency = getattr(fi, "currency", "USD")
        exchange = getattr(fi, "exchange", "N/A")

        if price is None:
            return f"Could not retrieve price for ticker '{ticker.upper()}'. Check the symbol is correct."

        change = price - prev_close if prev_close else 0.0
        pct = (change / prev_close * 100) if prev_close else 0.0
        direction = "▲" if change >= 0 else "▼"

        # Try to get a human-readable name
        try:
            name = t.info.get("shortName") or ticker.upper()
        except Exception:
            name = ticker.upper()

        return (
            f"{name} ({ticker.upper()}) — {exchange}\n"
            f"Price  : {currency} {price:,.4f}\n"
            f"Change : {direction} {abs(change):,.4f} ({pct:+.2f}% vs prev close)\n"
            f"Prev Close: {currency} {prev_close:,.4f}"
        )
    except Exception as exc:
        return f"Error fetching stock price for '{ticker}': {exc}"


async def get_stock_price(ticker: str) -> str:
    """Fetch the latest market price for a stock ticker symbol using yfinance."""
    logger.info(f"get_stock_price: {ticker!r}")
    return await asyncio.to_thread(_sync_get_stock_price, ticker)
