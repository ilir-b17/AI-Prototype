"""
Ground-truth routing test cases for AIDEN's _assess_request_route.

Each case specifies:
  input_message: the user message
  expected_mode: "direct" | "single_tool" | "multi_ticker" | "graph"
  expected_tool: (for single_tool mode) which tool should be selected
  expected_tickers: (for multi_ticker mode) list of ticker symbols
  should_not_match_tool: (optional) tool that must NOT be selected
  tags: for filtering
"""

from __future__ import annotations
from typing import Any, Dict, List, Optional


def _case(
    id: str,
    message: str,
    mode: str,
    tool: Optional[str] = None,
    tickers: Optional[List[str]] = None,
    excluded_tool: Optional[str] = None,
    tags: Optional[List[str]] = None,
) -> Dict[str, Any]:
    return {
        "id": id,
        "message": message,
        "expected_mode": mode,
        "expected_tool": tool,
        "expected_tickers": tickers or [],
        "excluded_tool": excluded_tool,
        "tags": tags or [],
    }


# -- Direct response cases (no tool, simple conversational) --------------------
DIRECT_CASES = [
    _case("routing_001", "Hi", "direct", tags=["direct", "greeting"]),
    _case("routing_002", "Hello there", "direct", tags=["direct", "greeting"]),
    _case("routing_003", "Thanks", "direct", tags=["direct", "greeting"]),
    _case("routing_004", "Good morning", "direct", tags=["direct", "greeting"]),
    _case("routing_005", "What is machine learning?", "direct", tags=["direct", "definition"]),
    _case("routing_006", "What's the capital of France?", "direct", tags=["direct", "definition"]),
    _case("routing_007", "Who is Alan Turing?", "direct", tags=["direct", "definition"]),
    _case("routing_008", "Define entropy", "direct", tags=["direct", "definition"]),
    _case("routing_009", "Ok", "direct", tags=["direct", "greeting"]),
    _case("routing_010", "What is 2 + 2?", "direct", tags=["direct", "simple"]),
]

# -- System info fast path ------------------------------------------------------
SYSTEM_INFO_CASES = [
    _case("routing_011", "What time is it?", "single_tool", "get_system_info",
          tags=["single_tool", "system_info"]),
    _case("routing_012", "What is today's date?", "single_tool", "get_system_info",
          tags=["single_tool", "system_info"]),
    _case("routing_013", "What's my system's CPU usage?", "single_tool", "get_system_info",
          tags=["single_tool", "system_info"]),
    _case("routing_014", "What OS am I running?", "single_tool", "get_system_info",
          tags=["single_tool", "system_info"]),
    _case("routing_015", "What timezone am I in?", "single_tool", "get_system_info",
          tags=["single_tool", "system_info"]),
    _case("routing_016", "Tell me my platform info", "single_tool", "get_system_info",
          tags=["single_tool", "system_info"]),
]

# -- Stock price fast path ------------------------------------------------------
STOCK_CASES = [
    _case("routing_017", "What is the AAPL stock price?", "single_tool", "get_stock_price",
          tags=["single_tool", "stock"]),
    _case("routing_018", "TSLA stock quote", "single_tool", "get_stock_price",
          tags=["single_tool", "stock"]),
    _case("routing_019", "Current price of MSFT", "single_tool", "get_stock_price",
          tags=["single_tool", "stock"]),
    _case("routing_020", "How is NVDA trading today?", "single_tool", "get_stock_price",
          tags=["single_tool", "stock"]),
]

# -- Multi-ticker fast path -----------------------------------------------------
MULTI_TICKER_CASES = [
    _case("routing_021", "What are the prices of AAPL and MSFT?", "multi_ticker",
          tickers=["AAPL", "MSFT"], tags=["multi_ticker"]),
    _case("routing_022", "Compare TSLA NVDA AMZN stock prices", "multi_ticker",
          tickers=["TSLA", "NVDA", "AMZN"], tags=["multi_ticker"]),
    _case("routing_023", "Show me GOOG and META market quotes", "multi_ticker",
          tickers=["GOOG", "META"], tags=["multi_ticker"]),
]

# -- Weather fast path ----------------------------------------------------------
WEATHER_CASES = [
    _case("routing_024", "What is the weather in Vienna?", "single_tool", "weather_current",
          tags=["single_tool", "weather"]),
    _case("routing_025", "Current temperature in Tokyo", "single_tool", "weather_current",
          tags=["single_tool", "weather"]),
    _case("routing_026", "Is it raining in London today?", "single_tool", "weather_current",
          tags=["single_tool", "weather"]),
    _case("routing_027", "Weather forecast for Berlin", "single_tool", "weather_current",
          tags=["single_tool", "weather"]),
    _case("routing_028", "How cold is it in New York?", "single_tool", "weather_current",
          tags=["single_tool", "weather"]),
]

# -- Unit converter fast path ---------------------------------------------------
UNIT_CASES = [
    _case("routing_029", "Convert 100 km to miles", "single_tool", "unit_converter",
          tags=["single_tool", "unit"]),
    _case("routing_030", "How many kg is 180 pounds?", "single_tool", "unit_converter",
          tags=["single_tool", "unit"]),
    _case("routing_031", "Convert 32 fahrenheit to celsius", "single_tool", "unit_converter",
          tags=["single_tool", "unit"]),
    _case("routing_032", "How many bytes in 5 GB?", "single_tool", "unit_converter",
          tags=["single_tool", "unit"]),
]

# -- Web search fast path -------------------------------------------------------
WEB_SEARCH_CASES = [
    _case("routing_033", "What is in the news today?", "single_tool", "web_search",
          tags=["single_tool", "web_search"]),
    _case("routing_034", "Latest headlines", "single_tool", "web_search",
          tags=["single_tool", "web_search"]),
    _case("routing_035", "Current weather in Sydney", "single_tool", "weather_current",
          excluded_tool="web_search",
          tags=["single_tool", "weather", "not_web_search"]),
]

# -- Graph mode cases (complex, multi-step, or no single tool match) ----------
GRAPH_CASES = [
    _case("routing_036",
          "Build me a stock screening pipeline that alerts me when AAPL drops 5%",
          "graph", tags=["graph", "complex"]),
    _case("routing_037",
          "Read the quarterly_report.pdf in my downloads and summarize it",
          "graph", tags=["graph", "pdf"]),
    _case("routing_038",
          "Search my memory for notes about the AIDEN refactor and then update my objectives",
          "graph", tags=["graph", "memory", "multi_step"]),
    _case("routing_039",
          "Analyze the sales.csv file and compare it to last quarter's numbers",
          "graph", tags=["graph", "analysis"]),
    _case("routing_040",
          "What is the weather in Vienna and also get me the AAPL stock price and then write a morning briefing",
          "graph", tags=["graph", "multi_step"]),
    _case("routing_041",
          "Create an Epic for building a portfolio tracker with price alerts, weekly reports, and tax loss harvesting suggestions",
          "graph", tags=["graph", "planning"]),
    _case("routing_042",
          "Debug the failing tests in src/core/orchestrator.py then commit the fix",
          "graph", tags=["graph", "complex", "code"]),
    _case("routing_043",
          "Search my archival memory for past conversations about energy budgeting and summarise the key decisions",
          "graph", tags=["graph", "memory"]),
    _case("routing_044",
          "What is the weather in Vienna?\nAlso, can you write a haiku about rain afterwards?",
          "graph", tags=["graph", "multi_step"]),
]

# -- Edge cases -----------------------------------------------------------------
EDGE_CASES = [
    # Should NOT trigger weather fast path (needs research)
    _case("routing_045",
          "What causes the weather patterns in the Alps?",
          "graph", excluded_tool="weather_current",
          tags=["edge", "graph"]),
    # Single ticker with common word -- should NOT go to multi_ticker
    _case("routing_046",
          "What is the A stock price?", "graph",
          tags=["edge", "ticker_stopword"]),
    # Ambiguous -- "search" alone is not enough for web_search fast path
    _case("routing_047",
          "Please search", "direct",
          tags=["edge", "ambiguous"]),
    # Multi-step with tool name in message -- should go to graph
    _case("routing_048",
          "Use web_search to find today's AI news and then add it to my objectives",
          "graph", tags=["edge", "multi_step"]),
]

ALL_ROUTING_CASES = (
    DIRECT_CASES
    + SYSTEM_INFO_CASES
    + STOCK_CASES
    + MULTI_TICKER_CASES
    + WEATHER_CASES
    + UNIT_CASES
    + WEB_SEARCH_CASES
    + GRAPH_CASES
    + EDGE_CASES
)
