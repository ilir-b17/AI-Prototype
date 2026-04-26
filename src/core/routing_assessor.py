"""Request routing assessment utilities.

RoutingAssessor is stateless except for a reference to the SkillRegistry
it reads tool schemas from.  It can be tested without an Orchestrator.
"""
from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

# -- Constants (moved verbatim from orchestrator.py) ----------------------------
_ROUTING_STOPWORDS = {
    "a", "an", "and", "are", "at", "be", "can", "could", "do", "for", "from",
    "hello", "hey", "hi", "i", "in", "is", "it", "me", "my", "now", "of",
    "on", "or", "please", "right", "tell", "the", "this", "to", "today", "what",
    "would", "you", "your",
    # Generic action verbs - too common to discriminate between tools
    "get", "give", "find", "show", "fetch", "list", "like", "have", "has",
    "want", "need", "also", "just", "did", "how",
    # Common discourse words that appear in tool descriptions but carry no routing signal
    "that", "which", "where", "when", "there", "then", "them", "they",
    "these", "those", "with", "been", "will", "use", "not",
}
_NON_UTILITY_TOOL_PREFIXES = ("update_", "request_", "spawn_", "run_", "execute_")
_NON_UTILITY_TOOL_NAMES = {
    "ask_admin_for_guidance",
    "consolidate_memory",
    "manage_file_system",
    "query_highest_priority_task",
    # Archival memory search requires supervisor-level reasoning about WHEN to use it;
    # auto-selecting it in fast-path on any message with 'memory' or 'information' is wrong.
    "search_archival_memory",
}
_ROUTING_TOKEN_RE = r"[a-z0-9]+"
_DIRECT_ROUTE_MAX_COMPLEXITY = 0
_SINGLE_TOOL_MAX_COMPLEXITY = 1
_SINGLE_TOOL_MIN_SCORE = 1.5
_SINGLE_TOOL_MIN_MARGIN = 0.5
_FAST_PATH_DIRECT_MAX_TOKENS = 12
_FAST_PATH_SINGLE_TOOL_MAX_TOKENS = 14
_FAST_PATH_SINGLE_TOOL_ALLOWLIST = frozenset(
    {
        "get_system_info",
        "get_stock_price",
        "weather_current",
        "unit_converter",
        "web_search",
    }
)
_FINANCE_INTENT_RE = re.compile(r"\b(?:prices?|quotes?|stocks?|tickers?|shares?|markets?|trading)\b", re.IGNORECASE)


class RoutingAssessor:
    """Scores incoming requests against the skill registry to select a dispatch mode."""

    # Uppercase tokens that are common words, not ticker symbols.
    _TICKER_STOPWORDS: frozenset = frozenset({
        "A", "I", "AN", "AT", "BE", "BY", "DO", "GO", "IF", "IN", "IS", "IT",
        "ME", "MY", "NO", "OF", "OK", "ON", "OR", "SO", "TO", "UP", "WE",
        "AND", "ARE", "BUT", "CAN", "FOR", "GET", "HAS", "HIM", "HIS", "HOW",
        "ITS", "MAY", "NOT", "NOW", "OFF", "OLD", "OUR", "OUT", "OWN", "THE",
        "TOO", "TWO", "USE", "WAS", "WHO", "WHY", "YES", "YET", "YOU",
        "ALSO", "BEEN", "BOTH", "DOES", "DONE", "EACH", "FROM", "GIVE",
        "GOOD", "HAVE", "HERE", "HIGH", "JUST", "KNOW", "LAST", "LIKE", "LIVE",
        "LONG", "LOOK", "MAKE", "MANY", "MORE", "MOST", "MUCH", "NEED", "NEXT",
        "ONCE", "ONLY", "OPEN", "OVER", "PAST", "REAL", "SAME", "SEND", "SHOW",
        "SOME", "SUCH", "SURE", "TAKE", "TELL", "THAN", "THAT", "THEM", "THEN",
        "THEY", "THIS", "TIME", "TOLD", "VERY", "WANT", "WELL", "WERE", "WHAT",
        "WHEN", "WITH", "WORK", "YEAR", "YOUR",
    })

    def __init__(self, registry: Any) -> None:
        """
        Args:
            registry: A SkillRegistry instance. Only get_schemas() is called.
        """
        self.registry = registry

    @staticmethod
    def _routing_keywords(text: str) -> set:
        return {
            token for token in re.findall(_ROUTING_TOKEN_RE, (text or "").lower())
            if len(token) > 2 and token not in _ROUTING_STOPWORDS
        }

    def _estimate_request_complexity(self, text: str) -> int:
        lowered = (text or "").lower()
        tokens = re.findall(_ROUTING_TOKEN_RE, lowered)
        score = 0

        if len(tokens) > 12:
            score += 1
        if len(tokens) > 24:
            score += 1
        if len(re.findall(r"[?.!]", text or "")) > 1:
            score += 1
        if any(marker in (text or "") for marker in ("\n", "```", ":", ";")):
            score += 1
        if re.search(r"\b(and|then|after|before|also|plus|compare|step|steps|plan)\b", lowered):
            score += 1
        if re.search(
            r"\b(build|create|implement|debug|fix|refactor|analyze|analyse|review|modify|update|delete|write|open|extract|read|summari[sz]e|code)\b",
            lowered,
        ):
            score += 1

        return score

    @staticmethod
    def _is_utility_tool_schema(schema: Dict[str, Any]) -> bool:
        name = schema.get("name", "")
        if not name or name in _NON_UTILITY_TOOL_NAMES or name.startswith(_NON_UTILITY_TOOL_PREFIXES):
            return False
        params = schema.get("parameters", {}) or {}
        required = list(params.get("required", []) or [])
        if len(required) > 1:
            return False
        if not required:
            return True
        # "ticker" is also handled by _prepare_utility_tool_arguments
        return required[0] in {"query", "url", "ticker"}

    @staticmethod
    def _tool_schema_keywords(schema: Dict[str, Any]) -> Dict[str, set]:
        params = (schema.get("parameters") or {}).get("properties", {}) or {}
        name_tokens = RoutingAssessor._routing_keywords(schema.get("name", "").replace("_", " "))
        desc_tokens = RoutingAssessor._routing_keywords(schema.get("description", ""))
        param_tokens = set()
        for param_name, param_schema in params.items():
            param_tokens |= RoutingAssessor._routing_keywords(param_name.replace("_", " "))
            param_tokens |= RoutingAssessor._routing_keywords(param_schema.get("description", ""))
        return {
            "name": name_tokens,
            "description": desc_tokens,
            "parameters": param_tokens,
        }

    @staticmethod
    def _extract_multiple_tickers(user_message: str) -> List[str]:
        """Return all valid uppercase ticker symbols found in the message (2+ expected for multi-ticker path)."""
        text = str(user_message or "")
        has_finance_intent = bool(_FINANCE_INTENT_RE.search(text))

        candidates = [
            tok for tok in re.findall(r"\b([A-Z]{1,5})\b", text)
            if tok not in RoutingAssessor._TICKER_STOPWORDS
        ]
        if len(candidates) < 2:
            return []

        lowercase_tokens = set(re.findall(r"\b([a-z]{1,5})\b", text))
        if any(candidate.lower() in lowercase_tokens for candidate in candidates):
            return []

        # Permit "Compare TSLA NVDA AMZN" style ticker requests even when
        # finance keywords are omitted.
        if not has_finance_intent and not re.search(r"\b(compare|vs|versus)\b", text, re.IGNORECASE):
            return []

        return candidates

    @staticmethod
    def _strip_optional_tool_fallback_clause(user_message: str) -> str:
        text = (user_message or "").strip()
        stripped = re.sub(
            r"(?:[,.!?]\s*|\s+)(?:please\s+)?(?:search|browse|check|look(?:\s+up)?|find|get|use)\s+(?:the\s+)?(?:web|internet|online)\b[^.?!]*?\bif\s+(?:you\s+)?(?:must|need(?:\s+to)?|have\s+to|required|necessary)\b.*$",
            "",
            text,
            flags=re.IGNORECASE,
        ).strip(" ,.!?;")
        return stripped or text

    def _prepare_utility_tool_arguments(
        self,
        schema: Dict[str, Any],
        user_message: str,
        chat_history: Optional[List[Dict[str, str]]] = None,
    ) -> Optional[Dict[str, Any]]:
        params = schema.get("parameters", {}) or {}
        properties = params.get("properties", {}) or {}
        required = list(params.get("required", []) or [])

        if not required:
            return {}

        if len(required) != 1:
            return None

        field_name = required[0]
        if field_name == "query":
            cleaned = re.sub(
                r"^(can|could|would|please|hey|hi|hello|nice|ok|okay|what about)\b[\s,]*",
                "",
                user_message.strip(),
                flags=re.IGNORECASE,
            ).strip(" ?!.,")
            # Strip residual "you [action-verb] [me] [for/about]?" prefix left after
            # removing polite openers. e.g. "Can you tell me X" -> "you tell me X" -> "X"
            cleaned = re.sub(
                r"^you\s+(?:search|look|find|browse|check|tell|get|show|fetch|look up|search for)\s+(?:me\s+)?(?:for\s+|about\s+|up\s+)?",
                "",
                cleaned,
                flags=re.IGNORECASE,
            ).strip(" ?!.,")
            if not cleaned:
                cleaned = user_message.strip()
            # If reduced to a bare action verb (e.g. "Please search" -> "search"),
            # look back at the last substantive user turn for the real topic.
            _bare_verbs = {"search", "find", "look", "do", "go", "try", "run", "proceed", "continue"}
            if cleaned.lower() in _bare_verbs and chat_history:
                for turn in reversed(chat_history):
                    if turn.get("role") == "user":
                        prev = turn.get("content", "").strip()
                        if len(prev) > 15:
                            cleaned = prev
                            break
            args = {"query": cleaned}
            if "max_results" in properties:
                args["max_results"] = 5
            return args

        if field_name == "url":
            url_match = re.search(r"https?://\S+", user_message)
            if not url_match:
                return None
            return {"url": url_match.group(0)}

        if field_name == "ticker":
            lowered_message = (user_message or "").lower()
            if not re.search(r"\b(stock|price|quote|ticker|market|trading|shares?)\b", lowered_message):
                return None

            # Extract uppercase ticker symbols (1-5 chars), excluding common words.
            candidates = [
                tok for tok in re.findall(r"\b([A-Z]{1,5})\b", user_message)
                if tok not in self._TICKER_STOPWORDS
            ]
            if len(candidates) == 1:
                return {"ticker": candidates[0]}
            # Zero or multiple tickers: let the supervisor handle it.
            return None

        return None

    def _score_tool_for_request(self, user_message: str, schema: Dict[str, Any]) -> float:
        query_keywords = self._routing_keywords(user_message)
        if not query_keywords:
            return 0.0
        schema_keywords = self._tool_schema_keywords(schema)
        lowered = (user_message or "").lower()
        tool_name = str(schema.get("name") or "")

        score = 0.0
        score += 2.5 * len(query_keywords & schema_keywords["name"])
        score += 1.5 * len(query_keywords & schema_keywords["description"])
        score += 0.75 * len(query_keywords & schema_keywords["parameters"])
        if not (schema.get("parameters", {}) or {}).get("required"):
            score += 0.25

        # Intent boosts for frequently-misrouted fast-path prompts.
        if tool_name == "get_system_info" and re.search(
            r"\b(time|date|timezone|system|platform|cpu|memory|os)\b", lowered
        ):
            score += 2.0
        elif tool_name == "get_stock_price" and (
            re.search(r"\b(stock|price|quote|ticker|market|trading)\b", lowered)
            or bool(re.search(r"\b([A-Z]{1,5})\b", str(user_message or "")))
        ):
            score += 2.0
        elif tool_name == "weather_current" and re.search(
            r"\b(weather|temperature|rain|raining|forecast|cold|hot|humid|humidity|wind|snow)\b",
            lowered,
        ):
            score += 2.0
        elif tool_name == "unit_converter" and re.search(
            r"\b(convert|conversion|how many|fahrenheit|celsius|miles?|km|kilometers?|kg|pounds?|bytes?|gb|mb|tb)\b",
            lowered,
        ):
            score += 2.0
        elif tool_name == "web_search" and re.search(
            r"\b(news|latest|headlines?|today|current|now|score)\b", lowered
        ):
            score += 1.5

        return score

    def assess_request_route(
        self,
        user_message: str,
        chat_history: Optional[List[Dict[str, str]]] = None,
    ) -> Dict[str, Any]:
        routing_message = self._strip_optional_tool_fallback_clause(user_message)
        complexity = self._estimate_request_complexity(routing_message)

        # Multi-ticker: two or more valid stock symbols -> sequential get_stock_price calls.
        # Checked before general scoring so the supervisor never sees this pattern.
        tickers = self._extract_multiple_tickers(routing_message)
        if len(tickers) >= 2:
            return {
                "mode": "multi_ticker",
                "tickers": tickers,
                "complexity": complexity,
            }

        candidates = []

        for schema in self.registry.get_schemas():
            if not self._is_utility_tool_schema(schema):
                continue
            arguments = self._prepare_utility_tool_arguments(schema, routing_message, chat_history)
            if arguments is None:
                continue
            score = self._score_tool_for_request(routing_message, schema)
            if score <= 0:
                continue
            candidates.append({
                "tool_name": schema.get("name", ""),
                "arguments": arguments,
                "score": score,
            })

        candidates.sort(key=lambda item: item["score"], reverse=True)
        top = candidates[0] if candidates else None
        next_score = candidates[1]["score"] if len(candidates) > 1 else 0.0
        trivial_direct = self._is_trivial_direct_intent(routing_message)

        if (
            top
            and complexity <= _SINGLE_TOOL_MAX_COMPLEXITY
            and top["score"] >= _SINGLE_TOOL_MIN_SCORE
            and (top["score"] - next_score) >= _SINGLE_TOOL_MIN_MARGIN
            and self._is_trivial_single_tool_intent(top["tool_name"], routing_message, complexity)
        ):
            return {
                "mode": "single_tool",
                "complexity": complexity,
                "tool_name": top["tool_name"],
                "arguments": top["arguments"],
            }

        if (
            trivial_direct
            and complexity <= _DIRECT_ROUTE_MAX_COMPLEXITY
        ):
            return {"mode": "direct", "complexity": complexity}

        return {"mode": "graph", "complexity": complexity}

    @staticmethod
    def _is_trivial_direct_intent(user_message: str) -> bool:
        lowered = (user_message or "").strip().lower()
        if not lowered:
            return False

        if re.fullmatch(r"(?:please\s+)?(?:search|find|lookup|look up)[.!? ]*", lowered):
            return True

        if re.fullmatch(
            r"(?:hi|hello|hey)(?:\s+[a-z0-9]{1,20})?[!.? ]*|(?:thanks|thank you|ok|okay|good morning|good evening|good night)[!.? ]*",
            lowered,
        ):
            return True

        token_count = len(re.findall(_ROUTING_TOKEN_RE, lowered))
        if token_count > _FAST_PATH_DIRECT_MAX_TOKENS:
            return False

        # Domain intents should not bypass fast-path tools.
        if re.search(
            r"\b(stock|price|quote|ticker|market|trading|weather|temperature|forecast|convert|conversion|fahrenheit|celsius|kg|pounds|bytes|time|date|timezone|cpu|memory|os|platform)\b",
            lowered,
        ):
            return False

        return bool(re.search(r"\b(what is|what's|who is|define|meaning of)\b", lowered))

    @staticmethod
    def _is_trivial_single_tool_intent(tool_name: str, user_message: str, complexity: int) -> bool:
        if tool_name not in _FAST_PATH_SINGLE_TOOL_ALLOWLIST:
            return False
        if complexity > _SINGLE_TOOL_MAX_COMPLEXITY:
            return False

        lowered = (user_message or "").lower()
        token_count = len(re.findall(_ROUTING_TOKEN_RE, lowered))
        if token_count > _FAST_PATH_SINGLE_TOOL_MAX_TOKENS:
            return False

        if re.search(r"\b(and then|then|after|before|also|plus|compare|step|steps|plan)\b", lowered):
            return False

        if tool_name == "get_system_info":
            return bool(re.search(r"\b(time|date|timezone|system info|system information|platform|cpu|memory|os)\b", lowered))
        if tool_name == "get_stock_price":
            return bool(re.search(r"\b(stock|price|quote|ticker|market)\b", lowered))
        if tool_name == "weather_current":
            if re.search(r"\b(cause|causes|why|pattern|patterns|climate|explain|explanation)\b", lowered):
                return False
            return bool(re.search(r"\b(weather|temperature|rain|raining|forecast|cold|hot|humid|humidity|wind|snow)\b", lowered))
        if tool_name == "unit_converter":
            return bool(
                re.search(
                    r"\b(convert|conversion|how many|fahrenheit|celsius|miles?|km|kilometers?|kg|pounds?|bytes?|gb|mb|tb)\b",
                    lowered,
                )
            )
        if tool_name == "web_search":
            return bool(
                re.search(
                    r"\b(news|latest|current|today|now|headlines?|score|weather|temperature|forecast|rain|snow)\b",
                    lowered,
                )
            )

        return False
