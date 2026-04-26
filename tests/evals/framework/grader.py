"""
Rubric-based LLM grader for Tier 3 response quality evals.

Uses System 1 (local Ollama) to grade responses against a rubric.
Never calls cloud APIs. Gracefully skips when Ollama is offline.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

_GRADER_SYSTEM_PROMPT = (
    "You are a strict response quality grader. "
    "You receive a user prompt, an AI response, and a grading rubric. "
    "Return ONLY valid JSON with exactly these keys: "
    '{"scores": {"<criterion>": <1-5>, ...}, "reasoning": "<brief>", "passed": true/false}. '
    "passed is true if ALL criterion scores are >= 3. "
    "No markdown. No prose outside the JSON object."
)


def _parse_grader_response(raw: str) -> Optional[Dict[str, Any]]:
    text = str(raw or "").strip()
    if not text:
        return None

    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s*```$", "", text).strip()

    try:
        payload = json.loads(text)
        return payload if isinstance(payload, dict) else None
    except (ValueError, TypeError):
        pass

    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None

    try:
        payload = json.loads(text[start : end + 1])
        return payload if isinstance(payload, dict) else None
    except (ValueError, TypeError):
        return None


def _extract_message_content(response: Any) -> str:
    if isinstance(response, dict):
        message = response.get("message") or {}
        if isinstance(message, dict):
            return str(message.get("content", "") or "")
        return str(getattr(message, "content", "") or "")

    message = getattr(response, "message", None)
    if isinstance(message, dict):
        return str(message.get("content", "") or "")
    if message is not None:
        return str(getattr(message, "content", "") or "")

    return str(getattr(response, "content", "") or "")


def _extract_model_name(model_entry: Any) -> str:
    if isinstance(model_entry, dict):
        return str(model_entry.get("name") or model_entry.get("model") or "")
    return str(getattr(model_entry, "name", "") or getattr(model_entry, "model", "") or "")


def _available_model_names_from_payload(models_payload: Any) -> list[str]:
    if isinstance(models_payload, dict):
        models = list(models_payload.get("models") or [])
    else:
        models = list(getattr(models_payload, "models", []) or [])

    names: list[str] = []
    for model in models:
        name = _extract_model_name(model).strip()
        if name:
            names.append(name)
    return names


def _model_name_matches(requested_model: str, available_name: str) -> bool:
    requested = requested_model.strip()
    available = available_name.strip()
    if not requested or not available:
        return False
    if requested == available:
        return True
    if requested in available:
        return True
    requested_base = requested.split(":", 1)[0]
    available_base = available.split(":", 1)[0]
    return bool(requested_base and requested_base == available_base)


async def grade_response(
    *,
    user_prompt: str,
    ai_response: str,
    rubric: Dict[str, str],
    local_model: str = "gemma4:e4b",
    timeout_seconds: float = 60.0,
) -> Dict[str, Any]:
    """Grade an AI response against a rubric using the local model.

    Returns dict with keys: scores, reasoning, passed, grader_error.
    On grader failure, returns passed=False with grader_error set.
    """
    user_payload = json.dumps(
        {
            "user_prompt": user_prompt,
            "ai_response": ai_response,
            "rubric": {
                criterion: f"{description} (score 1-5, pass >= 3)"
                for criterion, description in rubric.items()
            },
        },
        indent=2,
    )

    messages = [
        {"role": "system", "content": _GRADER_SYSTEM_PROMPT},
        {"role": "user", "content": user_payload},
    ]

    try:
        import ollama

        client = ollama.AsyncClient()
        response = await asyncio.wait_for(
            client.chat(
                model=local_model,
                messages=messages,
                options={"temperature": 0, "num_predict": 300},
            ),
            timeout=timeout_seconds,
        )

        raw_content = _extract_message_content(response)
        parsed = _parse_grader_response(raw_content)
        if parsed is None:
            return {
                "scores": {},
                "reasoning": f"Grader returned unparseable output: {raw_content[:200]}",
                "passed": False,
                "grader_error": "parse_failure",
            }

        return {
            "scores": parsed.get("scores", {}),
            "reasoning": str(parsed.get("reasoning", "")),
            "passed": bool(parsed.get("passed", False)),
            "grader_error": None,
        }

    except asyncio.TimeoutError:
        return {
            "scores": {},
            "reasoning": "Grader timed out",
            "passed": False,
            "grader_error": "timeout",
        }
    except Exception as exc:
        logger.debug("Local grader failed: %s", exc, exc_info=True)
        return {
            "scores": {},
            "reasoning": str(exc),
            "passed": False,
            "grader_error": str(exc),
        }


def is_ollama_available(local_model: str = "gemma4:e4b") -> bool:
    """Check if Ollama is running with the required model."""
    try:
        import ollama

        models_payload = ollama.list()
        if asyncio.iscoroutine(models_payload):
            models_payload = asyncio.run(models_payload)
        available_names = _available_model_names_from_payload(models_payload)

        requested = str(local_model or "").strip()
        if not requested:
            return bool(available_names)

        return any(_model_name_matches(requested, name) for name in available_names)
    except Exception:
        return False
