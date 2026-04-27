"""Capability catalog helpers for the Orchestrator facade."""
from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Optional

from src.core.orchestrator_constants import (
    _CATALOG_MATCH_STOPWORDS,
    _CATALOG_META_TOOL_NAMES,
)

logger = logging.getLogger(__name__)
_CATALOG_TOKEN_RE = r"[a-z0-9]+"


def load_capability_catalog_entries(
    registry: Any,
    *,
    log: Optional[logging.Logger] = None,
) -> List[Dict[str, str]]:
    active_logger = log or logger
    if registry is None:
        return []

    catalog_getter = getattr(registry, "get_skill_catalog", None)
    if callable(catalog_getter):
        try:
            raw_catalog = catalog_getter()
            if isinstance(raw_catalog, list):
                return [item for item in raw_catalog if isinstance(item, dict)]
        except Exception as exc:
            active_logger.warning("Failed to load skill catalog from registry: %s", exc)

    schema_getter = getattr(registry, "get_schemas", None)
    if not callable(schema_getter):
        return []

    fallback: List[Dict[str, str]] = []
    for schema in schema_getter():
        fallback.append(
            {
                "name": str(schema.get("name") or "").strip(),
                "description": str(schema.get("description") or "").strip(),
            }
        )
    return fallback


def load_executable_capability_catalog_entries(
    registry: Any,
    *,
    log: Optional[logging.Logger] = None,
) -> List[Dict[str, str]]:
    active_logger = log or logger
    if registry is not None:
        executable_getter = getattr(registry, "get_executable_skill_catalog", None)
        if callable(executable_getter):
            try:
                entries = executable_getter()
                if isinstance(entries, list):
                    return [item for item in entries if isinstance(item, dict)]
            except Exception as exc:
                active_logger.warning("Failed to load executable skill catalog from registry: %s", exc)
    return load_capability_catalog_entries(registry, log=active_logger)


def build_capability_catalog_rows(entries: List[Dict[str, str]]) -> List[str]:
    rows: List[str] = []
    for item in entries:
        name = str(item.get("name") or "").strip()
        description = str(item.get("description") or "").strip()
        if name and description:
            rows.append(f"- {name}: {description}")
    return rows


def get_capabilities_string(entries: List[Dict[str, str]]) -> str:
    rows = build_capability_catalog_rows(entries)
    if not rows:
        return "Available skills catalog (name: description): none loaded"
    return "Available skills catalog (name: description):\n" + "\n".join(rows)


def build_scoped_skill_runtime_context(
    registry: Any,
    skill_name: str,
    catalog_entries: List[Dict[str, str]],
) -> str:
    if registry is None:
        return ""

    get_skill_body = getattr(registry, "get_skill_body", None)
    if not callable(get_skill_body):
        return ""

    raw_body = str(get_skill_body(skill_name) or "").strip()
    if not raw_body:
        return ""

    description = ""
    for item in catalog_entries:
        if str(item.get("name") or "").strip() == skill_name:
            description = str(item.get("description") or "").strip()
            break

    return (
        "<scoped_skill_context>\n"
        f"Skill: {skill_name}\n"
        f"Description: {description}\n"
        "Scope: immediate execution turn only\n\n"
        "SKILL_BODY:\n"
        f"{raw_body}\n"
        "</scoped_skill_context>"
    )


def capability_catalog_tokens(text: str) -> set[str]:
    return {
        token
        for token in re.findall(_CATALOG_TOKEN_RE, str(text or "").lower())
        if len(token) > 2 and token not in _CATALOG_MATCH_STOPWORDS
    }


def find_local_skill_catalog_match(
    gap_description: str,
    suggested_tool_name: str,
    entries: List[Dict[str, str]],
) -> Optional[Dict[str, str]]:
    query_tokens = capability_catalog_tokens(gap_description)
    query_tokens |= capability_catalog_tokens(str(suggested_tool_name or "").replace("_", " "))
    if not query_tokens:
        return None

    best_item: Optional[Dict[str, str]] = None
    best_score = 0.0

    for item in entries:
        name = str(item.get("name") or "").strip()
        if not name or name in _CATALOG_META_TOOL_NAMES:
            continue

        name_tokens = capability_catalog_tokens(name.replace("_", " "))
        desc_tokens = capability_catalog_tokens(item.get("description", ""))

        score = 0.0
        if suggested_tool_name and name.lower() == str(suggested_tool_name).strip().lower():
            score += 6.0
        score += 3.0 * len(query_tokens & name_tokens)
        score += 1.5 * len(query_tokens & desc_tokens)

        if score > best_score:
            best_item = {"name": name, "description": str(item.get("description") or "").strip()}
            best_score = score

    return best_item if best_score >= 1.5 else None

