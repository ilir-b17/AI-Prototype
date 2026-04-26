"""Cloud payload redaction utilities.

All functions are pure (no side effects, no I/O).  They can be tested in
isolation without instantiating the Orchestrator.
"""
from __future__ import annotations

import re
from typing import Dict, List

# ── Redaction sentinels ────────────────────────────────────────────────────────
_REDACTED_CONTEXT_BLOCK = "[REDACTED_CONTEXT_BLOCK]"
_REDACTED_SENSORY_STATE = "[REDACTED_SENSORY_STATE]"
_REDACTED_EMPTY_PAYLOAD = "[REDACTED_EMPTY_PAYLOAD]"

# ── Block patterns (match XML-wrapped sections) ────────────────────────────────
_CLOUD_REDACTION_BLOCK_PATTERNS = (
    r"<Core_Working_Memory>.*?</Core_Working_Memory>",
    r"<Archival_Memory>.*?</Archival_Memory>",
    r"<chat_history>.*?</chat_history>",
)

# ── Line-level sensory patterns ────────────────────────────────────────────────
_CLOUD_REDACTION_LINE_PATTERNS = (
    (r"\[Machine Context[^\]]*\].*?(?=\n\s*\n|$)", _REDACTED_SENSORY_STATE),
    (
        r"^\s*(Host[_ ]?OS|OS|CPU(?:[_ ]?usage)?|CWD|Platform|Memory Usage"
        r"|Disk Usage)\s*[:=].*$",
        _REDACTED_SENSORY_STATE,
    ),
    (
        r"<\s*(host_os|os|cpu_usage|cwd|machine|platform)\s*>.*?"
        r"<\s*/\s*(host_os|os|cpu_usage|cwd|machine|platform)\s*>",
        _REDACTED_SENSORY_STATE,
    ),
)

# ── PII patterns ───────────────────────────────────────────────────────────────
_CLOUD_REDACTION_PII_PATTERNS = (
    (r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b", "[REDACTED_EMAIL]"),
    (r"\+?\d[\d\-\s().]{7,}\d", "[REDACTED_PHONE]"),
    (
        r"\b\d{1,5}\s+[A-Za-z0-9.\- ]+\s(?:Street|St|Avenue|Ave|Road|Rd"
        r"|Boulevard|Blvd|Lane|Ln|Drive|Dr)\b",
        "[REDACTED_ADDRESS]",
    ),
    (
        r"\b(?:my name is|i am|i'm)\s+[A-Za-z][A-Za-z'\-]+"
        r"(?:\s+[A-Za-z][A-Za-z'\-]+){0,2}\b",
        "[REDACTED_NAME]",
    ),
    (
        r"\b(?:i live in|located in|from)\s+[A-Za-z][A-Za-z'\-]+"
        r"(?:[\s,]+[A-Za-z][A-Za-z'\-]+){0,2}\b",
        "[REDACTED_LOCATION]",
    ),
    (
        r"\b(?:user[_ ]?id|ssn|social security|passport|account number)\b[^\n]*",
        "[REDACTED_PII_FIELD]",
    ),
    (r"[A-Za-z]:\\[^\n]*", "[REDACTED_PATH]"),
)


def is_safe_redacted_context_line(line: str) -> bool:
    """Return True if *line* is empty, a section header, or a redaction sentinel."""
    stripped = line.strip()
    if not stripped:
        return True
    if re.fullmatch(
        r"---\s*(Sensory Context|Core Memory|Archival Memory)\s*---",
        stripped,
        flags=re.IGNORECASE,
    ):
        return True
    return bool(re.fullmatch(r"\[REDACTED_[A-Z_]+\]", stripped))


def redact_context_and_memory_contents(text: str) -> str:
    """Strip sensitive sub-blocks inside <context_and_memory> tags."""

    def _replace(match: re.Match) -> str:
        opening, inner, closing = match.group(1), match.group(2), match.group(3)
        redacted_inner = inner
        for pattern in _CLOUD_REDACTION_BLOCK_PATTERNS:
            redacted_inner = re.sub(
                pattern, _REDACTED_CONTEXT_BLOCK, redacted_inner,
                flags=re.IGNORECASE | re.DOTALL,
            )
        for pattern, replacement in _CLOUD_REDACTION_LINE_PATTERNS:
            redacted_inner = re.sub(
                pattern, replacement, redacted_inner,
                flags=re.IGNORECASE | re.DOTALL | re.MULTILINE,
            )
        kept_lines = []
        dropped_content = False
        for line in redacted_inner.splitlines():
            if is_safe_redacted_context_line(line):
                kept_lines.append(line)
            else:
                dropped_content = True
        if dropped_content:
            kept_lines.append(_REDACTED_CONTEXT_BLOCK)
        safe_inner = "\n".join(kept_lines).strip()
        return f"{opening}\n{safe_inner}\n{closing}"

    return re.sub(
        r"(<context_and_memory>)(.*?)(</context_and_memory>)",
        _replace,
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )


def redact_text_for_cloud(
    text: str,
    *,
    allow_sensitive_context: bool = False,
    max_chars: int = 3500,
) -> str:
    """Redact a single text string before it is sent to a cloud provider."""
    raw_text = str(text or "")
    if not raw_text:
        return ""
    if allow_sensitive_context:
        compact = raw_text.strip()
        if len(compact) > max_chars:
            return compact[:max_chars].rstrip() + "\n[TRUNCATED_FOR_SIZE]"
        return compact

    redacted = redact_context_and_memory_contents(raw_text)
    for pattern in _CLOUD_REDACTION_BLOCK_PATTERNS:
        redacted = re.sub(
            pattern, _REDACTED_CONTEXT_BLOCK, redacted,
            flags=re.IGNORECASE | re.DOTALL,
        )
    for pattern, replacement in _CLOUD_REDACTION_LINE_PATTERNS:
        redacted = re.sub(
            pattern, replacement, redacted,
            flags=re.IGNORECASE | re.DOTALL | re.MULTILINE,
        )
    for pattern, replacement in _CLOUD_REDACTION_PII_PATTERNS:
        redacted = re.sub(pattern, replacement, redacted, flags=re.IGNORECASE)

    redacted = re.sub(r"\n{3,}", "\n\n", redacted).strip()
    if not redacted:
        return _REDACTED_EMPTY_PAYLOAD
    if len(redacted) > max_chars:
        return redacted[:max_chars].rstrip() + "\n[TRUNCATED_FOR_PRIVACY]"
    return redacted


def redact_messages_for_cloud(
    messages: List[Dict[str, str]],
    *,
    allow_sensitive_context: bool = False,
) -> List[Dict[str, str]]:
    """Reduce a messages list to a safe subset for cloud transmission."""
    if not messages:
        return [{"role": "user", "content": _REDACTED_EMPTY_PAYLOAD}]

    if allow_sensitive_context:
        source_messages = list(messages)
    else:
        system_messages = [m for m in messages if str(m.get("role", "")).lower() == "system"]
        user_messages   = [m for m in messages if str(m.get("role", "")).lower() == "user"]
        source_messages = []
        if system_messages:
            source_messages.append(system_messages[0])
            if len(system_messages) > 1:
                source_messages.append(system_messages[-1])
        if user_messages:
            source_messages.append(user_messages[-1])
        elif messages:
            source_messages.append(messages[-1])

    sanitized: List[Dict[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for message in source_messages:
        role = str(message.get("role") or "user")
        content = redact_text_for_cloud(
            str(message.get("content") or ""),
            allow_sensitive_context=allow_sensitive_context,
        )
        if not content:
            continue
        key = (role, content)
        if key in seen:
            continue
        seen.add(key)
        sanitized.append({"role": role, "content": content})

    return sanitized or [{"role": "user", "content": _REDACTED_EMPTY_PAYLOAD}]


def compute_payload_sha256(
    messages: List[Dict[str, str]],
    allowed_tools,
) -> str:
    """Return a SHA-256 digest of the outgoing cloud payload for audit logging."""
    import hashlib, json
    payload = {
        "messages": list(messages or []),
        "allowed_tools": list(allowed_tools or []),
    }
    serialized = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()
