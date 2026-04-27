"""Prompt payload normalization helpers for Orchestrator.

These functions are intentionally stateless so they can be exercised without
constructing the full orchestration engine.
"""
from __future__ import annotations

from typing import Any, Dict

from src.core.state_model import normalize_state


def extract_audio_bytes(payload: Dict[str, Any]) -> bytes:
    if not isinstance(payload, dict):
        return b""
    raw = payload.get("audio_bytes")
    if isinstance(raw, bytes):
        return raw
    if isinstance(raw, bytearray):
        return bytes(raw)
    if isinstance(raw, memoryview):
        return raw.tobytes()
    return b""


def coerce_user_prompt_payload(user_message: Any) -> Dict[str, Any]:
    if isinstance(user_message, dict):
        payload = dict(user_message)
        text = str(
            payload.get("text")
            or payload.get("user_input")
            or payload.get("content")
            or ""
        ).strip()
        audio_bytes = extract_audio_bytes(payload)
        if not text and audio_bytes:
            mime = str(payload.get("audio_mime_type") or "audio/ogg")
            text = f"[Voice note · {len(audio_bytes)} bytes · {mime}]"

        return {
            "text": text,
            "audio_bytes": audio_bytes,
            "audio_mime_type": str(payload.get("audio_mime_type") or "audio/ogg"),
            "audio_source": str(payload.get("audio_source") or ""),
            "audio_file_id": str(payload.get("audio_file_id") or ""),
        }

    text = str(user_message or "").strip()
    return {
        "text": text,
        "audio_bytes": b"",
        "audio_mime_type": "",
        "audio_source": "",
        "audio_file_id": "",
    }


def state_has_audio_prompt(state: Dict[str, Any]) -> bool:
    prompt = state.get("user_prompt")
    if not isinstance(prompt, dict):
        return False
    return bool(extract_audio_bytes(prompt))


def build_user_prompt_message(state: Dict[str, Any]) -> Dict[str, Any]:
    user_text = str(state.get("user_input") or "")
    message: Dict[str, Any] = {
        "role": "user",
        "content": f"<user_input>{user_text}</user_input>",
    }
    prompt = state.get("user_prompt")
    if not isinstance(prompt, dict):
        return message

    audio_bytes = extract_audio_bytes(prompt)
    if audio_bytes:
        message["audio_bytes"] = audio_bytes
        message["audio_mime_type"] = str(prompt.get("audio_mime_type") or "audio/ogg")
    return message


def strip_audio_bytes_for_persistence(state: Dict[str, Any]) -> Dict[str, Any]:
    sanitized = normalize_state(dict(state))
    prompt = sanitized.get("user_prompt")
    if isinstance(prompt, dict) and "audio_bytes" in prompt:
        prompt = dict(prompt)
        prompt.pop("audio_bytes", None)
        sanitized["user_prompt"] = prompt
    sanitized.pop("_energy_gate_cleared", None)
    return sanitized
