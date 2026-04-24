"""
Security Module for Core Agent Operations.

This module provides security checks, including Multi-Factor Authentication (MFA)
using passphrase challenges, to authorize sensitive operations like modifying
the charter or altering core memory.
"""

import os
import re
from typing import Optional

try:
    import pyotp
except ImportError:  # pragma: no cover - exercised only when optional dependency is missing
    pyotp = None


_MIN_MFA_PASSPHRASE_LENGTH = 12
_COMMON_MFA_PASSPHRASES: frozenset[str] = frozenset({
    "blue",
    "password",
    "admin",
    "secret",
    "aiden",
    "test",
})


def _normalize_mfa_text(value: str) -> str:
    return re.sub(
        r"\s+",
        " ",
        re.sub(r"[^\w\s]", " ", value.lower()),
    ).strip()


def validate_mfa_passphrase(passphrase: str) -> None:
    cleaned_passphrase = str(passphrase or "").strip()
    if not cleaned_passphrase:
        raise RuntimeError("MFA_PASSPHRASE must be set before startup.")
    if len(cleaned_passphrase) < _MIN_MFA_PASSPHRASE_LENGTH:
        raise RuntimeError("MFA_PASSPHRASE must be at least 12 characters long.")

    normalized = _normalize_mfa_text(cleaned_passphrase)
    normalized_words = set(normalized.split())
    if normalized in _COMMON_MFA_PASSPHRASES or normalized_words & _COMMON_MFA_PASSPHRASES:
        raise RuntimeError("MFA_PASSPHRASE must not use a common authorization word.")


def get_configured_mfa_passphrase() -> str:
    passphrase = os.getenv("MFA_PASSPHRASE", "")
    validate_mfa_passphrase(passphrase)
    return passphrase.strip()


def validate_mfa_configuration() -> None:
    get_configured_mfa_passphrase()
    if os.getenv("MFA_TOTP_SECRET", "").strip() and pyotp is None:
        raise RuntimeError("MFA_TOTP_SECRET is set but pyotp is not installed.")


def _verify_totp_challenge(user_response: str) -> bool:
    totp_secret = os.getenv("MFA_TOTP_SECRET", "").strip()
    if not totp_secret:
        return False
    if pyotp is None:
        raise RuntimeError("MFA_TOTP_SECRET is set but pyotp is not installed.")

    candidate = str(user_response or "").strip()
    if not re.fullmatch(r"\d{6}", candidate):
        return False
    return bool(pyotp.TOTP(totp_secret).verify(candidate, valid_window=1))


def verify_mfa_challenge(user_response: str, expected_answer: Optional[str] = None) -> bool:
    """
    Verify MFA by requiring the configured passphrase or a valid TOTP code.

    Args:
        user_response: The answer provided by the user.
        expected_answer: Optional explicit answer for compatibility with tests
            and legacy direct callers. Runtime callers use MFA_PASSPHRASE.

    Returns:
        bool: True if the answer matches, False otherwise.
    """
    if not user_response:
        return False

    if _verify_totp_challenge(user_response):
        return True

    expected = expected_answer if expected_answer is not None else get_configured_mfa_passphrase()

    # Normalize response/answer by lowercasing, stripping punctuation,
    # and collapsing whitespace.
    cleaned_response = _normalize_mfa_text(user_response)
    cleaned_expected = _normalize_mfa_text(expected)

    if not cleaned_expected:
        return False

    # Require whole-word match, not substring-in-word.
    pattern = r"\b" + re.escape(cleaned_expected) + r"\b"
    return bool(re.search(pattern, cleaned_response))
