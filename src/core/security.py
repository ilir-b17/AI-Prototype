"""
Security Module for Core Agent Operations.

This module provides security checks, including Multi-Factor Authentication (MFA)
using passphrase challenges, to authorize sensitive operations like modifying
the charter or altering core memory.
"""

import re

def verify_mfa_challenge(user_response: str, expected_answer: str = "blue") -> bool:
    """
    Verify MFA by requiring the expected answer as a standalone word.

    Args:
        user_response: The answer provided by the user.
        expected_answer: The expected answer to authorize the request.

    Returns:
        bool: True if the answer matches, False otherwise.
    """
    if not user_response:
        return False

    # Normalize response/answer by lowercasing, stripping punctuation,
    # and collapsing whitespace.
    cleaned_response = re.sub(
        r"\s+",
        " ",
        re.sub(r"[^\w\s]", "", user_response.lower()),
    ).strip()
    cleaned_expected = re.sub(
        r"\s+",
        " ",
        re.sub(r"[^\w\s]", "", expected_answer.lower()),
    ).strip()

    if not cleaned_expected:
        return False

    # Require whole-word match, not substring-in-word.
    pattern = r"\b" + re.escape(cleaned_expected) + r"\b"
    return bool(re.search(pattern, cleaned_response))

# verify_mfa_challenge("The sky is blue")  -> True
# verify_mfa_challenge("blueprint")        -> False
# verify_mfa_challenge("blueberry jam")    -> False
# verify_mfa_challenge("BLUE!")            -> True
# verify_mfa_challenge("")                 -> False
