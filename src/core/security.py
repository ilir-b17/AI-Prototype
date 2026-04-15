"""
Security Module for Core Agent Operations.

This module provides security checks, including Multi-Factor Authentication (MFA)
using passphrase challenges, to authorize sensitive operations like modifying
the charter or altering core memory.
"""

def verify_mfa_challenge(user_response: str, expected_answer: str = "blue") -> bool:
    """
    Verifies the user's response to an MFA challenge.

    Args:
        user_response: The answer provided by the user.
        expected_answer: The expected answer to authorize the request.

    Returns:
        bool: True if the answer matches, False otherwise.
    """
    if not user_response:
        return False

    # Simple case-insensitive exact match or substring match, but let's be lenient
    # to typical input variations like "Blue.", "blue!", etc.
    cleaned_response = "".join(c for c in user_response if c.isalnum()).lower()
    cleaned_expected = "".join(c for c in expected_answer if c.isalnum()).lower()

    return cleaned_expected in cleaned_response
