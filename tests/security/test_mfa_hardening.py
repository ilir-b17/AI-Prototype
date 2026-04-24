from typing import Coroutine, Any
from unittest.mock import AsyncMock

import pytest

from src.core.llm_router import RouterResult
from src.core.orchestrator import Orchestrator
from src.core.security import verify_mfa_challenge


def _policy_test_orchestrator() -> Orchestrator:
    orchestrator = Orchestrator.__new__(Orchestrator)
    orchestrator.charter_text = "Full charter for MFA configuration tests."
    return orchestrator


def _close_coroutine(coro: Coroutine[Any, Any, Any]) -> None:
    coro.close()


def test_enforce_charter_policy_requires_mfa_passphrase(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("MFA_PASSPHRASE", raising=False)
    monkeypatch.delenv("MFA_TOTP_SECRET", raising=False)
    orchestrator = _policy_test_orchestrator()

    with pytest.raises(RuntimeError, match="MFA_PASSPHRASE"):
        Orchestrator._enforce_charter_policy(orchestrator)


def test_enforce_charter_policy_rejects_short_mfa_passphrase(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MFA_PASSPHRASE", "short")
    monkeypatch.delenv("MFA_TOTP_SECRET", raising=False)
    orchestrator = _policy_test_orchestrator()

    with pytest.raises(RuntimeError, match="at least 12 characters"):
        Orchestrator._enforce_charter_policy(orchestrator)


@pytest.mark.parametrize("common_word", ["blue", "password", "admin", "secret", "aiden", "test"])
def test_enforce_charter_policy_rejects_common_mfa_words(
    monkeypatch: pytest.MonkeyPatch,
    common_word: str,
) -> None:
    monkeypatch.setenv("MFA_PASSPHRASE", f"{common_word} authorization phrase")
    monkeypatch.delenv("MFA_TOTP_SECRET", raising=False)
    orchestrator = _policy_test_orchestrator()

    with pytest.raises(RuntimeError, match="common authorization word"):
        Orchestrator._enforce_charter_policy(orchestrator)


def test_enforce_charter_policy_accepts_strong_mfa_passphrase(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MFA_PASSPHRASE", "correct horse battery staple")
    monkeypatch.delenv("MFA_TOTP_SECRET", raising=False)
    orchestrator = _policy_test_orchestrator()

    Orchestrator._enforce_charter_policy(orchestrator)


def test_verify_mfa_challenge_uses_env_passphrase(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MFA_PASSPHRASE", "correct horse battery staple")
    monkeypatch.delenv("MFA_TOTP_SECRET", raising=False)

    assert verify_mfa_challenge("please authorize with correct horse battery staple") is True
    assert verify_mfa_challenge("correct horse") is False


def test_verify_mfa_challenge_accepts_totp_code(monkeypatch: pytest.MonkeyPatch) -> None:
    pyotp = pytest.importorskip("pyotp")
    totp_seed = pyotp.random_base32()
    monkeypatch.setenv("MFA_PASSPHRASE", "correct horse battery staple")
    monkeypatch.setenv("MFA_TOTP_SECRET", totp_seed)

    assert verify_mfa_challenge(pyotp.TOTP(totp_seed).now()) is True


@pytest.mark.asyncio
async def test_handle_blocked_result_mfa_prompt_is_generic() -> None:
    orchestrator = Orchestrator.__new__(Orchestrator)
    orchestrator.pending_mfa = {}
    orchestrator.ledger_memory = AsyncMock()
    orchestrator._fire_and_forget = _close_coroutine

    result = RouterResult(
        status="mfa_required",
        mfa_tool_name="request_core_update",
        mfa_arguments={"key": "value"},
    )

    message = await Orchestrator._handle_blocked_result(orchestrator, result, "user-1", {})

    assert message == "SECURITY LOCK: Provide the authorization passphrase to continue."
    assert "sky" not in message.lower()
    assert "blue" not in message.lower()
    assert orchestrator.pending_mfa["user-1"]["name"] == "request_core_update"
