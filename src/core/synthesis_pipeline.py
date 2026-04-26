"""Tool synthesis, self-test, and approval pipeline.

SynthesisPipeline owns the full lifecycle from capability-gap detection
through Admin approval and dynamic tool deployment.
"""
from __future__ import annotations

import asyncio
import hashlib
import logging
import os
import re
import sys
import tempfile
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

from src.core import cloud_redaction
from src.core.llm_router import RouterResult

logger = logging.getLogger(__name__)

_SAFE_ENV_KEYS = {"PATH", "PYTHONPATH", "SYSTEMROOT", "TEMP", "TMP", "HOME", "USER", "LANG", "LC_ALL"}
_BLOCKED_ENV_PREFIXES = ("TELEGRAM_", "GROQ_", "GEMINI_", "ANTHROPIC_", "OPENAI_", "OLLAMA_CLOUD_", "ADMIN_")


# Move this helper verbatim from orchestrator.py:
def _build_safe_subprocess_env() -> Dict[str, str]:
    safe_env: Dict[str, str] = {}
    for key, value in os.environ.items():
        if key in _SAFE_ENV_KEYS:
            safe_env[key] = value
            continue
        if any(key.startswith(prefix) for prefix in _BLOCKED_ENV_PREFIXES):
            continue
        safe_env[key] = value
    # Ensure Python can find packages
    safe_env.setdefault("PYTHONPATH", os.getcwd())
    return safe_env


def _format_voice_placeholder(audio_bytes: bytes, mime_type: str) -> str:
    safe_mime_type = str(mime_type or "audio/ogg").strip() or "audio/ogg"
    return f"[Voice note · {len(audio_bytes)} bytes · {safe_mime_type}]"


_SYNTHESIS_SELF_TEST_TIMEOUT_DEFAULT_SECONDS = 12.0
_MAX_SYNTHESIS_RETRIES = int(os.getenv("MAX_SYNTHESIS_RETRIES", "3"))
_PENDING_STATE_TTL_SECONDS = int(os.getenv("PENDING_STATE_TTL_SECONDS", "86400"))
_SYNTHESIS_LOCKOUT_TTL_SECONDS = int(os.getenv("SYNTHESIS_LOCKOUT_TTL_SECONDS", "600"))


@dataclass
class _ApprovalOutcome:
    reply_text: str
    follow_up_input: Optional[Any] = None


class SynthesisPipeline:
    """Manages the full tool synthesis lifecycle."""

    def __init__(
        self,
        *,
        cognitive_router: Any,
        ledger_memory: Any,
        core_memory: Any,
        pending_tool_approval: Dict[str, Any],   # shared dict reference from Orchestrator
        on_tools_changed: Callable[[], None],     # calls orchestrator._invalidate_capabilities_cache
        process_message_fn: Callable,             # calls orchestrator.process_message
        route_to_system_2_redacted: Callable,     # calls orchestrator._route_to_system_2_redacted
        fire_and_forget: Callable,                # calls orchestrator._fire_and_forget
        notify_admin: Callable,                   # calls orchestrator._notify_admin
        outbound_queue: Optional[asyncio.Queue] = None,
    ) -> None:
        self.cognitive_router = cognitive_router
        self.ledger_memory = ledger_memory
        self.core_memory = core_memory
        self.pending_tool_approval = pending_tool_approval
        self._on_tools_changed = on_tools_changed
        self._process_message_fn = process_message_fn
        self._route_to_system_2_redacted = route_to_system_2_redacted
        self._fire_and_forget = fire_and_forget
        self._notify_admin = notify_admin
        self.outbound_queue = outbound_queue

    @staticmethod
    def _extract_audio_bytes(payload: Dict[str, Any]) -> bytes:
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

    @classmethod
    def _coerce_user_prompt_payload(cls, user_message: Any) -> Dict[str, Any]:
        if isinstance(user_message, dict):
            payload = dict(user_message)
            text = str(
                payload.get("text")
                or payload.get("user_input")
                or payload.get("content")
                or ""
            ).strip()
            audio_bytes = cls._extract_audio_bytes(payload)
            audio_mime_type = str(payload.get("audio_mime_type") or "audio/ogg")
            if not text and audio_bytes:
                text = _format_voice_placeholder(audio_bytes, audio_mime_type)

            normalized: Dict[str, Any] = {
                "text": text,
                "audio_bytes": audio_bytes,
                "audio_mime_type": audio_mime_type,
                "audio_source": str(payload.get("audio_source") or ""),
                "audio_file_id": str(payload.get("audio_file_id") or ""),
            }
            return normalized

        text = str(user_message or "").strip()
        return {
            "text": text,
            "audio_bytes": b"",
            "audio_mime_type": "",
            "audio_source": "",
            "audio_file_id": "",
        }

    @staticmethod
    def _get_synthesis_self_test_timeout_seconds() -> float:
        """Return a bounded timeout for generated-tool self-tests."""
        raw_value = os.getenv(
            "SYNTHESIS_SELF_TEST_TIMEOUT_SECONDS",
            str(_SYNTHESIS_SELF_TEST_TIMEOUT_DEFAULT_SECONDS),
        )
        try:
            timeout_seconds = float(raw_value)
        except ValueError:
            timeout_seconds = _SYNTHESIS_SELF_TEST_TIMEOUT_DEFAULT_SECONDS
        # Keep timeout strict to avoid orchestration hangs from infinite loops.
        return min(max(timeout_seconds, 1.0), 15.0)

    @staticmethod
    def _get_max_syntheses_per_user_per_hour() -> int:
        raw_value = os.getenv("MAX_SYNTHESES_PER_USER_PER_HOUR", "5")
        try:
            budget = int(raw_value)
        except ValueError:
            budget = 5
        return max(1, budget)

    @staticmethod
    def _extract_pytest_counts(output_text: str) -> Dict[str, int]:
        """Extract pass/fail/error counts from pytest textual output."""
        counts = {
            "passed": 0,
            "failed": 0,
            "errors": 0,
            "skipped": 0,
        }
        patterns = {
            "passed": r"(\d+)\s+passed",
            "failed": r"(\d+)\s+failed",
            "errors": r"(\d+)\s+error(?:s)?",
            "skipped": r"(\d+)\s+skipped",
        }
        for key, pattern in patterns.items():
            match = re.search(pattern, output_text)
            if match:
                counts[key] = int(match.group(1))
        return counts

    @staticmethod
    def _write_text_file(path: str, content: str) -> None:
        with open(path, "w", encoding="utf-8", newline="\n") as file_obj:
            file_obj.write(content)

    @staticmethod
    def _compute_synthesis_proof_sha256(tool_code: str, pytest_code: str) -> str:
        """Return SHA-256 digest for the exact tool+test artifact pair."""
        normalized_tool = str(tool_code or "").replace("\r\n", "\n").replace("\r", "\n")
        normalized_pytest = str(pytest_code or "").replace("\r\n", "\n").replace("\r", "\n")
        proof_payload = (
            "SYNTHESIZED_TOOL_CODE\n"
            f"{normalized_tool}\n\n"
            "SYNTHESIZED_PYTEST_CODE\n"
            f"{normalized_pytest}\n"
        )
        return hashlib.sha256(proof_payload.encode("utf-8")).hexdigest()

    @staticmethod
    def _build_synthesis_test_summary(
        attempt_number: int,
        max_retries: int,
        self_test_result: Dict[str, Any],
    ) -> str:
        status = str(self_test_result.get("status") or "failed").upper()
        passed = int(self_test_result.get("passed") or 0)
        failed = int(self_test_result.get("failed") or 0)
        errors = int(self_test_result.get("errors") or 0)
        duration_ms = int(self_test_result.get("duration_ms") or 0)
        return (
            f"{status} on attempt {attempt_number}/{max_retries} "
            f"(passed={passed}, failed={failed}, errors={errors}, duration={duration_ms} ms)"
        )

    @staticmethod
    def _extract_synthesis_failure_trace(self_test_result: Dict[str, Any], max_chars: int = 3000) -> str:
        stderr_text = str(self_test_result.get("stderr") or "").strip()
        stdout_text = str(self_test_result.get("stdout") or "").strip()
        error_text = str(self_test_result.get("error") or "").strip()

        trace_text = "\n\n".join(
            chunk for chunk in [stderr_text, stdout_text, error_text] if chunk
        ).strip()
        if not trace_text:
            return "(no traceback captured)"
        if len(trace_text) > max_chars:
            return trace_text[-max_chars:]
        return trace_text

    @staticmethod
    def _build_synthesis_generation_failure_result(error_message: str) -> Dict[str, Any]:
        return {
            "status": "failed",
            "timed_out": False,
            "timeout_seconds": 0,
            "exit_code": None,
            "duration_ms": 0,
            "passed": 0,
            "failed": 0,
            "errors": 1,
            "skipped": 0,
            "stdout": "",
            "stderr": "",
            "error": str(error_message or "Tool generation failed."),
        }

    async def _create_synthesis_run_if_supported(
        self,
        *,
        user_id: str,
        gap_description: str,
        suggested_tool_name: str,
        original_input: str,
        max_retries: int,
    ) -> Optional[int]:
        creator = getattr(self.ledger_memory, "create_synthesis_run", None)
        if not callable(creator):
            return None
        try:
            return await creator(
                user_id=user_id,
                gap_description=gap_description,
                suggested_tool_name=suggested_tool_name,
                original_input=original_input,
                max_retries=max_retries,
            )
        except Exception as e:
            logger.warning("Could not create synthesis run record: %s", e)
            return None

    async def _append_synthesis_attempt_if_supported(
        self,
        *,
        run_id: Optional[int],
        attempt_number: int,
        phase: str,
        synthesis_payload: Dict[str, Any],
        self_test_result: Dict[str, Any],
        code_sha256: str,
    ) -> None:
        if run_id is None:
            return
        appender = getattr(self.ledger_memory, "append_synthesis_attempt", None)
        if not callable(appender):
            return
        try:
            await appender(
                run_id=run_id,
                attempt_number=attempt_number,
                phase=phase,
                synthesis_payload=synthesis_payload,
                self_test_result=self_test_result,
                code_sha256=code_sha256,
            )
        except Exception as e:
            logger.warning("Could not append synthesis attempt %s for run %s: %s", attempt_number, run_id, e)

    async def _update_synthesis_run_status_if_supported(
        self,
        run_id: Optional[int],
        **kwargs: Any,
    ) -> None:
        if run_id is None:
            return
        updater = getattr(self.ledger_memory, "update_synthesis_run_status", None)
        if not callable(updater):
            return
        try:
            await updater(run_id, **kwargs)
        except Exception as e:
            logger.warning("Could not update synthesis run %s status: %s", run_id, e)

    async def _run_synthesis_self_test(self, synthesis: Dict[str, Any]) -> Dict[str, Any]:
        """Run generated pytest code in an isolated subprocess with a hard timeout."""
        tool_name = synthesis["tool_name"]
        tool_code = synthesis["code"]
        pytest_code = synthesis["pytest_code"]
        timeout_seconds = self._get_synthesis_self_test_timeout_seconds()
        started = time.perf_counter()

        tool_file_name = f"{tool_name}.py"
        test_file_name = f"test_{tool_name}.py"

        process = None
        stdout_text = ""
        stderr_text = ""

        try:
            tmp_dir_obj = tempfile.TemporaryDirectory(prefix="synth_selftest_")
            try:
                temp_dir = tmp_dir_obj.name
                tool_path = os.path.join(temp_dir, tool_file_name)
                test_path = os.path.join(temp_dir, test_file_name)

                await asyncio.to_thread(self._write_text_file, tool_path, tool_code)
                await asyncio.to_thread(self._write_text_file, test_path, pytest_code)

                process = await asyncio.create_subprocess_exec(
                    sys.executable,
                    "-m",
                    "pytest",
                    "-q",
                    test_file_name,
                    cwd=temp_dir,
                    env=_build_safe_subprocess_env(),
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                )

                try:
                    stdout_bytes, stderr_bytes = await asyncio.wait_for(
                        process.communicate(), timeout=timeout_seconds
                    )
                    stdout_text = stdout_bytes.decode(errors="replace")
                    stderr_text = stderr_bytes.decode(errors="replace")
                except asyncio.TimeoutError:
                    if process.returncode is None:
                        process.kill()
                    stdout_bytes, stderr_bytes = await process.communicate()
                    stdout_text = stdout_bytes.decode(errors="replace")
                    stderr_text = stderr_bytes.decode(errors="replace")
                    elapsed_ms = int((time.perf_counter() - started) * 1000)
                    counts = self._extract_pytest_counts(f"{stdout_text}\n{stderr_text}")
                    return {
                        "status": "timeout",
                        "timed_out": True,
                        "timeout_seconds": timeout_seconds,
                        "exit_code": process.returncode,
                        "duration_ms": elapsed_ms,
                        "passed": counts["passed"],
                        "failed": counts["failed"],
                        "errors": counts["errors"],
                        "skipped": counts["skipped"],
                        "stdout": stdout_text,
                        "stderr": stderr_text,
                        "error": (
                            f"Sandboxed self-test timed out after {timeout_seconds:.1f}s "
                            f"(possible infinite loop or blocking operation)."
                        ),
                    }
            finally:
                try:
                    tmp_dir_obj.cleanup()
                except Exception as cleanup_err:
                    logger.debug("Temp dir cleanup failed (non-fatal): %s", cleanup_err)
        except Exception as exc:
            elapsed_ms = int((time.perf_counter() - started) * 1000)
            return {
                "status": "failed",
                "timed_out": False,
                "timeout_seconds": timeout_seconds,
                "exit_code": process.returncode if process is not None else None,
                "duration_ms": elapsed_ms,
                "passed": 0,
                "failed": 0,
                "errors": 1,
                "skipped": 0,
                "stdout": stdout_text,
                "stderr": stderr_text,
                "error": f"Sandboxed self-test harness crashed: {exc}",
            }

        elapsed_ms = int((time.perf_counter() - started) * 1000)
        counts = self._extract_pytest_counts(f"{stdout_text}\n{stderr_text}")
        passed = process.returncode == 0

        return {
            "status": "passed" if passed else "failed",
            "timed_out": False,
            "timeout_seconds": timeout_seconds,
            "exit_code": process.returncode,
            "duration_ms": elapsed_ms,
            "passed": counts["passed"],
            "failed": counts["failed"],
            "errors": counts["errors"],
            "skipped": counts["skipped"],
            "stdout": stdout_text,
            "stderr": stderr_text,
            "error": "" if passed else "Sandboxed self-test reported pytest failures.",
        }

    @staticmethod
    def _build_synthesis_fallback_payload(
        failed_candidate: Optional[Dict[str, Any]],
        suggested_tool_name: str,
    ) -> Dict[str, Any]:
        if isinstance(failed_candidate, dict):
            return dict(failed_candidate)
        return {
            "tool_name": suggested_tool_name,
            "description": "",
            "code": "",
            "pytest_code": "",
            "schema_json": "",
            "test_manifest_json": "",
        }

    async def _request_synthesis_candidate(
        self,
        *,
        attempt_number: int,
        suggested_tool_name: str,
        redacted_gap_description: str,
        redacted_user_query: str,
        failed_candidate: Optional[Dict[str, Any]],
        last_failure_summary: str,
        last_failure_trace: str,
    ) -> Dict[str, Any]:
        if attempt_number == 1:
            return await asyncio.wait_for(
                self.cognitive_router.synthesize_tool(
                    gap_description=redacted_gap_description,
                    suggested_tool_name=suggested_tool_name,
                    user_query=redacted_user_query,
                ),
                timeout=60.0,
            )

        if failed_candidate is None:
            raise RuntimeError("Repair loop missing failed candidate payload.")

        redacted_failure_summary = cloud_redaction.redact_text_for_cloud(
            last_failure_summary,
            allow_sensitive_context=False,
            max_chars=1200,
        )
        redacted_failure_trace = cloud_redaction.redact_text_for_cloud(
            last_failure_trace,
            allow_sensitive_context=False,
            max_chars=2800,
        )

        return await asyncio.wait_for(
            self.cognitive_router.repair_synthesized_tool(
                gap_description=redacted_gap_description,
                suggested_tool_name=suggested_tool_name,
                user_query=redacted_user_query,
                previous_tool_name=str(failed_candidate.get("tool_name") or suggested_tool_name),
                previous_code=str(failed_candidate.get("code") or ""),
                previous_pytest_code=str(failed_candidate.get("pytest_code") or ""),
                failure_summary=redacted_failure_summary,
                failure_trace=redacted_failure_trace,
            ),
            timeout=60.0,
        )

    async def _run_single_synthesis_attempt(
        self,
        *,
        run_id: Optional[int],
        attempt_number: int,
        max_retries: int,
        suggested_tool_name: str,
        redacted_gap_description: str,
        redacted_user_query: str,
        failed_candidate: Optional[Dict[str, Any]],
        last_failure_summary: str,
        last_failure_trace: str,
    ) -> Dict[str, Any]:
        phase = "synthesis" if attempt_number == 1 else "repair"
        try:
            candidate = await self._request_synthesis_candidate(
                attempt_number=attempt_number,
                suggested_tool_name=suggested_tool_name,
                redacted_gap_description=redacted_gap_description,
                redacted_user_query=redacted_user_query,
                failed_candidate=failed_candidate,
                last_failure_summary=last_failure_summary,
                last_failure_trace=last_failure_trace,
            )
        except Exception as e:
            error_message = (
                f"{phase.capitalize()} generation failed on attempt {attempt_number}/{max_retries}: {e}"
            )
            logger.warning(error_message)
            failure_result = self._build_synthesis_generation_failure_result(error_message)
            await self._append_synthesis_attempt_if_supported(
                run_id=run_id,
                attempt_number=attempt_number,
                phase=phase,
                synthesis_payload=self._build_synthesis_fallback_payload(
                    failed_candidate,
                    suggested_tool_name,
                ),
                self_test_result=failure_result,
                code_sha256="",
            )
            return {
                "status": "generation_failed",
                "candidate": failed_candidate,
                "test_result": failure_result,
                "failure_summary": error_message,
                "failure_trace": error_message,
            }

        test_result = await self._run_synthesis_self_test(candidate)
        proof_sha256 = self._compute_synthesis_proof_sha256(
            candidate.get("code", ""),
            candidate.get("pytest_code", ""),
        )
        test_summary = self._build_synthesis_test_summary(
            attempt_number,
            max_retries,
            test_result,
        )
        candidate["self_test_result"] = test_result
        candidate["self_test_summary"] = test_summary
        candidate["synthesis_proof_sha256"] = proof_sha256

        await self._append_synthesis_attempt_if_supported(
            run_id=run_id,
            attempt_number=attempt_number,
            phase=phase,
            synthesis_payload=candidate,
            self_test_result=test_result,
            code_sha256=proof_sha256,
        )

        if str(test_result.get("status") or "") == "passed":
            return {
                "status": "passed",
                "candidate": candidate,
                "test_result": test_result,
                "proof_sha256": proof_sha256,
                "test_summary": test_summary,
            }

        return {
            "status": "failed",
            "candidate": candidate,
            "test_result": test_result,
            "failure_summary": f"{test_summary}. {str(test_result.get('error') or '').strip()}".strip(),
            "failure_trace": self._extract_synthesis_failure_trace(test_result),
        }

    async def _build_blocked_synthesis_result(
        self,
        *,
        run_id: Optional[int],
        attempts_used: int,
        max_retries: int,
        failed_candidate: Optional[Dict[str, Any]],
        last_test_result: Dict[str, Any],
    ) -> Dict[str, Any]:
        trace_excerpt = self._extract_synthesis_failure_trace(last_test_result)
        blocked_reason = (
            f"Synthesis run blocked after {attempts_used}/{max_retries} attempts; no passing candidate produced."
        )
        blocked_message = (
            "System 2 synthesis run was blocked after bounded repair attempts. "
            "The main orchestrator remains active.\n\n"
            f"Attempts: {attempts_used}/{max_retries}\n"
            f"Last status: {str(last_test_result.get('status') or 'failed')}\n"
            f"Details: {str(last_test_result.get('error') or blocked_reason)}\n\n"
            f"Trace excerpt:\n{trace_excerpt}"
        )
        await self._update_synthesis_run_status_if_supported(
            run_id,
            status="blocked",
            total_attempts=attempts_used,
            blocked_reason=blocked_reason,
            synthesis_payload=failed_candidate,
        )
        return {
            "status": "blocked",
            "run_id": run_id,
            "attempts_used": attempts_used,
            "max_retries": max_retries,
            "synthesis": failed_candidate,
            "self_test_result": last_test_result,
            "blocked_reason": blocked_reason,
            "blocked_message": blocked_message,
        }

    async def _execute_synthesis_repair_loop(
        self,
        *,
        user_id: str,
        gap_description: str,
        suggested_tool_name: str,
        user_input: str,
    ) -> Dict[str, Any]:
        """Run synthesis + bounded repair retries until tests pass or run is blocked."""
        max_retries = max(1, int(_MAX_SYNTHESIS_RETRIES))
        run_id = await self._create_synthesis_run_if_supported(
            user_id=user_id,
            gap_description=gap_description,
            suggested_tool_name=suggested_tool_name,
            original_input=user_input,
            max_retries=max_retries,
        )

        redacted_gap_description = cloud_redaction.redact_text_for_cloud(gap_description)
        redacted_user_query = cloud_redaction.redact_text_for_cloud(user_input)

        failed_candidate: Optional[Dict[str, Any]] = None
        last_test_result: Dict[str, Any] = {}
        last_failure_summary = ""
        last_failure_trace = ""
        attempts_used = 0

        for attempt_number in range(1, max_retries + 1):
            attempts_used = attempt_number
            attempt_result = await self._run_single_synthesis_attempt(
                run_id=run_id,
                attempt_number=attempt_number,
                max_retries=max_retries,
                suggested_tool_name=suggested_tool_name,
                redacted_gap_description=redacted_gap_description,
                redacted_user_query=redacted_user_query,
                failed_candidate=failed_candidate,
                last_failure_summary=last_failure_summary,
                last_failure_trace=last_failure_trace,
            )

            status = str(attempt_result.get("status") or "failed")
            if status == "passed":
                return {
                    "status": "passed",
                    "run_id": run_id,
                    "attempts_used": attempt_number,
                    "max_retries": max_retries,
                    "proof_sha256": str(attempt_result.get("proof_sha256") or ""),
                    "test_summary": str(attempt_result.get("test_summary") or ""),
                    "synthesis": dict(attempt_result.get("candidate") or {}),
                    "self_test_result": dict(attempt_result.get("test_result") or {}),
                }

            failed_candidate = dict(attempt_result.get("candidate") or failed_candidate or {})
            last_test_result = dict(attempt_result.get("test_result") or {})
            last_failure_summary = str(attempt_result.get("failure_summary") or "")
            last_failure_trace = str(attempt_result.get("failure_trace") or "")
            if status == "generation_failed":
                break

        return await self._build_blocked_synthesis_result(
            run_id=run_id,
            attempts_used=attempts_used,
            max_retries=max_retries,
            failed_candidate=failed_candidate,
            last_test_result=last_test_result,
        )

    async def _count_prior_failed_synthesis_runs(self, suggested_tool_name: str) -> tuple[int, int]:
        exact_counter = getattr(self.ledger_memory, "count_synthesis_runs_by_tool_status", None)
        fuzzy_counter = getattr(self.ledger_memory, "count_synthesis_failures_fuzzy", None)

        exact_count = 0
        if callable(exact_counter):
            try:
                exact_count = int(
                    await exact_counter(
                        suggested_tool_name=str(suggested_tool_name or ""),
                        statuses=["blocked", "rejected"],
                    )
                )
            except Exception as exc:
                logger.warning("Could not query exact prior synthesis failures for %s: %s", suggested_tool_name, exc)

        fuzzy_count = 0
        if callable(fuzzy_counter):
            try:
                fuzzy_count = int(
                    await fuzzy_counter(
                        str(suggested_tool_name or ""),
                        threshold=0.75,
                        window_hours=24,
                    )
                )
            except Exception as exc:
                logger.warning("Could not query fuzzy prior synthesis failures for %s: %s", suggested_tool_name, exc)

        return max(0, exact_count), max(0, fuzzy_count)

    async def _count_recent_synthesis_runs_for_user(self, user_id: str, *, window_hours: int = 1) -> int:
        counter = getattr(self.ledger_memory, "count_synthesis_runs_for_user_window", None)
        if not callable(counter):
            return 0
        try:
            return int(
                await counter(
                    user_id=str(user_id or ""),
                    window_hours=max(1, int(window_hours or 1)),
                )
            )
        except Exception as exc:
            logger.warning("Could not query per-user synthesis budget usage for %s: %s", user_id, exc)
            return 0

    async def tool_synthesis_node(
        self,
        state: Dict[str, Any],
        router_result: RouterResult,
    ) -> Any:
        """
        Cognitive handoff: router flagged a gap -> System 2 synthesises a tool
        -> Admin approves via HITL -> tool registered -> original query retried.

        Returns a HITL prompt string, or an outbound payload dict when the code
        must be attached as a document. The orchestrator stores the synthesis
        payload in pending_tool_approval until the Admin replies.
        """
        user_id = state["user_id"]
        gap_description = router_result.gap_description
        suggested_tool_name = router_result.suggested_tool_name
        logger.info(f"Tool synthesis triggered: gap='{gap_description}'")

        if not self.cognitive_router.get_system_2_available():
            return (
                "System 1 identified a capability gap but System 2 is offline - "
                "cannot synthesise a new tool right now. Please configure GROQ_API_KEY."
            )

        hourly_budget = self._get_max_syntheses_per_user_per_hour()
        recent_runs = await self._count_recent_synthesis_runs_for_user(user_id, window_hours=1)
        if recent_runs >= hourly_budget:
            return (
                "HITL REQUIRED: Per-user synthesis budget exhausted "
                f"({hourly_budget} per hour). Manual intervention needed."
            )

        exact_failed_runs, fuzzy_failed_runs = await self._count_prior_failed_synthesis_runs(suggested_tool_name)
        prior_failed_runs = max(exact_failed_runs, fuzzy_failed_runs)
        if prior_failed_runs >= 2:
            return (
                "HITL REQUIRED: This capability has been attempted "
                f"{prior_failed_runs} times without success. Manual intervention needed."
            )

        loop_result = await self._execute_synthesis_repair_loop(
            user_id=user_id,
            gap_description=gap_description,
            suggested_tool_name=suggested_tool_name,
            user_input=str(state.get("user_input") or ""),
        )
        if loop_result.get("status") != "passed":
            return str(loop_result.get("blocked_message") or "Tool synthesis run blocked.")

        synthesis = dict(loop_result["synthesis"])
        proof_sha256 = str(loop_result.get("proof_sha256") or "")
        attempts_used = int(loop_result.get("attempts_used") or 1)
        max_retries = int(loop_result.get("max_retries") or 1)
        test_summary = str(loop_result.get("test_summary") or "")
        synthesis_run_id = loop_result.get("run_id")

        synthesis["synthesis_run_id"] = synthesis_run_id
        synthesis["synthesis_attempts_used"] = attempts_used
        synthesis["synthesis_max_retries"] = max_retries
        synthesis["synthesis_proof_sha256"] = proof_sha256
        synthesis["self_test_summary"] = test_summary

        await self._update_synthesis_run_status_if_supported(
            synthesis_run_id,
            status="pending_approval",
            total_attempts=attempts_used,
            successful_attempt=attempts_used,
            final_tool_name=str(synthesis.get("tool_name") or ""),
            code_sha256=proof_sha256,
            test_summary=test_summary,
            synthesis_payload=synthesis,
        )

        # Store payload for approval resumption (in-memory + persisted to DB)
        self.pending_tool_approval[user_id] = {
            "synthesis": synthesis,
            "original_state": state,
            "synthesis_run_id": synthesis_run_id,
            "_created_at": time.time(),
        }
        await self.ledger_memory.save_pending_approval(
            user_id, synthesis, state["user_input"]
        )

        # Build Admin-facing HITL prompt with the proposed code
        hitl_msg = (
            f"🔧 TOOL SYNTHESIS REQUEST\n\n"
            f"System 1 could not answer: \"{state['user_input']}\"\n"
            f"Gap identified: {gap_description}\n\n"
            f"System 2 has drafted the following tool:\n"
            f"─────────────────────────────\n"
            f"Name: {synthesis['tool_name']}\n"
            f"Description: {synthesis['description']}\n\n"
            f"Sandboxed self-test: {test_summary}\n"
            f"Cryptographic proof (SHA-256 tool+tests): {proof_sha256}\n"
            f"Audit run id: {synthesis_run_id if synthesis_run_id is not None else 'n/a'}\n\n"
            f"Code:\n```python\n{synthesis['code']}\n```\n"
            f"─────────────────────────────\n"
            f"Reply YES to approve and deploy, or NO to reject."
        )
        if len(hitl_msg) > 3500:
            attachment_path = self._write_synthesis_code_attachment(synthesis)
            attachment_name = os.path.basename(attachment_path)
            metadata_msg = (
                "TOOL SYNTHESIS REQUEST\n\n"
                f"System 1 could not answer: \"{state['user_input']}\"\n"
                f"Gap identified: {gap_description}\n\n"
                f"Name: {synthesis['tool_name']}\n"
                f"Description: {synthesis['description']}\n\n"
                f"Sandboxed self-test: {test_summary}\n"
                f"Cryptographic proof (SHA-256 tool+tests): {proof_sha256}\n"
                f"Audit run id: {synthesis_run_id if synthesis_run_id is not None else 'n/a'}\n"
                f"Repair attempt: {attempts_used}/{max_retries}\n\n"
                f"Code attached as Telegram document: {attachment_name}\n"
                "Reply YES to approve and deploy, or NO to reject."
            )
            return {
                "text": metadata_msg[:3500],
                "document_path": attachment_path,
                "document_filename": attachment_name,
                "delete_after_send": True,
            }
        return hitl_msg

    @staticmethod
    def _write_synthesis_code_attachment(synthesis: Dict[str, Any]) -> str:
        tool_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(synthesis.get("tool_name") or "tool"))
        with tempfile.NamedTemporaryFile(
            "w",
            encoding="utf-8",
            suffix=f"_{tool_name}.py",
            prefix="aiden_synth_",
            delete=False,
        ) as temp_file:
            temp_file.write(str(synthesis.get("code") or ""))
            temp_file.write("\n")
            return temp_file.name

    async def _pop_pending_tool_approval_payload(self, user_id: str) -> Optional[Dict[str, Any]]:
        if user_id not in self.pending_tool_approval:
            return None
        payload = self.pending_tool_approval[user_id]
        age = time.time() - payload.get("_created_at", 0)
        if age > _PENDING_STATE_TTL_SECONDS:
            self.pending_tool_approval.pop(user_id, None)
            await self.ledger_memory.clear_pending_approval(user_id)
            logger.info("Expired stale pending_tool_approval for %s (age %.0fs).", user_id, age)
            return None
        self.pending_tool_approval.pop(user_id, None)
        await self.ledger_memory.clear_pending_approval(user_id)
        return payload

    async def _reject_synthesized_tool(
        self,
        *,
        user_id: str,
        synthesis_run_id: Optional[int],
        attempts_used: int,
        synthesis: Dict[str, Any],
    ) -> str:
        logger.info("Admin rejected tool synthesis for user %s", user_id)
        await self._update_synthesis_run_status_if_supported(
            synthesis_run_id,
            status="rejected",
            total_attempts=attempts_used if attempts_used > 0 else None,
            final_tool_name=str(synthesis.get("tool_name") or ""),
            code_sha256=str(synthesis.get("synthesis_proof_sha256") or ""),
            test_summary=str(synthesis.get("self_test_summary") or ""),
            blocked_reason="Rejected by admin during HITL approval.",
            synthesis_payload=synthesis,
        )
        return f"Tool proposal rejected. The capability gap remains: {synthesis['description']}"

    async def _verify_synthesis_payload_digest(
        self,
        *,
        synthesis: Dict[str, Any],
        synthesis_run_id: Optional[int],
        attempts_used: int,
        tool_name: str,
    ) -> Optional[str]:
        expected_digest = str(synthesis.get("synthesis_proof_sha256") or "").strip()
        if not expected_digest:
            return None

        actual_digest = self._compute_synthesis_proof_sha256(
            str(synthesis.get("code") or ""),
            str(synthesis.get("pytest_code") or ""),
        )
        if actual_digest == expected_digest:
            return None

        mismatch_reason = (
            "Tool approval blocked: synthesis proof digest mismatch between sandbox-passed "
            "artifact and pending deployment payload."
        )
        await self._update_synthesis_run_status_if_supported(
            synthesis_run_id,
            status="blocked",
            total_attempts=attempts_used if attempts_used > 0 else None,
            final_tool_name=str(tool_name),
            code_sha256=actual_digest,
            test_summary=str(synthesis.get("self_test_summary") or ""),
            blocked_reason=mismatch_reason,
            synthesis_payload=synthesis,
        )
        logger.error(mismatch_reason)
        return mismatch_reason

    async def _deploy_approved_synthesized_tool(
        self,
        *,
        tool_name: str,
        synthesis: Dict[str, Any],
        synthesis_run_id: Optional[int],
        attempts_used: int,
        original_state: Dict[str, Any],
    ) -> _ApprovalOutcome:
        await self.cognitive_router.register_dynamic_tool(tool_name, synthesis["code"], synthesis["schema_json"])
        self._on_tools_changed()
        await self.ledger_memory.register_tool(
            name=tool_name,
            description=synthesis["description"],
            code=synthesis["code"],
            schema_json=synthesis["schema_json"],
        )
        await self.ledger_memory.approve_tool(tool_name)
        await self._update_synthesis_run_status_if_supported(
            synthesis_run_id,
            status="approved",
            total_attempts=attempts_used if attempts_used > 0 else None,
            successful_attempt=attempts_used if attempts_used > 0 else None,
            final_tool_name=str(tool_name),
            code_sha256=str(synthesis.get("synthesis_proof_sha256") or ""),
            test_summary=str(synthesis.get("self_test_summary") or ""),
            synthesis_payload=synthesis,
        )
        core = await self.core_memory.get_all()
        caps = core.get("known_capabilities", "")
        await self.core_memory.update("known_capabilities", f"{caps}, {tool_name}".lstrip(", "))
        logger.info("Tool '%s' approved, registered, and logged to core memory", tool_name)

        retry_payload = self._coerce_user_prompt_payload(
            original_state.get("user_prompt", original_state.get("user_input", ""))
        )
        retry_message = str(retry_payload.get("text") or "").strip()
        if not retry_message and bool(self._extract_audio_bytes(retry_payload)):
            retry_message = _format_voice_placeholder(
                self._extract_audio_bytes(retry_payload),
                str(retry_payload.get("audio_mime_type") or "audio/ogg"),
            )
            retry_payload["text"] = retry_message
        elif not retry_message:
            retry_message = str(original_state.get("user_input") or "").strip()
            retry_payload["text"] = retry_message

        return _ApprovalOutcome(
            reply_text=f"✅ Tool '{tool_name}' deployed.",
            follow_up_input=retry_payload,
        )

    async def _handle_synthesized_tool_deploy_failure(
        self,
        *,
        tool_name: str,
        synthesis: Dict[str, Any],
        synthesis_run_id: Optional[int],
        attempts_used: int,
        error: Exception,
    ) -> str:
        await self._update_synthesis_run_status_if_supported(
            synthesis_run_id,
            status="blocked",
            total_attempts=attempts_used if attempts_used > 0 else None,
            final_tool_name=str(tool_name),
            code_sha256=str(synthesis.get("synthesis_proof_sha256") or ""),
            test_summary=str(synthesis.get("self_test_summary") or ""),
            blocked_reason=f"Deployment failed after approval: {error}",
            synthesis_payload=synthesis,
        )
        logger.error("Tool registration failed: %s", error, exc_info=True)
        return f"Error deploying tool '{tool_name}': {error}"

    async def try_resume_tool_approval(self, user_id: str, user_message: str) -> Optional[_ApprovalOutcome]:
        """Handle YES/NO tool synthesis approval. Returns approval outcome, or None if not pending."""
        payload = await self._pop_pending_tool_approval_payload(user_id)
        if payload is None:
            return None

        synthesis = payload["synthesis"]
        original_state = payload["original_state"]
        synthesis_run_id = payload.get("synthesis_run_id") or synthesis.get("synthesis_run_id")
        attempts_used = int(synthesis.get("synthesis_attempts_used") or 0)
        tool_name = str(synthesis.get("tool_name") or "")

        if not user_message.strip().upper().startswith("YES"):
            return _ApprovalOutcome(
                reply_text=await self._reject_synthesized_tool(
                    user_id=user_id,
                    synthesis_run_id=synthesis_run_id,
                    attempts_used=attempts_used,
                    synthesis=synthesis,
                )
            )

        mismatch = await self._verify_synthesis_payload_digest(
            synthesis=synthesis,
            synthesis_run_id=synthesis_run_id,
            attempts_used=attempts_used,
            tool_name=tool_name,
        )
        if mismatch:
            return _ApprovalOutcome(reply_text=mismatch)

        try:
            return await self._deploy_approved_synthesized_tool(
                tool_name=tool_name,
                synthesis=synthesis,
                synthesis_run_id=synthesis_run_id,
                attempts_used=attempts_used,
                original_state=original_state,
            )
        except Exception as e:
            return _ApprovalOutcome(
                reply_text=await self._handle_synthesized_tool_deploy_failure(
                    tool_name=tool_name,
                    synthesis=synthesis,
                    synthesis_run_id=synthesis_run_id,
                    attempts_used=attempts_used,
                    error=e,
                )
            )

    async def async_tool_synthesis(
        self,
        user_id: str,
        result: RouterResult,
        state: Dict[str, Any],
    ) -> None:
        """Background task: run tool synthesis and send the HITL prompt to admin."""
        try:
            async def _run_synthesis() -> None:
                hitl_prompt = await self.tool_synthesis_node(state, result)
                await self._notify_admin(hitl_prompt)

            await asyncio.wait_for(
                _run_synthesis(),
                timeout=float(_SYNTHESIS_LOCKOUT_TTL_SECONDS),
            )
        except asyncio.TimeoutError:
            logger.critical(
                "Background tool synthesis timed out for user %s after %ss.",
                user_id,
                _SYNTHESIS_LOCKOUT_TTL_SECONDS,
            )
            try:
                await self._notify_admin(
                    "CRITICAL: Tool synthesis timed out and was aborted. "
                    f"User: {user_id}. Timeout: {_SYNTHESIS_LOCKOUT_TTL_SECONDS}s."
                )
            except Exception as notify_error:
                logger.error("Failed to notify admin about synthesis timeout: %s", notify_error)
        except Exception as e:
            logger.error(f"Background tool synthesis failed: {e}", exc_info=True)
        finally:
            in_progress = getattr(self, "_synthesis_in_progress", None)
            if isinstance(in_progress, dict):
                in_progress.pop(user_id, None)
