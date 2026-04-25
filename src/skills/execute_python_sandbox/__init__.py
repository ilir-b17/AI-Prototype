import asyncio
import json
import logging
import os
from typing import Optional

from src.core.dynamic_tool_worker import DynamicToolWorkerClient
from src.core.runtime_context import get_orchestrator

logger = logging.getLogger(__name__)


def _execution_timeout_seconds() -> float:
    try:
        return float(os.getenv("TOOL_EXEC_TIMEOUT_SECONDS", "10.0"))
    except ValueError:
        return 10.0


def _get_existing_worker() -> Optional[DynamicToolWorkerClient]:
    orchestrator = get_orchestrator()
    router = getattr(orchestrator, "cognitive_router", None) if orchestrator is not None else None
    worker = getattr(router, "_dynamic_tool_worker", None)
    if isinstance(worker, DynamicToolWorkerClient):
        return worker
    return None


async def execute_python_sandbox(code_string: str) -> str:
    """
    Execute Python through the isolated dynamic worker builtin tool.

    The execution path inherits dynamic worker guardrails (AST validation,
    blocked imports/modules, POSIX resource limits, and worker-local /tmp).
    """
    logger.info("execute_python_sandbox called.")

    if not isinstance(code_string, str) or not code_string.strip():
        return json.dumps(
            {
                "status": "error",
                "message": "Invalid code_string",
                "details": "The code_string must be a non-empty Python script.",
            }
        )

    timeout_seconds = _execution_timeout_seconds()
    worker = _get_existing_worker()
    created_worker = False

    if worker is None:
        worker = DynamicToolWorkerClient(
            call_timeout_seconds=timeout_seconds,
            register_timeout_seconds=min(10.0, max(1.0, timeout_seconds)),
        )
        created_worker = True

    try:
        response = await worker.call_tool(
            "__exec_user_python__",
            {"code_string": code_string},
        )
        if not response.get("ok"):
            return json.dumps(
                {
                    "status": "error",
                    "message": "Sandbox execution blocked",
                    "details": str(response.get("error") or "Unknown worker error"),
                }
            )

        result = response.get("result")
        return str(result) if result else "Script executed successfully with no output."
    except asyncio.CancelledError:
        raise
    except Exception as exc:
        logger.error("execute_python_sandbox error: %s", exc, exc_info=True)
        return json.dumps(
            {
                "status": "error",
                "message": "Sandbox execution blocked",
                "details": str(exc),
            }
        )
    finally:
        if created_worker:
            try:
                await worker.shutdown()
            except Exception:
                logger.debug("Temporary sandbox worker shutdown failed", exc_info=True)
