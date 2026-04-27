import asyncio
import json
import logging
import os
import shlex
from typing import List

logger = logging.getLogger(__name__)

DEFAULT_ALLOWLIST = "git status,ls,dir,wc,echo,cat,type,pwd,hostname"
SHELL_METACHARACTERS = set(";&|<>`$")


def _get_timeout_seconds() -> float:
    try:
        return float(os.getenv("TOOL_EXEC_TIMEOUT_SECONDS", "10.0"))
    except ValueError:
        return 10.0


def _security_error(details: str) -> str:
    return json.dumps(
        {
            "status": "error",
            "message": "Command blocked by security policy",
            "details": details,
        }
    )


def _contains_shell_metacharacters_outside_quotes(command: str) -> bool:
    in_single_quote = False
    in_double_quote = False
    escaped = False

    for char in command:
        if escaped:
            escaped = False
            continue

        if char == "\\" and not in_single_quote:
            escaped = True
            continue
        if char == "'" and not in_double_quote:
            in_single_quote = not in_single_quote
            continue
        if char == '"' and not in_single_quote:
            in_double_quote = not in_double_quote
            continue
        if not in_single_quote and not in_double_quote and char in SHELL_METACHARACTERS:
            return True
    return False


def _split_command(command: str) -> List[str]:
    return shlex.split(command, posix=(os.name != "nt"))


def _allowlisted_prefixes() -> List[List[str]]:
    raw_allowlist = os.getenv("AIDEN_TERMINAL_ALLOWLIST", DEFAULT_ALLOWLIST)
    allowed_prefixes: List[List[str]] = []
    for entry in str(raw_allowlist).split(","):
        normalized = entry.strip()
        if not normalized:
            continue
        try:
            tokens = _split_command(normalized)
        except ValueError:
            tokens = normalized.split()
        prefix = [token.strip().lower() for token in tokens if token.strip()]
        if prefix:
            allowed_prefixes.append(prefix)
    return allowed_prefixes


def _is_allowlisted_command(argv: List[str]) -> bool:
    lowered = [str(token).strip().lower() for token in argv]
    return any(lowered[:len(prefix)] == prefix for prefix in _allowlisted_prefixes())

async def run_terminal_command(command: str) -> str:
    """
    Execute allowlisted terminal commands without invoking a shell.

    Security contract:
    - Reject shell metacharacters outside quoted strings.
    - Parse into argv and execute with create_subprocess_exec (no shell).
    - Enforce first-token allowlist from AIDEN_TERMINAL_ALLOWLIST.
    """
    logger.info(f"run_terminal_command called with command: {command}")

    if not isinstance(command, str) or not command.strip():
        return json.dumps({
            "status": "error",
            "message": "Invalid command",
            "details": "The command must be a non-empty string."
        })

    if _contains_shell_metacharacters_outside_quotes(command):
        return _security_error(
            "Shell metacharacters are not allowed. Submit a direct command with plain arguments only."
        )

    try:
        argv = _split_command(command)
    except ValueError as exc:
        return _security_error(f"Command parsing failed: {exc}")

    if not argv:
        return _security_error("Command is empty after parsing.")

    if not _is_allowlisted_command(argv):
        return _security_error(
            f"Command '{' '.join(argv[:2])}' is not in the allowlist."
        )

    timeout_seconds = _get_timeout_seconds()

    try:
        process = await asyncio.create_subprocess_exec(
            *argv,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

        try:
            stdout_bytes, stderr_bytes = await asyncio.wait_for(process.communicate(), timeout=timeout_seconds)
        except asyncio.TimeoutError:
            try:
                process.kill()
                await process.wait()
            except ProcessLookupError:
                pass
            return json.dumps({
                "status": "error",
                "message": f"Timeout exceeded ({timeout_seconds}s)",
                "details": "The command took too long to complete and was terminated. Check if it requires user input or runs an infinite loop."
            })

        stdout = stdout_bytes.decode('utf-8', errors='replace') if stdout_bytes else ""
        stderr = stderr_bytes.decode('utf-8', errors='replace') if stderr_bytes else ""

        output = ""
        if stdout:
            output += f"--- STDOUT ---\n{stdout}\n"
        if stderr:
            output += f"--- STDERR ---\n{stderr}\n"

        if not output:
            output = "Command executed successfully with no output."

        return output

    except Exception as e:
        logger.error(f"run_terminal_command error: {str(e)}", exc_info=True)
        return json.dumps({
            "status": "error",
            "message": f"Failed to execute command: {str(e)}",
            "details": "Check syntax or verify if the shell environment supports this command."
        })
