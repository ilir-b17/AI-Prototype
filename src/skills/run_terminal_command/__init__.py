import asyncio
import json
import logging
import os
import shlex
from typing import List, Optional

from src.skills._common.path_guard import get_default_allowed_roots, resolve_confined_path

logger = logging.getLogger(__name__)

# 'cat' and 'type' are removed from the default allowlist because they can read
# arbitrary files (SSH keys, /etc/passwd, etc.).  File reading is handled by the
# manage_file_system skill which applies path_guard confinement.
DEFAULT_ALLOWLIST = "git status,ls,dir,wc,echo,pwd,hostname"
SHELL_METACHARACTERS = set(";&|<>`$")

# Commands that accept a filesystem path argument must have that path confined to
# the configured allowed roots so they cannot list/stat sensitive directories.
_PATH_ARGUMENT_COMMANDS: frozenset = frozenset({"ls", "dir", "wc"})


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


def _check_path_argument(argv: List[str]) -> Optional[str]:
    """
    For commands that accept a filesystem path argument (ls, dir, wc), verify
    that every non-flag argument resolves within the configured allowed roots.
    Returns an error string on a path violation, or None if the arguments are safe.
    """
    cmd = argv[0].lower() if argv else ""
    if cmd not in _PATH_ARGUMENT_COMMANDS:
        return None
    path_args = [arg for arg in argv[1:] if not arg.startswith("-")]
    if not path_args:
        # No explicit path — command operates on CWD (project root), which is safe.
        return None
    allowed_roots = get_default_allowed_roots()
    for path_arg in path_args:
        try:
            resolve_confined_path(path_arg, allowed_roots)
        except (PermissionError, ValueError, OSError):
            return f"Path argument '{path_arg}' is outside the allowed roots."
    return None


async def run_terminal_command(command: str) -> str:
    """
    Execute allowlisted terminal commands without invoking a shell.

    Security contract:
    - Reject shell metacharacters outside quoted strings.
    - Parse into argv and execute with create_subprocess_exec (no shell).
    - Enforce first-token allowlist from AIDEN_TERMINAL_ALLOWLIST.
    - For commands that accept path arguments (ls, dir, wc), confine those
      paths to the configured allowed roots via path_guard.
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

    path_err = _check_path_argument(argv)
    if path_err:
        return _security_error(path_err)

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
