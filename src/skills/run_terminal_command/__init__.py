import asyncio
import re
import logging

logger = logging.getLogger(__name__)

# Regex strictly prevents the execution of destructive base commands.
# Covers both Unix destructive commands and Windows equivalents.
BLACKLIST_REGEX = re.compile(
    r"^\s*(?:sudo\s+)?"
    r"(rm|mkfs|dd|shutdown|reboot|chmod|chown|mv"
    r"|del|erase|format|rmdir|rd|cipher|attrib|icacls|takeown|reg\s+delete|net\s+stop|sc\s+delete)"
    r"(?:\s|$)",
    re.IGNORECASE
)

import os
import json

async def run_terminal_command(command: str) -> str:
    """
    Executes a terminal command safely via shell.
    Blocks inherently destructive commands using a blacklist regex.
    Applies a strict timeout, configurable via TOOL_EXEC_TIMEOUT_SECONDS.
    """
    logger.info(f"run_terminal_command called with command: {command}")

    if not isinstance(command, str) or not command.strip():
        return json.dumps({
            "status": "error",
            "message": "Invalid command",
            "details": "The command must be a non-empty string."
        })

    if BLACKLIST_REGEX.search(command):
        return json.dumps({
            "status": "error",
            "message": "Command blocked by security policy",
            "details": "Destructive operations (like rm, mv, shutdown) are not allowed. Please modify your command."
        })

    try:
        timeout_seconds = float(os.getenv("TOOL_EXEC_TIMEOUT_SECONDS", "10.0"))
    except ValueError:
        timeout_seconds = 10.0

    try:
        # Use create_subprocess_shell to run the command
        process = await asyncio.create_subprocess_shell(
            command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )

        try:
            stdout_bytes, stderr_bytes = await asyncio.wait_for(process.communicate(), timeout=timeout_seconds)
        except asyncio.TimeoutError:
            try:
                process.kill()
                await process.wait() # Make sure process is properly cleaned up
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
