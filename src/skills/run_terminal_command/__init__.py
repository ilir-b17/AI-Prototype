import asyncio
import re
import logging

logger = logging.getLogger(__name__)

# Regex strictly prevents the execution of destructive base commands
# Matches optional whitespace, optional 'sudo ' (and any combination of spaces), followed by the blacklisted commands
BLACKLIST_REGEX = re.compile(r"^\s*(?:sudo\s+)?(rm|mkfs|dd|shutdown|reboot|chmod|chown|mv)(?:\s|$)", re.IGNORECASE)

async def run_terminal_command(command: str) -> str:
    """
    Executes a terminal command safely via shell.
    Blocks inherently destructive commands using a blacklist regex.
    Applies a strict 10-second timeout.
    """
    logger.info(f"run_terminal_command called with command: {command}")

    if BLACKLIST_REGEX.search(command):
        return "Error: Command is blocked by security policy. Destructive operations are not allowed."

    try:
        # Use create_subprocess_shell to run the command
        process = await asyncio.create_subprocess_shell(
            command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )

        # Apply a 10-second timeout
        try:
            stdout_bytes, stderr_bytes = await asyncio.wait_for(process.communicate(), timeout=10.0)
        except asyncio.TimeoutError:
            try:
                process.kill()
                await process.wait() # Make sure process is properly cleaned up
            except ProcessLookupError:
                pass
            return "Error: The terminal command exceeded the strict 10-second timeout limit and was terminated."

        stdout = stdout_bytes.decode('utf-8') if stdout_bytes else ""
        stderr = stderr_bytes.decode('utf-8') if stderr_bytes else ""

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
        return "An error occurred while trying to execute the terminal command."
