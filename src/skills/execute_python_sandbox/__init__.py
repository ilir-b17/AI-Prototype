import asyncio
import sys
import tempfile
import os
import logging

logger = logging.getLogger(__name__)

import json

async def execute_python_sandbox(code_string: str) -> str:
    """
    Executes Python code safely in a subprocess with a configurable timeout.

    Uses asyncio.create_subprocess_exec so that:
    - The event loop is never blocked (no thread pool required).
    - asyncio.CancelledError correctly kills the child process instead of
      leaving it orphaned in a thread-pool worker.
    """
    logger.info("execute_python_sandbox called.")

    if not isinstance(code_string, str) or not code_string.strip():
        return json.dumps({
            "status": "error",
            "message": "Invalid code_string",
            "details": "The code_string must be a non-empty Python script."
        })

    try:
        timeout_seconds = float(os.getenv("TOOL_EXEC_TIMEOUT_SECONDS", "10.0"))
    except ValueError:
        timeout_seconds = 10.0

    tmp_dir_obj = tempfile.TemporaryDirectory()
    process = None
    try:
        temp_dir = tmp_dir_obj.name
        script_path = os.path.join(temp_dir, "sandbox_script.py")

        try:
            with open(script_path, "w") as f:
                f.write(code_string)
        except Exception as e:
            logger.error(f"Failed to write temporary script: {e}", exc_info=True)
            return json.dumps({
                "status": "error",
                "message": "Failed to write temporary script",
                "details": str(e)
            })

        try:
            process = await asyncio.create_subprocess_exec(
                sys.executable, script_path,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            try:
                stdout, stderr = await asyncio.wait_for(
                    process.communicate(), timeout=timeout_seconds
                )
            except asyncio.TimeoutError:
                process.kill()
                await process.wait()
                return json.dumps({
                    "status": "error",
                    "message": f"Execution timeout exceeded ({timeout_seconds}s)",
                    "details": "The Python script took too long to complete and was terminated. Check for infinite loops or long network requests."
                })
        except asyncio.CancelledError:
            # Ensure the child process is terminated when the parent task is
            # cancelled so it does not continue running in the background.
            if process is not None and process.returncode is None:
                process.kill()
                await process.wait()
            raise  # Re-raise so the asyncio task cancellation propagates correctly

        output = ""
        if stdout:
            output += f"--- STDOUT ---\n{stdout.decode(errors='replace')}\n"
        if stderr:
            output += f"--- STDERR ---\n{stderr.decode(errors='replace')}\n"

        return output or "Script executed successfully with no output."

    except asyncio.CancelledError:
        raise
    except Exception as e:
        logger.error(f"execute_python_sandbox error: {e}", exc_info=True)
        return json.dumps({
            "status": "error",
            "message": "An error occurred during sandbox execution",
            "details": str(e)
        })
    finally:
        tmp_dir_obj.cleanup()
