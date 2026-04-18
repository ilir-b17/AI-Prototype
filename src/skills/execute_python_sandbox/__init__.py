import asyncio
import sys
import tempfile
import os
import logging

logger = logging.getLogger(__name__)

async def execute_python_sandbox(code_string: str) -> str:
    """
    Executes Python code safely in a subprocess with a strict 10-second timeout.

    Uses asyncio.create_subprocess_exec so that:
    - The event loop is never blocked (no thread pool required).
    - asyncio.CancelledError correctly kills the child process instead of
      leaving it orphaned in a thread-pool worker.
    """
    logger.info("execute_python_sandbox called.")

    tmp_dir_obj = tempfile.TemporaryDirectory()
    process: asyncio.subprocess.Process = None
    try:
        temp_dir = tmp_dir_obj.name
        script_path = os.path.join(temp_dir, "sandbox_script.py")

        try:
            with open(script_path, "w") as f:
                f.write(code_string)
        except Exception as e:
            logger.error(f"Failed to write temporary script: {e}", exc_info=True)
            return "Error: Failed to write the temporary script to disk."

        try:
            process = await asyncio.create_subprocess_exec(
                sys.executable, script_path,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            try:
                stdout, stderr = await asyncio.wait_for(
                    process.communicate(), timeout=10
                )
            except asyncio.TimeoutError:
                process.kill()
                await process.wait()
                return (
                    "Error: The python script execution exceeded the strict "
                    "10-second timeout limit and was terminated."
                )
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
        return "An error occurred while trying to execute the python script."
    finally:
        tmp_dir_obj.cleanup()
