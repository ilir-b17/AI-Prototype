import asyncio
import subprocess
import sys
import tempfile
import os
import logging

logger = logging.getLogger(__name__)

async def execute_python_sandbox(code_string: str) -> str:
    """
    Executes Python code safely using a temporary script and a strict 10-second timeout.
    """
    logger.info("execute_python_sandbox called.")

    # We use a temporary directory to keep the workspace clean
    with tempfile.TemporaryDirectory() as temp_dir:
        script_path = os.path.join(temp_dir, "sandbox_script.py")

        try:
            with open(script_path, "w") as f:
                f.write(code_string)
        except Exception as e:
            logger.error(f"Failed to write temporary script: {e}", exc_info=True)
            return "Error: Failed to write the temporary script to disk."

        try:
            # We run it synchronously in an executor or just run it since it's subprocess.run
            # To avoid blocking the event loop entirely, let's wrap it in asyncio.to_thread
            def run_script():
                return subprocess.run(
                    [sys.executable, script_path],
                    capture_output=True,
                    text=True,
                    timeout=10
                )

            result = await asyncio.to_thread(run_script)

            output = ""
            if result.stdout:
                output += f"--- STDOUT ---\n{result.stdout}\n"
            if result.stderr:
                output += f"--- STDERR ---\n{result.stderr}\n"

            if not output:
                output = "Script executed successfully with no output."

            return output

        except subprocess.TimeoutExpired:
            return "Error: The python script execution exceeded the strict 10-second timeout limit and was terminated."
        except Exception as e:
            logger.error(f"execute_python_sandbox error: {e}", exc_info=True)
            return f"An error occurred while trying to execute the python script."
