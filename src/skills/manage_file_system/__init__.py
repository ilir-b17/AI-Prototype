import asyncio
import os
import aiofiles
import logging

from src.skills._common.path_guard import get_default_allowed_roots, resolve_confined_path

logger = logging.getLogger(__name__)

def _read_pdf(file_path: str) -> str:
    from pypdf import PdfReader
    reader = PdfReader(file_path)
    pages = []
    for page in reader.pages:
        text = page.extract_text()
        if text:
            pages.append(text)
    return "\n\n".join(pages) if pages else "(No extractable text found in PDF)"

import json

async def manage_file_system(action: str, file_path: str, content: str = None) -> str:
    """
    Manages file system operations.
    'read' and 'list' operations execute immediately.
    'write' and 'delete' operations are restricted and trigger a security lock message.
    """
    logger.info(f"manage_file_system called with action: {action}, file_path: {file_path}")

    if action not in ('read', 'write', 'list', 'delete'):
        return json.dumps({
            "status": "error",
            "message": f"Unsupported action: {action}",
            "details": "Supported actions are 'read', 'write', 'list', 'delete'. Please correct the action and try again."
        })

    if not isinstance(file_path, str) or not file_path.strip():
        return json.dumps({
            "status": "error",
            "message": "Invalid file_path",
            "details": "The file_path must be a non-empty string."
        })

    if action in ('write', 'delete'):
        return "SECURITY LOCK: Write and delete operations are restricted. You must use the 'request_core_update' tool to propose these file modifications to the Admin for MFA authorization."

    try:
        try:
            resolved_path = resolve_confined_path(file_path, get_default_allowed_roots())
        except PermissionError:
            return json.dumps({
                "status": "error",
                "message": "Path is outside the allowed roots"
            })

        if action == 'read':
            if not os.path.exists(resolved_path):
                return json.dumps({
                    "status": "error",
                    "message": f"File not found: {resolved_path}",
                    "details": "Check if the path is correct or if you need to use the 'list' action on the parent directory to find the correct filename."
                })
            if not os.path.isfile(resolved_path):
                return json.dumps({
                    "status": "error",
                    "message": f"Target is a directory, not a file: {resolved_path}",
                    "details": "Use the 'list' action instead to view the contents of this directory."
                })

            if resolved_path.lower().endswith('.pdf'):
                loop = asyncio.get_running_loop()
                data = await loop.run_in_executor(None, _read_pdf, resolved_path)
                return data

            async with aiofiles.open(resolved_path, mode='r', encoding='utf-8', errors='replace') as f:
                data = await f.read()
            return data

        elif action == 'list':
            if not os.path.exists(resolved_path):
                return json.dumps({
                    "status": "error",
                    "message": f"Directory not found: {resolved_path}",
                    "details": "Check if the path is correct or list the parent directory to verify its contents."
                })
            if not os.path.isdir(resolved_path):
                return json.dumps({
                    "status": "error",
                    "message": f"Target is a file, not a directory: {resolved_path}",
                    "details": "Use the 'read' action instead to view the contents of this file."
                })

            entries = os.listdir(resolved_path)
            return json.dumps({"status": "success", "entries": entries}, indent=2)

    except Exception as e:
        logger.error(f"manage_file_system error: {str(e)}", exc_info=True)
        return json.dumps({
            "status": "error",
            "message": f"An error occurred during '{action}' on '{file_path}': {str(e)}",
            "details": "Ensure you have the necessary permissions and the file system is accessible."
        })
