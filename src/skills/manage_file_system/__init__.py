import asyncio
import os
import aiofiles
import logging

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

async def manage_file_system(action: str, file_path: str, content: str = None) -> str:
    """
    Manages file system operations.
    'read' and 'list' operations execute immediately.
    'write' and 'delete' operations are restricted and trigger a security lock message.
    """
    logger.info(f"manage_file_system called with action: {action}, file_path: {file_path}")

    if action in ('write', 'delete'):
        return "SECURITY LOCK: Write and delete operations are restricted. You must use the 'request_core_update' tool to propose these file modifications to the Admin for MFA authorization."

    try:
        if action == 'read':
            if not os.path.exists(file_path) or not os.path.isfile(file_path):
                return f"Error: File not found or is a directory - {file_path}"

            if file_path.lower().endswith('.pdf'):
                loop = asyncio.get_event_loop()
                data = await loop.run_in_executor(None, _read_pdf, file_path)
                return data

            async with aiofiles.open(file_path, mode='r', encoding='utf-8', errors='replace') as f:
                data = await f.read()
            return data

        elif action == 'list':
            if not os.path.exists(file_path) or not os.path.isdir(file_path):
                return f"Error: Directory not found or is a file - {file_path}"

            entries = os.listdir(file_path)
            return "\n".join(entries)

        else:
            return f"Error: Unsupported action - {action}. Supported actions are 'read', 'write', 'list', 'delete'."

    except Exception as e:
        logger.error(f"manage_file_system error: {str(e)}", exc_info=True)
        return f"An error occurred while trying to perform '{action}' on '{file_path}'."
