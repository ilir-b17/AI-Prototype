import platform
import logging
from datetime import datetime

logger = logging.getLogger(__name__)


async def get_system_info() -> str:
    """Returns current datetime, timezone, and host platform."""
    try:
        now = datetime.now().astimezone()
        tz = now.strftime("%Z %z")
        return (
            f"DateTime: {now.strftime('%Y-%m-%d %H:%M:%S')} {tz} | "
            f"Platform: {platform.system()} {platform.release()} | "
            f"Machine: {platform.machine()}"
        )
    except Exception as exc:
        return f"Error retrieving system info: {exc}"
