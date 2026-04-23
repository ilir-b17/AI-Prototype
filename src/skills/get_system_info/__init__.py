import platform
import logging
from datetime import datetime
import json
import psutil

logger = logging.getLogger(__name__)


def get_system_info() -> str:
    """Returns current datetime, timezone, and host hardware platform details."""
    try:
        now = datetime.now().astimezone()
        tz = now.strftime("%Z %z")

        # Hardware specs
        cpu_count = psutil.cpu_count(logical=True)
        cpu_percent = psutil.cpu_percent(interval=0.1)

        memory = psutil.virtual_memory()
        total_ram_gb = memory.total / (1024 ** 3)
        available_ram_gb = memory.available / (1024 ** 3)
        ram_percent = memory.percent

        disk = psutil.disk_usage('/')
        total_disk_gb = disk.total / (1024 ** 3)
        free_disk_gb = disk.free / (1024 ** 3)
        disk_percent = disk.percent

        info = {
            "datetime": f"{now.strftime('%Y-%m-%d %H:%M:%S')} {tz}",
            "os": f"{platform.system()} {platform.release()}",
            "machine": platform.machine(),
            "cpu": {
                "cores": cpu_count,
                "usage_percent": cpu_percent
            },
            "ram": {
                "total_gb": round(total_ram_gb, 2),
                "available_gb": round(available_ram_gb, 2),
                "usage_percent": ram_percent
            },
            "disk": {
                "total_gb": round(total_disk_gb, 2),
                "free_gb": round(free_disk_gb, 2),
                "usage_percent": disk_percent
            }
        }

        return json.dumps(info, indent=2)

    except Exception as exc:
        logger.error(f"Error retrieving system info: {exc}", exc_info=True)
        return json.dumps({
            "status": "error",
            "message": f"Failed to retrieve system information: {str(exc)}",
            "details": "The psutil library might have encountered permission issues or the environment might lack necessary system access."
        })
