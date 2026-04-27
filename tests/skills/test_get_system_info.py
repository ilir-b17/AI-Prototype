import json
from datetime import datetime, timezone

import src.skills.get_system_info as system_info_skill


class _FakeDateTime:
    @staticmethod
    def now() -> datetime:
        return datetime(2026, 1, 2, 3, 4, 5, tzinfo=timezone.utc)


def test_get_system_info_success(monkeypatch) -> None:
    monkeypatch.setattr(system_info_skill, "datetime", _FakeDateTime)
    monkeypatch.setattr(system_info_skill.platform, "system", lambda: "TestOS")
    monkeypatch.setattr(system_info_skill.platform, "release", lambda: "1.0")
    monkeypatch.setattr(system_info_skill.platform, "machine", lambda: "x86_test")
    monkeypatch.setattr(system_info_skill.psutil, "cpu_count", lambda logical=True: 8)
    monkeypatch.setattr(system_info_skill.psutil, "cpu_percent", lambda interval=0.1: 12.5)
    monkeypatch.setattr(
        system_info_skill.psutil,
        "virtual_memory",
        lambda: type("Mem", (), {"total": 16 * 1024**3, "available": 10 * 1024**3, "percent": 37.5})(),
    )
    monkeypatch.setattr(
        system_info_skill.psutil,
        "disk_usage",
        lambda _path: type("Disk", (), {"total": 500 * 1024**3, "free": 125 * 1024**3, "percent": 75.0})(),
    )

    payload = json.loads(system_info_skill.get_system_info())

    assert payload["datetime"] == "2026-01-02 03:04:05 UTC +0000"
    assert payload["os"] == "TestOS 1.0"
    assert payload["machine"] == "x86_test"
    assert payload["cpu"] == {"cores": 8, "usage_percent": 12.5}
    assert payload["ram"] == {"total_gb": 16.0, "available_gb": 10.0, "usage_percent": 37.5}
    assert payload["disk"] == {"total_gb": 500.0, "free_gb": 125.0, "usage_percent": 75.0}


def test_get_system_info_returns_error_payload_when_exception_occurs(monkeypatch) -> None:
    def _raise(*_args, **_kwargs):
        raise RuntimeError("psutil unavailable")

    monkeypatch.setattr(system_info_skill.psutil, "cpu_count", _raise)

    payload = json.loads(system_info_skill.get_system_info())

    assert payload["status"] == "error"
    assert payload["message"] == "Failed to retrieve system information: psutil unavailable"
    assert "permission issues" in payload["details"]

