import json
import os
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from src.core.orchestrator import Orchestrator
from src.core.skill_manager import SkillRegistry
from src.skills.analyze_table_file import analyze_table_file
from src.skills.extract_pdf_text import extract_pdf_text
from src.skills.manage_file_system import manage_file_system
from src.skills._common.path_guard import resolve_confined_path


def _outside_system_path() -> str:
    if os.name == "nt":
        return r"C:\Windows\System32\drivers\etc\hosts"
    return "/etc/passwd"


@pytest.fixture
def confined_downloads(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    downloads = tmp_path / "downloads"
    downloads.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("AIDEN_DOWNLOADS_DIR", str(downloads))
    monkeypatch.delenv("AIDEN_EXTRA_ALLOWED_ROOTS", raising=False)
    return downloads


def test_path_guard_blocks_traversal_with_dotdot(tmp_path: Path) -> None:
    allowed_root = tmp_path / "allowed"
    allowed_root.mkdir(parents=True, exist_ok=True)
    outside_target = tmp_path / "outside.txt"
    outside_target.write_text("secret", encoding="utf-8")

    traversal_input = os.path.join(str(allowed_root), "..", outside_target.name)

    with pytest.raises(PermissionError):
        resolve_confined_path(traversal_input, [str(allowed_root)])


def test_path_guard_blocks_absolute_path_outside_allowed_roots(tmp_path: Path) -> None:
    allowed_root = tmp_path / "allowed"
    allowed_root.mkdir(parents=True, exist_ok=True)
    outside_target = tmp_path / "outside.txt"
    outside_target.write_text("secret", encoding="utf-8")

    with pytest.raises(PermissionError):
        resolve_confined_path(str(outside_target), [str(allowed_root)])


def test_path_guard_blocks_symlink_pointing_outside_allowed_roots(tmp_path: Path) -> None:
    allowed_root = tmp_path / "allowed"
    allowed_root.mkdir(parents=True, exist_ok=True)
    outside_target = tmp_path / "outside.txt"
    outside_target.write_text("secret", encoding="utf-8")

    symlink_path = allowed_root / "inside_link.txt"
    try:
        os.symlink(outside_target, symlink_path)
    except (OSError, NotImplementedError):
        pytest.skip("Symlink creation is not available in this environment")

    with pytest.raises(PermissionError):
        resolve_confined_path(str(symlink_path), [str(allowed_root)])


def test_path_guard_allows_valid_relative_path_inside_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    allowed_root = tmp_path / "allowed"
    nested_dir = allowed_root / "nested"
    nested_dir.mkdir(parents=True, exist_ok=True)
    inside_file = nested_dir / "ok.txt"
    inside_file.write_text("ok", encoding="utf-8")

    monkeypatch.chdir(allowed_root)
    resolved = resolve_confined_path(os.path.join("nested", "ok.txt"), [str(allowed_root)])

    assert resolved == os.path.realpath(str(inside_file))


def test_path_guard_allows_valid_absolute_path_inside_downloads(tmp_path: Path) -> None:
    downloads = tmp_path / "downloads"
    downloads.mkdir(parents=True, exist_ok=True)
    inside_file = downloads / "ok.txt"
    inside_file.write_text("ok", encoding="utf-8")

    resolved = resolve_confined_path(str(inside_file), [str(downloads)])

    assert resolved == os.path.realpath(str(inside_file))


@pytest.mark.asyncio
async def test_manage_file_system_blocks_os_sensitive_path(confined_downloads: Path) -> None:
    result = await manage_file_system("read", _outside_system_path())
    payload = json.loads(result)

    assert payload["status"] == "error"
    assert payload["message"] == "Path is outside the allowed roots"


@pytest.mark.asyncio
async def test_manage_file_system_reads_file_inside_downloads(confined_downloads: Path) -> None:
    inside_file = confined_downloads / "inside.txt"
    inside_file.write_text("safe data", encoding="utf-8")

    result = await manage_file_system("read", str(inside_file))

    assert result == "safe data"


@pytest.mark.asyncio
async def test_extract_pdf_text_blocks_os_sensitive_path(confined_downloads: Path) -> None:
    result = await extract_pdf_text(_outside_system_path())
    payload = json.loads(result)

    assert payload["status"] == "error"
    assert payload["message"] == "Path is outside the allowed roots"


@pytest.mark.asyncio
async def test_extract_pdf_text_allows_downloads_path(confined_downloads: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    pdf_path = confined_downloads / "doc.pdf"
    pdf_path.write_bytes(b"%PDF-1.4")

    monkeypatch.setattr(
        "src.skills.extract_pdf_text._extract_pdf_text_sync",
        lambda file_path, max_pages, max_chars, full_context=False: json.dumps(
            {
                "status": "success",
                "file": os.path.basename(file_path),
                "text": "ok",
                "truncated": False,
            }
        ),
    )

    result = await extract_pdf_text(str(pdf_path))
    payload = json.loads(result)

    assert payload["status"] == "success"


@pytest.mark.asyncio
async def test_analyze_table_file_blocks_os_sensitive_path(confined_downloads: Path) -> None:
    result = await analyze_table_file(_outside_system_path())
    payload = json.loads(result)

    assert payload["status"] == "error"
    assert payload["message"] == "Path is outside the allowed roots"


@pytest.mark.asyncio
async def test_analyze_table_file_reads_csv_inside_downloads(confined_downloads: Path) -> None:
    csv_path = confined_downloads / "sample.csv"
    csv_path.write_text("a,b\n1,2\n", encoding="utf-8")

    result = await analyze_table_file(str(csv_path))
    payload = json.loads(result)

    assert payload["status"] == "success"
    assert payload["file"] == "sample.csv"


def test_capabilities_catalog_mentions_confined_path_descriptions() -> None:
    registry = SkillRegistry()
    orchestrator = Orchestrator.__new__(Orchestrator)
    orchestrator.cognitive_router = MagicMock()
    orchestrator.cognitive_router.registry = registry

    capabilities = Orchestrator._get_capabilities_string(orchestrator)

    assert "manage_file_system" in capabilities
    assert "extract_pdf_text" in capabilities
    assert "analyze_table_file" in capabilities
    assert "resolves and confines paths to the configured allowed roots" in capabilities
