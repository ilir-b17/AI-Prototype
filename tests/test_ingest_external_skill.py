import json
from pathlib import Path

import pytest

from scripts.ingest_external_skill import ingest_external_skill


def test_ingestion_rejects_malformed_url(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="Malformed URL"):
        ingest_external_skill(
            "not-a-valid-url",
            skills_root=tmp_path / "skills",
            downloader=lambda _url, _timeout: "",
        )



def test_ingestion_stages_skill_under_pending_directory(tmp_path: Path) -> None:
    skills_root = tmp_path / "skills"
    source_url = "https://example.com/skills/cost-optimizer/SKILL.md"
    markdown = """---
name: cost-optimizer
description: Gives guidance for cloud cost optimization decisions.
---

Use this skill to reason about cost tradeoffs.
"""

    staged_dir = ingest_external_skill(
        source_url,
        skills_root=skills_root,
        downloader=lambda _url, _timeout: markdown,
    )

    assert staged_dir == skills_root / "_pending" / "cost-optimizer"
    assert (staged_dir / "SKILL.md").exists()
    assert not (skills_root / "cost-optimizer").exists()

    metadata = json.loads((staged_dir / "ingest_meta.json").read_text(encoding="utf-8"))
    assert metadata["source_url"] == source_url
    assert metadata["skill_name"] == "cost-optimizer"
