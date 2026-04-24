from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Optional
from urllib.parse import urlparse
from urllib.request import Request, urlopen

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.core.skill_manager import SkillRegistry

_URL_SCHEMES = {"http", "https"}
_SKILL_NAME_RE = re.compile(r"^[a-z0-9][a-z0-9_-]{0,63}$")


def validate_source_url(url: str) -> str:
    parsed = urlparse(str(url or "").strip())
    if parsed.scheme.lower() not in _URL_SCHEMES or not parsed.netloc:
        raise ValueError(
            "Malformed URL. Expected an absolute http(s) URL, for example: "
            "https://raw.githubusercontent.com/owner/repo/main/path/SKILL.md"
        )
    return parsed.geturl()


def _download_markdown(url: str, timeout: float = 20.0) -> str:
    request = Request(url, headers={"User-Agent": "AIDEN-Skill-Ingest/1.0"})
    with urlopen(request, timeout=timeout) as response:
        data = response.read()
    return data.decode("utf-8", errors="replace")


def _safe_skill_name(raw_name: str, fallback: str = "ingested-skill") -> str:
    candidate = str(raw_name or "").strip().lower().replace(" ", "-")
    candidate = re.sub(r"[^a-z0-9_-]+", "-", candidate).strip("-_")
    candidate = candidate or fallback
    if not _SKILL_NAME_RE.fullmatch(candidate):
        raise ValueError(
            f"Invalid skill name '{raw_name}'. Allowed pattern: "
            "[a-z0-9][a-z0-9_-]{0,63}"
        )
    return candidate


def _validate_frontmatter(skill_markdown: str, source_label: str) -> dict:
    front_matter = SkillRegistry._extract_front_matter(skill_markdown, source_label)
    if front_matter is None:
        raise ValueError("Skill ingestion requires YAML frontmatter in SKILL.md.")

    metadata = SkillRegistry._parse_front_matter(front_matter, source_label)
    description = str(metadata.get("description") or "").strip()
    if not description:
        raise ValueError("Frontmatter field 'description' is required for ingestion.")
    return metadata


def ingest_external_skill(
    source_url: str,
    *,
    skills_root: Optional[Path] = None,
    pending_dir_name: str = "_pending",
    force: bool = False,
    timeout: float = 20.0,
    downloader: Optional[Callable[[str, float], str]] = None,
) -> Path:
    validated_url = validate_source_url(source_url)
    fetch = downloader or _download_markdown
    markdown = fetch(validated_url, timeout)

    metadata = _validate_frontmatter(markdown, validated_url)
    fallback_name = Path(urlparse(validated_url).path).stem or "ingested-skill"
    skill_name = _safe_skill_name(str(metadata.get("name") or fallback_name))

    root = Path(skills_root) if skills_root is not None else Path(__file__).resolve().parents[1] / "src" / "skills"
    pending_root = root / pending_dir_name
    pending_root.mkdir(parents=True, exist_ok=True)

    target_dir = pending_root / skill_name
    if target_dir.exists() and not force:
        raise FileExistsError(
            f"Staging target already exists: {target_dir}. Use --force to overwrite staged files."
        )
    target_dir.mkdir(parents=True, exist_ok=True)

    (target_dir / "SKILL.md").write_text(markdown, encoding="utf-8")
    (target_dir / "ingest_meta.json").write_text(
        json.dumps(
            {
                "source_url": validated_url,
                "ingested_at": datetime.now(timezone.utc).isoformat(),
                "skill_name": skill_name,
                "pending_directory": str(target_dir),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    return target_dir


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Download an external SKILL.md and stage it safely under src/skills/_pending/ "
            "for manual admin review."
        )
    )
    parser.add_argument("source_url", help="Absolute http(s) URL to the external SKILL.md")
    parser.add_argument(
        "--skills-root",
        default=str(Path(__file__).resolve().parents[1] / "src" / "skills"),
        help="Skills root directory (default: src/skills)",
    )
    parser.add_argument(
        "--pending-dir",
        default="_pending",
        help="Staging directory name under skills root (default: _pending)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite staged files if the target already exists.",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=20.0,
        help="Network timeout in seconds for downloading the source markdown.",
    )
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    try:
        staged_dir = ingest_external_skill(
            args.source_url,
            skills_root=Path(args.skills_root),
            pending_dir_name=args.pending_dir,
            force=bool(args.force),
            timeout=float(args.timeout),
        )
    except Exception as exc:
        print(f"[ingest_external_skill] ERROR: {exc}", file=sys.stderr)
        return 1

    print("[ingest_external_skill] SUCCESS")
    print(f"staged_dir={staged_dir}")
    print("next_step=Review SKILL.md manually, then move approved skills into src/skills/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
