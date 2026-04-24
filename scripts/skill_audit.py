from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Dict, List

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.core.skill_manager import SkillRegistry

CATEGORY_LEGACY_JSON = "Legacy JSON"
CATEGORY_HYBRID = "Hybrid (Frontmatter + Executable)"
CATEGORY_INSTRUCTIONAL = "Instructional (Frontmatter Only)"
CATEGORY_UNCATEGORIZED = "Uncategorized"


def _has_json_schema_block(markdown: str) -> bool:
    return bool(re.search(r"```json\s*", markdown or ""))


def _classify_skill_dir(skill_dir: Path) -> str:
    skill_md = skill_dir / "SKILL.md"
    init_py = skill_dir / "__init__.py"

    if not skill_md.exists():
        return CATEGORY_UNCATEGORIZED

    markdown = skill_md.read_text(encoding="utf-8")
    has_init = init_py.exists()
    has_schema = _has_json_schema_block(markdown)

    try:
        front_matter = SkillRegistry._extract_front_matter(markdown, str(skill_md))
    except ValueError:
        return CATEGORY_UNCATEGORIZED

    has_frontmatter = front_matter is not None

    if has_frontmatter and has_init:
        return CATEGORY_HYBRID
    if has_frontmatter and not has_init:
        return CATEGORY_INSTRUCTIONAL
    if not has_frontmatter and has_init and has_schema:
        return CATEGORY_LEGACY_JSON
    return CATEGORY_UNCATEGORIZED


def audit_skills(skills_root: Path, *, include_pending: bool = False) -> Dict[str, List[str]]:
    report: Dict[str, List[str]] = {
        CATEGORY_LEGACY_JSON: [],
        CATEGORY_HYBRID: [],
        CATEGORY_INSTRUCTIONAL: [],
        CATEGORY_UNCATEGORIZED: [],
    }

    for entry in sorted(skills_root.iterdir()):
        if not entry.is_dir():
            continue
        if entry.name.startswith("_") and not include_pending:
            continue
        category = _classify_skill_dir(entry)
        report[category].append(entry.name)

    return report


def print_report(report: Dict[str, List[str]]) -> None:
    print("Skill Migration Audit")
    print("====================")
    for category in (
        CATEGORY_LEGACY_JSON,
        CATEGORY_HYBRID,
        CATEGORY_INSTRUCTIONAL,
        CATEGORY_UNCATEGORIZED,
    ):
        items = report.get(category, [])
        print(f"\n{category}: {len(items)}")
        for name in items:
            print(f"  - {name}")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Audit src/skills and categorize migration status by schema format."
    )
    parser.add_argument(
        "--skills-root",
        default=str(Path(__file__).resolve().parents[1] / "src" / "skills"),
        help="Skills root directory (default: src/skills)",
    )
    parser.add_argument(
        "--include-pending",
        action="store_true",
        help="Include underscore-prefixed directories such as _pending.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    skills_root = Path(args.skills_root)
    if not skills_root.exists() or not skills_root.is_dir():
        print(f"ERROR: skills root does not exist: {skills_root}")
        return 1

    report = audit_skills(skills_root, include_pending=bool(args.include_pending))
    print_report(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
