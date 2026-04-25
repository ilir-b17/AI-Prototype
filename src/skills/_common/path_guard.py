import os
from typing import List

_PROJECT_ROOT = os.path.realpath(
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
)
_DEFAULT_DOWNLOADS_DIR = os.path.join(_PROJECT_ROOT, "downloads")
_PROJECT_DENY_PATHS = (
    os.path.realpath(os.path.join(_PROJECT_ROOT, "data")),
    os.path.realpath(os.path.join(_PROJECT_ROOT, ".env")),
    os.path.realpath(os.path.join(_PROJECT_ROOT, "logs")),
    os.path.realpath(os.path.join(_PROJECT_ROOT, "src", "skills", "_pending")),
)


def _normalize_root(path_value: str) -> str:
    normalized = str(path_value or "").strip()
    if not normalized:
        raise PermissionError("Path is outside the allowed roots")
    return os.path.realpath(os.path.abspath(normalized))


def _is_within(path_value: str, root_value: str) -> bool:
    try:
        return os.path.commonpath([path_value, root_value]) == root_value
    except ValueError:
        return False


def _parse_extra_roots(raw_extra_roots: str) -> List[str]:
    if not raw_extra_roots:
        return []
    return [item.strip() for item in str(raw_extra_roots).split(":") if item.strip()]


def get_default_allowed_roots() -> List[str]:
    downloads_root = os.getenv("AIDEN_DOWNLOADS_DIR", _DEFAULT_DOWNLOADS_DIR)
    extra_roots = _parse_extra_roots(os.getenv("AIDEN_EXTRA_ALLOWED_ROOTS", ""))

    roots: List[str] = []
    for candidate in [downloads_root, _PROJECT_ROOT, *extra_roots]:
        try:
            resolved = _normalize_root(candidate)
        except PermissionError:
            continue
        if resolved not in roots:
            roots.append(resolved)
    return roots


def resolve_confined_path(user_path: str, allowed_roots: List[str]) -> str:
    """
    Resolve user_path to an absolute realpath and verify it is inside one of allowed_roots.
    Raises PermissionError on traversal attempts, missing input, or roots outside the allowlist.
    Symlinks are resolved before the check.
    """
    raw_path = str(user_path or "").strip()
    if not raw_path:
        raise PermissionError("Path is outside the allowed roots")

    if not isinstance(allowed_roots, list) or not allowed_roots:
        raise PermissionError("Path is outside the allowed roots")

    normalized_roots: List[str] = []
    for root_value in allowed_roots:
        resolved_root = _normalize_root(root_value)
        if resolved_root not in normalized_roots:
            normalized_roots.append(resolved_root)

    resolved_user_path = os.path.realpath(os.path.abspath(raw_path))
    if not any(_is_within(resolved_user_path, root) for root in normalized_roots):
        raise PermissionError("Path is outside the allowed roots")

    project_root_allowed = any(root == _PROJECT_ROOT for root in normalized_roots)
    if project_root_allowed:
        for denied_path in _PROJECT_DENY_PATHS:
            if _is_within(resolved_user_path, denied_path):
                raise PermissionError("Path is outside the allowed roots")

    return resolved_user_path
