"""
system_tools.py — DEPRECATED

All tools have been migrated to the modular skill registry under src/skills/.
This file is kept as a compatibility shim and is no longer the source of truth.

Do not add new tools here. Add them as a new folder in src/skills/ instead.
"""
import warnings
warnings.warn(
    "src.tools.system_tools is deprecated — tools now live in src/skills/. "
    "This shim will be removed in a future sprint.",
    DeprecationWarning,
    stacklevel=2,
)
