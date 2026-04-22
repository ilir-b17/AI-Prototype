from __future__ import annotations

from typing import Optional

from src.memory.core_memory import CoreMemory
from src.memory.ledger_db import LedgerMemory


_ledger: Optional[LedgerMemory] = None
_core: Optional[CoreMemory] = None


def set_runtime_context(ledger: Optional[LedgerMemory], core: Optional[CoreMemory]) -> None:
    global _ledger, _core
    _ledger = ledger
    _core = core


def get_ledger() -> Optional[LedgerMemory]:
    return _ledger


def get_core_memory() -> Optional[CoreMemory]:
    return _core
