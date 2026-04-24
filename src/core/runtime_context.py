from __future__ import annotations

from typing import Optional, TYPE_CHECKING

from src.memory.core_memory import CoreMemory
from src.memory.ledger_db import LedgerMemory

if TYPE_CHECKING:
    from src.memory.vector_db import VectorMemory


_ledger: Optional[LedgerMemory] = None
_core: Optional[CoreMemory] = None
_vector: Optional["VectorMemory"] = None


def set_runtime_context(
    ledger: Optional[LedgerMemory],
    core: Optional[CoreMemory],
    vector: Optional["VectorMemory"] = None,
) -> None:
    global _ledger, _core, _vector
    _ledger = ledger
    _core = core
    _vector = vector


def get_ledger() -> Optional[LedgerMemory]:
    return _ledger


def get_core_memory() -> Optional[CoreMemory]:
    return _core


def get_vector_memory() -> Optional["VectorMemory"]:
    return _vector
