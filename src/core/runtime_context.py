"""
Runtime context module - module-level references to shared resources.

These are set once during Orchestrator.async_init() and read by skills
and other modules that cannot import Orchestrator directly.
"""

from __future__ import annotations

from typing import Any, Awaitable, Callable, List, Optional, TYPE_CHECKING

from src.memory.core_memory import CoreMemory
from src.memory.ledger_db import LedgerMemory

if TYPE_CHECKING:
    from src.memory.vector_db import VectorMemory
    from src.core.orchestrator import Orchestrator
    from src.core.memory_reranker import MemoryReranker


_ledger: Optional[LedgerMemory] = None
_core: Optional[CoreMemory] = None
_vector: Optional["VectorMemory"] = None
_orchestrator: Optional["Orchestrator"] = None

# Bound async callable for two-stage memory retrieval.
# Set to orchestrator._rerank_memories during async_init.
# Signature: async (query, candidates, n_results) -> List[Dict]
_reranker_fn: Optional[Callable[..., Awaitable[List[Any]]]] = None


def set_runtime_context(
    ledger: Optional[LedgerMemory],
    core: Optional[CoreMemory],
    vector: Optional["VectorMemory"] = None,
    orchestrator: Optional["Orchestrator"] = None,
    reranker_fn: Optional[Callable[..., Awaitable[List[Any]]]] = None,
) -> None:
    global _ledger, _core, _vector, _orchestrator, _reranker_fn
    _ledger = ledger
    _core = core
    _vector = vector
    _orchestrator = orchestrator
    _reranker_fn = reranker_fn


def get_ledger() -> Optional[LedgerMemory]:
    return _ledger


def get_core_memory() -> Optional[CoreMemory]:
    return _core


def get_vector_memory() -> Optional["VectorMemory"]:
    return _vector


def get_orchestrator() -> Optional["Orchestrator"]:
    return _orchestrator


def get_reranker_fn() -> Optional[Callable[..., Awaitable[List[Any]]]]:
    """Return the bound two-stage memory reranker callable, or None.

    Returns None when:
      - ENABLE_MEMORY_RERANKING=false
      - Orchestrator not yet initialised
      - System 1 not available

    Skills must check for None and fall back to cosine-only retrieval.
    """
    return _reranker_fn
