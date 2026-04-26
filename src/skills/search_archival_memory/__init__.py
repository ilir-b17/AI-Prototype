"""
search_archival_memory skill — two-stage memory retrieval.

Stage 1 (recall):  ChromaDB cosine similarity with broad n_results.
Stage 2 (rerank):  System 1 batch relevance scoring via runtime_context.

Falls back to cosine-only when:
  - ENABLE_MEMORY_RERANKING=false
  - skip_reranking=true parameter
  - Reranker callable not registered in runtime_context
  - Stage 2 fails for any reason
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

_DEFAULT_N_RESULTS = 3
_MAX_N_RESULTS = 10


def _build_where_clause(
    session_id: Optional[int],
    epic_id: Optional[int],
) -> Optional[dict]:
    """Build ChromaDB $and where clause for session/epic filtering.

    Returns None when no filtering is needed.
    Called by orchestrator._get_archival_context — keep in module scope.
    """
    conditions = []
    if session_id is not None:
        conditions.append({"session_id": {"$eq": session_id}})
    if epic_id is not None:
        conditions.append({"epic_id": {"$eq": epic_id}})
    if not conditions:
        return None
    if len(conditions) == 1:
        return conditions[0]
    return {"$and": conditions}


def _coerce_optional_int(value: Any) -> Optional[int]:
    if value is None:
        return None
    try:
        parsed = int(value)
        return parsed if parsed > 0 else None
    except (TypeError, ValueError):
        return None


def _format_result(item: Dict[str, Any]) -> Dict[str, Any]:
    """Format a VectorMemory result dict for skill output."""
    result: Dict[str, Any] = {
        "document": item.get("document", ""),
        "metadata": item.get("metadata", {}),
        "distance": round(float(item.get("distance") or 1.0), 4),
    }
    # Include reranking metadata if available
    if "rerank_score" in item:
        result["rerank_score"] = round(float(item["rerank_score"]), 3)
        result["cosine_score"] = round(float(item.get("cosine_score", 0.0)), 3)
        result["combined_score"] = round(float(item.get("combined_score", 0.0)), 3)
        result["reranked"] = bool(item.get("reranked", False))
    return result


async def search_archival_memory(
    query: str,
    n_results: int = _DEFAULT_N_RESULTS,
    session_id: Optional[int] = None,
    epic_id: Optional[int] = None,
    skip_reranking: bool = False,
) -> str:
    """Search archival memory using two-stage retrieval.

    Returns JSON with status, results list, and retrieval metadata.
    """
    logger.info(
        "search_archival_memory: query=%r n_results=%d session=%s epic=%s "
        "skip_reranking=%s",
        query[:60] if query else "",
        n_results,
        session_id,
        epic_id,
        skip_reranking,
    )

    if not isinstance(query, str) or not query.strip():
        return json.dumps({
            "status": "error",
            "message": "Invalid query",
            "details": "The query must be a non-empty string.",
        })

    try:
        n_results = max(1, min(int(n_results) if n_results else _DEFAULT_N_RESULTS, _MAX_N_RESULTS))
    except (TypeError, ValueError):
        n_results = _DEFAULT_N_RESULTS

    _session_id = _coerce_optional_int(session_id)
    _epic_id = _coerce_optional_int(epic_id)
    _skip_reranking = bool(skip_reranking)

    timeout_seconds = float(os.getenv("TOOL_EXEC_TIMEOUT_SECONDS", "30.0"))

    vector_memory = None
    owns_instance = False

    try:
        from src.core.runtime_context import get_vector_memory, get_reranker_fn
        from src.memory.vector_db import VectorMemory

        vector_memory = get_vector_memory()
        if vector_memory is None:
            vector_memory = VectorMemory()
            owns_instance = True

        # Determine whether to use reranking
        reranker_fn = None if _skip_reranking else get_reranker_fn()
        use_reranking = reranker_fn is not None

        # Determine broad_n for Stage 1
        if use_reranking:
            try:
                from src.core.memory_reranker import MemoryReranker, load_rerank_config
                _tmp_config = load_rerank_config()
                _tmp_reranker = MemoryReranker(_tmp_config)
                broad_n = _tmp_reranker.broad_n_results(n_results)
            except Exception:
                broad_n = n_results * 4
        else:
            broad_n = n_results

        # Build session/epic where clause
        where_clause = _build_where_clause(_session_id, _epic_id)

        # Stage 1: broad cosine recall
        async def _run_query(where: Optional[dict]) -> List[Dict[str, Any]]:
            return await asyncio.wait_for(
                vector_memory.query_memory_async(query, n_results=broad_n, where=where),
                timeout=timeout_seconds,
            )

        if where_clause is not None:
            candidates = await _run_query(where_clause)
            # Fallback to global search if scoped returns too few results
            if len(candidates) < 2:
                logger.info(
                    "Scoped recall returned %d candidates; falling back to global.",
                    len(candidates),
                )
                global_candidates = await _run_query(None)
                # Merge: scoped results first (deduplicated by id)
                seen_ids = {c.get("id") for c in candidates}
                for c in global_candidates:
                    if c.get("id") not in seen_ids:
                        candidates.append(c)
                        seen_ids.add(c.get("id"))
        else:
            candidates = await _run_query(None)

        if not candidates:
            return json.dumps({
                "status": "success",
                "message": "No relevant archival memory found.",
                "results": [],
                "scoped": where_clause is not None,
                "reranked": False,
                "candidates_retrieved": 0,
            })

        # Stage 2: rerank if available and worthwhile
        if use_reranking and len(candidates) > n_results:
            try:
                results = await asyncio.wait_for(
                    reranker_fn(query, candidates, n_results),
                    timeout=timeout_seconds,
                )
                reranked = True
            except Exception as rerank_exc:
                logger.warning(
                    "Reranking failed in skill (cosine fallback): %s", rerank_exc
                )
                results = sorted(
                    candidates,
                    key=lambda c: float(c.get("distance") or 1.0),
                )[:n_results]
                reranked = False
        else:
            results = sorted(
                candidates,
                key=lambda c: float(c.get("distance") or 1.0),
            )[:n_results]
            reranked = False

        return json.dumps({
            "status": "success",
            "scoped": where_clause is not None,
            "session_id": _session_id,
            "epic_id": _epic_id,
            "reranked": reranked,
            "candidates_retrieved": len(candidates),
            "results_returned": len(results),
            "results": [_format_result(r) for r in results],
        }, indent=2)

    except asyncio.TimeoutError:
        return json.dumps({
            "status": "error",
            "message": f"Archival search timed out after {timeout_seconds}s.",
        })
    except Exception as exc:
        logger.error("search_archival_memory error: %s", exc, exc_info=True)
        return json.dumps({
            "status": "error",
            "message": "Could not search archival memory",
            "details": str(exc),
        })
    finally:
        if owns_instance and vector_memory is not None:
            try:
                vector_memory.close()
            except Exception:
                pass
