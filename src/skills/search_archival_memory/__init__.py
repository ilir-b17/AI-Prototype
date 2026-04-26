"""
search_archival_memory skill - semantic search over ChromaDB vector store
with optional session and epic scoping.
"""

import asyncio
import json
import logging
import os
from typing import Optional

logger = logging.getLogger(__name__)


async def search_archival_memory(
    query: str,
    n_results: int = 3,
    session_id: Optional[int] = None,
    epic_id: Optional[int] = None,
) -> str:
    """
    Queries the ChromaDB vector store for semantically similar memories.

    When session_id or epic_id are provided, results are filtered to memories
    tagged with those values. Falls back to unfiltered search if the scoped
    query returns fewer than 2 results (avoids over-narrowing).

    Args:
        query: The semantic search query.
        n_results: Number of results to return (1-10). Default 3.
        session_id: Optional session id to scope search to current project.
        epic_id: Optional epic id to scope search to a project epic.
    """
    logger.info(
        "search_archival_memory: query=%r session_id=%s epic_id=%s",
        query, session_id, epic_id,
    )

    if not isinstance(query, str) or not query.strip():
        return json.dumps({
            "status": "error",
            "message": "Invalid query",
            "details": "The query must be a non-empty string.",
        })

    try:
        n_results = max(1, min(int(n_results), 10))
    except (TypeError, ValueError):
        n_results = 3

    # Coerce session/epic ids to int or None
    _session_id: Optional[int] = None
    _epic_id: Optional[int] = None
    try:
        if session_id is not None:
            _session_id = int(session_id) if int(session_id) > 0 else None
    except (TypeError, ValueError):
        pass
    try:
        if epic_id is not None:
            _epic_id = int(epic_id) if int(epic_id) > 0 else None
    except (TypeError, ValueError):
        pass

    try:
        timeout_seconds = float(os.getenv("TOOL_EXEC_TIMEOUT_SECONDS", "10.0"))
    except ValueError:
        timeout_seconds = 10.0

    vector_memory = None
    owns_instance = False
    try:
        from src.core.runtime_context import get_vector_memory
        from src.memory.vector_db import VectorMemory

        vector_memory = get_vector_memory()
        if vector_memory is None:
            vector_memory = VectorMemory()
            owns_instance = True

        # Build ChromaDB where clause for scoped search
        where_clause = _build_where_clause(_session_id, _epic_id)

        async def _run_query(where):
            return await asyncio.wait_for(
                vector_memory.query_memory_async(
                    query,
                    n_results=n_results,
                    where=where,
                ),
                timeout=timeout_seconds,
            )

        if where_clause is not None:
            results = await _run_query(where_clause)
            # Fall back to global search if scoped results are too few
            if len(results) < 2:
                logger.info(
                    "Scoped archival search returned %d results; "
                    "falling back to global search.",
                    len(results),
                )
                global_results = await _run_query(None)
                # Merge: scoped results first, then global (deduplicated by id)
                seen_ids = {r.get("id") for r in results}
                for r in global_results:
                    if r.get("id") not in seen_ids:
                        results.append(r)
                        seen_ids.add(r.get("id"))
                results = results[:n_results]
        else:
            results = await _run_query(None)

        if not results:
            return json.dumps({
                "status": "success",
                "message": "No relevant archival memory found.",
                "results": [],
                "scoped": where_clause is not None,
            })

        return json.dumps({
            "status": "success",
            "scoped": where_clause is not None,
            "session_id": _session_id,
            "epic_id": _epic_id,
            "results": [
                {
                    "document": r.get("document", ""),
                    "metadata": r.get("metadata", {}),
                    "distance": r.get("distance"),
                }
                for r in results
            ],
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


def _build_where_clause(
    session_id: Optional[int],
    epic_id: Optional[int],
) -> Optional[dict]:
    """Build a ChromaDB $and where clause for session/epic filtering.

    ChromaDB where syntax uses $and for multiple conditions:
        {"$and": [{"session_id": {"$eq": 3}}, {"epic_id": {"$eq": 7}}]}
    Single condition uses direct form:
        {"session_id": {"$eq": 3}}
    Returns None when no filtering is needed.
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
