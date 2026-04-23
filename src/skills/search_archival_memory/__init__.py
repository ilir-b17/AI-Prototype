import asyncio
import logging

logger = logging.getLogger(__name__)


import json
import os

def _sync_search_archival_memory(query: str, n_results: int) -> str:
    try:
        from src.memory.vector_db import VectorMemory
        vector_memory = VectorMemory()
        results = vector_memory.query_memory(query, n_results=n_results)
        if not results:
            return json.dumps({
                "status": "success",
                "message": "No relevant archival memory found for the query.",
                "results": []
            })

        parsed_results = []
        for res in results:
            parsed_results.append({
                "document": res.get("document", ""),
                "metadata": res.get("metadata", {}),
                "distance": res.get("distance", None)
            })

        return json.dumps({
            "status": "success",
            "results": parsed_results
        }, indent=2)
    except Exception as exc:
        return json.dumps({
            "status": "error",
            "message": "Could not search archival memory",
            "details": str(exc),
            "suggestion": "Check if the ChromaDB vector storage is properly initialized."
        })


async def search_archival_memory(query: str, n_results: int = 3) -> str:
    logger.info(f"search_archival_memory: {query}")

    if not isinstance(query, str) or not query.strip():
        return json.dumps({
            "status": "error",
            "message": "Invalid query",
            "details": "The search query must be a non-empty string."
        })

    try:
        n_results = int(n_results)
        n_results = min(max(1, n_results), 10)
    except ValueError:
        n_results = 3

    try:
        timeout_seconds = float(os.getenv("TOOL_EXEC_TIMEOUT_SECONDS", "10.0"))
    except ValueError:
        timeout_seconds = 10.0

    try:
        return await asyncio.wait_for(
            asyncio.to_thread(_sync_search_archival_memory, query, n_results),
            timeout=timeout_seconds
        )
    except asyncio.TimeoutError:
         return json.dumps({
            "status": "error",
            "message": f"Archival search timed out after {timeout_seconds}s",
            "details": "The vector database took too long to respond."
         })
