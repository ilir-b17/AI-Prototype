import asyncio
import json
import logging
import os

logger = logging.getLogger(__name__)


async def search_archival_memory(query: str, n_results: int = 3) -> str:
    logger.info(f"search_archival_memory: {query}")

    if not isinstance(query, str) or not query.strip():
        return json.dumps({
            "status": "error",
            "message": "Invalid query",
            "details": "The query must be a non-empty string."
        })

    try:
        n_results = max(1, min(int(n_results), 10))
    except (TypeError, ValueError):
        n_results = 3

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

        results = await asyncio.wait_for(
            vector_memory.query_memory_async(query, n_results=n_results),
            timeout=timeout_seconds,
        )

        if not results:
            return json.dumps({
                "status": "success",
                "message": "No relevant archival memory found.",
                "results": []
            })

        return json.dumps({
            "status": "success",
            "results": [
                {
                    "document": result.get("document", ""),
                    "metadata": result.get("metadata", {}),
                    "distance": result.get("distance"),
                }
                for result in results
            ]
        }, indent=2)
    except asyncio.TimeoutError:
        return json.dumps({
            "status": "error",
            "message": f"Archival search timed out after {timeout_seconds}s."
        })
    except Exception as exc:
        logger.error(f"search_archival_memory error: {exc}", exc_info=True)
        return json.dumps({
            "status": "error",
            "message": "Could not search archival memory",
            "details": str(exc)
        })
    finally:
        if owns_instance and vector_memory is not None:
            try:
                vector_memory.close()
            except Exception:
                pass
