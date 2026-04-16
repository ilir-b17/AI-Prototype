import asyncio
import logging

logger = logging.getLogger(__name__)


def _sync_search_archival_memory(query: str) -> str:
    vector_memory = None
    try:
        from src.memory.vector_db import VectorMemory
        vector_memory = VectorMemory()
        results = vector_memory.query_memory(query, n_results=3)
        if not results:
            return "No relevant archival memory found."
        lines = ["Archival Results:"]
        for i, res in enumerate(results, 1):
            lines.append(f"{i}. {res['document']}")
        return "\n".join(lines)
    except Exception as exc:
        return f"Error: Could not search archival memory due to [{exc}]."


async def search_archival_memory(query: str) -> str:
    logger.info(f"search_archival_memory: {query}")
    return await asyncio.to_thread(_sync_search_archival_memory, query)
