"""
Vector Database Module for Long-Term Memory (The Hippocampus).

This module provides semantic vector storage for contextual memory using ChromaDB.
Memories are stored with embeddings, enabling similarity-based retrieval for
contextual awareness in agent decision-making.
"""

import os
import logging
import uuid
from typing import Dict, List, Any, Optional
import chromadb
from chromadb.config import Settings

logger = logging.getLogger(__name__)


class VectorMemory:
    """
    A wrapper around ChromaDB for managing long-term semantic memory.

    This class provides methods to store and retrieve memories based on semantic
    similarity. It uses ChromaDB's default embedding model for vector computation.
    """

    def __init__(self, persist_dir: str = "data/chroma_storage") -> None:
        """
        Initialize the VectorMemory instance.

        Args:
            persist_dir: Path where ChromaDB data will be persisted. Defaults to
                        "data/chroma_storage".

        Raises:
            Exception: If ChromaDB initialization fails.
        """
        self.persist_dir = persist_dir
        self.collection = None

        # Ensure the persistence directory exists
        os.makedirs(self.persist_dir, exist_ok=True)

        try:
            # Initialize ChromaDB client with persistence enabled
            # Disable telemetry to avoid capture() errors
            settings = Settings(
                is_persistent=True,
                persist_directory=self.persist_dir,
                anonymized_telemetry=False,
                allow_reset=True,
            )
            self.client = chromadb.Client(settings)

            # Try to get or create the default collection
            try:
                self.collection = self.client.get_or_create_collection(
                    name="agent_memory",
                    metadata={"hnsw:space": "cosine"}
                )
            except Exception as collection_error:
                logger.warning(f"Collection initialization failed: {collection_error}. Attempting recovery...")
                # Try to recover by resetting the collection
                try:
                    self.client.delete_collection(name="agent_memory")
                    logger.info("Deleted corrupted collection")
                except Exception:
                    pass  # Collection may not exist yet

                # Create fresh collection
                self.collection = self.client.create_collection(
                    name="agent_memory",
                    metadata={"hnsw:space": "cosine"}
                )
                logger.info("Created fresh collection after recovery")

            logger.info(f"VectorMemory initialized with persistence at {self.persist_dir}")
        except Exception as e:
            logger.error(f"Failed to initialize VectorMemory: {e}")
            raise

    def add_memory(
        self,
        text: str,
        metadata: Optional[Dict[str, Any]] = None,
        memory_id: Optional[str] = None
    ) -> str:
        """
        Add a semantic memory to the vector database with retry logic.

        Args:
            text: The text content of the memory.
            metadata: Optional dictionary of metadata associated with the memory.
                     Examples: {"type": "observation", "context": "user_interaction"}
            memory_id: Optional custom ID for the memory. If not provided, a UUID
                      will be generated automatically.

        Returns:
            str: The ID of the added memory.

        Raises:
            ValueError: If text is empty or None.
            Exception: If insertion into ChromaDB fails after retries.
        """
        if not text or not isinstance(text, str):
            raise ValueError("Text must be a non-empty string")

        if metadata is None:
            metadata = {}

        # Generate ID if not provided
        if not memory_id:
            memory_id = str(uuid.uuid4())

        # Retry logic for transient failures
        max_retries = 3
        for attempt in range(max_retries):
            try:
                # Add memory to the collection with explicit ID
                self.collection.add(
                    ids=[memory_id],
                    documents=[text],
                    metadatas=[metadata]
                )

                logger.info(f"Memory added with ID: {memory_id}")
                return memory_id
            except Exception as e:
                if attempt < max_retries - 1:
                    logger.warning(f"Failed to add memory (attempt {attempt + 1}/{max_retries}): {e}")
                    # Wait briefly before retry
                    import time
                    time.sleep(0.5)
                else:
                    logger.error(f"Failed to add memory after {max_retries} attempts: {e}")
                    raise

    def query_memory(
        self,
        query_text: str,
        n_results: int = 3
    ) -> List[Dict[str, Any]]:
        """
        Query the vector database for semantically similar memories.

        Args:
            query_text: The query text to search for similar memories.
            n_results: Number of results to retrieve. Defaults to 3.

        Returns:
            List of dictionaries containing:
                - id: Memory ID
                - document: The memory text
                - metadata: Associated metadata
                - distance: Semantic distance (lower = more similar)

        Raises:
            ValueError: If query_text is empty or n_results is less than 1.
            Exception: If the query fails.
        """
        if not query_text or not isinstance(query_text, str):
            raise ValueError("Query text must be a non-empty string")

        if n_results < 1:
            raise ValueError("n_results must be at least 1")

        try:
            # Query the collection
            results = self.collection.query(
                query_texts=[query_text],
                n_results=n_results
            )

            # Format results into a more readable structure
            formatted_results = []
            if results["ids"] and len(results["ids"]) > 0:
                for i, memory_id in enumerate(results["ids"][0]):
                    formatted_results.append({
                        "id": memory_id,
                        "document": results["documents"][0][i],
                        "metadata": results["metadatas"][0][i],
                        "distance": results["distances"][0][i]
                    })

            logger.info(f"Query returned {len(formatted_results)} results")
            return formatted_results
        except Exception as e:
            logger.error(f"Failed to query memory: {e}")
            raise

    def delete_memory(self, memory_id: str) -> None:
        """
        Delete a memory from the vector database by ID.

        Args:
            memory_id: The ID of the memory to delete.

        Raises:
            Exception: If deletion fails.
        """
        try:
            self.collection.delete(ids=[memory_id])
            logger.info(f"Memory {memory_id} deleted")
        except Exception as e:
            logger.error(f"Failed to delete memory {memory_id}: {e}")
            raise

    def get_memory_count(self) -> int:
        """
        Get the total number of memories in the database.

        Returns:
            int: The total count of memories.
        """
        try:
            count = self.collection.count()
            logger.info(f"Total memories in database: {count}")
            return count
        except Exception as e:
            logger.error(f"Failed to retrieve memory count: {e}")
            return 0

    def clear_all_memories(self) -> None:
        """
        Delete all memories from the database.

        WARNING: This operation is irreversible.

        Raises:
            Exception: If clearing fails.
        """
        try:
            # Delete the collection and recreate it
            self.client.delete_collection(name="agent_memory")
            self.collection = self.client.get_or_create_collection(
                name="agent_memory",
                metadata={"hnsw:space": "cosine"}
            )
            logger.warning("All memories cleared from the database")
        except Exception as e:
            logger.error(f"Failed to clear all memories: {e}")
            raise

    def close(self) -> None:
        """
        Close the ChromaDB client and release resources.
        """
        try:
            if hasattr(self, 'client') and self.client:
                # ChromaDB doesn't have an explicit close, but we can set to None
                logger.info("Closing VectorMemory connections")
                self.client = None
        except Exception as e:
            logger.warning(f"Error closing VectorMemory: {e}")
