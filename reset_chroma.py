#!/usr/bin/env python3
"""
ChromaDB Recovery Script

This script resets the ChromaDB storage to recover from collection corruption.
Run this if you encounter StopIteration errors when adding memories.
"""

import os
import shutil
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def reset_chroma_storage(storage_path: str = "data/chroma_storage") -> None:
    """
    Reset ChromaDB storage by removing and recreating the directory.
    
    Args:
        storage_path: Path to the ChromaDB storage directory.
    """
    if os.path.exists(storage_path):
        logger.info(f"Removing corrupted ChromaDB storage at {storage_path}...")
        try:
            shutil.rmtree(storage_path)
            logger.info("✓ Corrupted storage removed successfully")
        except Exception as e:
            logger.error(f"Failed to remove storage: {e}")
            return
    
    # Create fresh directory
    os.makedirs(storage_path, exist_ok=True)
    logger.info(f"✓ Fresh ChromaDB storage directory created at {storage_path}")
    
    logger.info("\n✓ ChromaDB has been reset successfully!")
    logger.info("The next run of main.py will initialize a fresh collection.")
    logger.info("\nTo test, run: python main.py")

if __name__ == "__main__":
    logger.info("=" * 60)
    logger.info("ChromaDB Recovery Script")
    logger.info("=" * 60)
    logger.info("\nThis will reset ChromaDB by removing corrupted storage.")
    logger.info("All previously stored memories will be cleared.\n")
    
    logger.warning("WARNING: This will permanently delete all long-term vector memories!")
    response = input("Type 'RESET' to confirm complete ChromaDB wipe: ").strip()
    if response == "RESET":
        reset_chroma_storage()
    else:
        logger.info("Reset cancelled (confirmation phrase did not match).")
