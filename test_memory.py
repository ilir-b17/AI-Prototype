"""
Integration Test Script for Memory Architecture (Hippocampus).

This script tests both the vector database (long-term semantic memory) and
the ledger database (short-term operational state) to verify the memory
architecture is functioning correctly.

Run this script to validate the memory layer before integrating with the
main bot interface.
"""

import sys
import logging
from pprint import pprint

# Configure logging for the test
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import memory modules
from src.memory.vector_db import VectorMemory
from src.memory.ledger_db import LedgerMemory, LogLevel, TaskStatus

# Use ASCII-safe symbols for cross-platform compatibility
CHECK = "[PASS]"
FAIL = "[FAIL]"


def test_vector_memory() -> bool:
    """
    Test the VectorMemory (ChromaDB) functionality.

    Returns:
        bool: True if all tests pass, False otherwise.
    """
    print("\n" + "=" * 80)
    print("TESTING VECTOR MEMORY (ChromaDB - Long-Term Semantic Memory)")
    print("=" * 80)

    try:
        # Initialize VectorMemory
        logger.info("Initializing VectorMemory...")
        vector_mem = VectorMemory(persist_dir="data/chroma_storage")
        print(f"{CHECK} VectorMemory initialized successfully")

        # Test 1: Add memories
        print("\n--- Test 1: Adding Memories ---")
        memory_1_id = vector_mem.add_memory(
            text="The user asked me about the weather and I provided a forecast.",
            metadata={"type": "observation", "context": "user_interaction", "timestamp": "2026-04-14T10:30:00"}
        )
        print(f"{CHECK} Added memory 1 with ID: {memory_1_id}")

        memory_2_id = vector_mem.add_memory(
            text="System initialized and connected to Telegram API successfully.",
            metadata={"type": "system_event", "context": "initialization", "importance": "high"}
        )
        print(f"{CHECK} Added memory 2 with ID: {memory_2_id}")

        memory_3_id = vector_mem.add_memory(
            text="The user requested information about AI capabilities and memory architecture.",
            metadata={"type": "observation", "context": "user_interaction", "timestamp": "2026-04-14T10:45:00"}
        )
        print(f"{CHECK} Added memory 3 with ID: {memory_3_id}")

        # Test 2: Query memories
        print("\n--- Test 2: Querying Similar Memories ---")
        query_results = vector_mem.query_memory(
            query_text="What did the user ask about?",
            n_results=2
        )
        print(f"{CHECK} Query returned {len(query_results)} results:")
        for i, result in enumerate(query_results, 1):
            print(f"\n  Result {i}:")
            print(f"    ID: {result['id']}")
            print(f"    Distance: {result['distance']:.4f}")
            print(f"    Text: {result['document']}")
            print(f"    Metadata: {result['metadata']}")

        # Test 3: Get memory count
        print("\n--- Test 3: Memory Statistics ---")
        total_count = vector_mem.get_memory_count()
        print(f"{CHECK} Total memories stored: {total_count}")

        print(f"\n{CHECK} VectorMemory tests PASSED")
        return True

    except Exception as e:
        logger.error(f"VectorMemory test failed: {e}")
        print(f"\n{FAIL} VectorMemory test FAILED: {e}")
        return False


import concurrent.futures
import threading

def test_ledger_memory_concurrency() -> bool:
    """
    Stress test LedgerMemory for multi-threading safety without locking errors.
    """
    print("\n" + "=" * 80)
    print("TESTING LEDGER MEMORY CONCURRENCY (Threading/WAL Mode)")
    print("=" * 80)

    try:
        ledger_mem = LedgerMemory(db_path="data/ledger_stress.db")
        print(f"{CHECK} LedgerMemory (Stress) initialized successfully")

        def worker_task(thread_id: int, iterations: int):
            for i in range(iterations):
                # Write operation
                task_id = ledger_mem.add_task(
                    task_description=f"Task {i} from thread {thread_id}",
                    priority=max(1, thread_id),
                    status="PENDING"
                )
                # Read operation
                _ = ledger_mem.get_pending_tasks(limit=5)
                # Update operation
                ledger_mem.update_task_status(task_id, "COMPLETED")
                # Log operation
                ledger_mem.log_event(
                    log_level=LogLevel.INFO,
                    message=f"Thread {thread_id} completed iteration {i}"
                )
            # Ensure the connection is closed for this thread
            ledger_mem.close()

        num_threads = 10
        iterations_per_thread = 20

        print(f"\n--- Running {num_threads} threads with {iterations_per_thread} iterations each ---")

        with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = [
                executor.submit(worker_task, t_id, iterations_per_thread)
                for t_id in range(num_threads)
            ]

            # Wait for all to complete
            for future in concurrent.futures.as_completed(futures):
                future.result() # Will raise any exception caught in thread

        print(f"{CHECK} All threads completed without OperationalError or locking issues.")

        # Verify total logs and tasks
        total_logs = len(ledger_mem.get_logs(limit=1000))
        # Depending on when logs are flushed, it should be at least num_threads * iterations
        if total_logs >= num_threads * iterations_per_thread:
            print(f"{CHECK} Data verification passed (Logs count: {total_logs})")
        else:
            print(f"{FAIL} Data verification failed (Logs count: {total_logs})")
            return False

        return True
    except Exception as e:
        logger.error(f"LedgerMemory concurrency test failed: {e}")
        print(f"\n{FAIL} LedgerMemory concurrency test FAILED: {e}")
        return False

def test_ledger_memory() -> bool:
    """
    Test the LedgerMemory (SQLite) functionality.

    Returns:
        bool: True if all tests pass, False otherwise.
    """
    print("\n" + "=" * 80)
    print("TESTING LEDGER MEMORY (SQLite - Short-Term Operational State)")
    print("=" * 80)

    try:
        # Initialize LedgerMemory
        logger.info("Initializing LedgerMemory...")
        ledger_mem = LedgerMemory(db_path="data/ledger.db")
        print(f"{CHECK} LedgerMemory initialized successfully")

        # Test 1: Add tasks
        print("\n--- Test 1: Adding Tasks ---")
        task_1_id = ledger_mem.add_task(
            task_description="Process user input from Telegram",
            priority=1,
            status="PENDING"
        )
        print(f"{CHECK} Added task 1 (High Priority): ID {task_1_id}")

        task_2_id = ledger_mem.add_task(
            task_description="Query vector memory for relevant context",
            priority=3,
            status="PENDING"
        )
        print(f"{CHECK} Added task 2 (Medium Priority): ID {task_2_id}")

        task_3_id = ledger_mem.add_task(
            task_description="Generate response using LLM",
            priority=2,
            status="PENDING"
        )
        print(f"{CHECK} Added task 3 (Medium-High Priority): ID {task_3_id}")

        # Test 2: Retrieve pending tasks
        print("\n--- Test 2: Retrieving Pending Tasks ---")
        pending_tasks = ledger_mem.get_pending_tasks(order_by_priority=True)
        print(f"{CHECK} Retrieved {len(pending_tasks)} pending tasks (ordered by priority):")
        for i, task in enumerate(pending_tasks, 1):
            print(f"\n  Task {i}:")
            print(f"    ID: {task['id']}")
            print(f"    Priority: {task['priority']}")
            print(f"    Description: {task['task_description']}")
            print(f"    Status: {task['status']}")
            print(f"    Created: {task['created_at']}")

        # Test 3: Update task status
        print("\n--- Test 3: Updating Task Status ---")
        success = ledger_mem.update_task_status(task_1_id, "IN_PROGRESS")
        print(f"{CHECK} Task {task_1_id} updated to IN_PROGRESS: {success}")

        success = ledger_mem.update_task_status(task_2_id, "COMPLETED")
        print(f"{CHECK} Task {task_2_id} updated to COMPLETED: {success}")

        # Test 4: Log events
        print("\n--- Test 4: Logging Events ---")
        log_1_id = ledger_mem.log_event(
            log_level=LogLevel.INFO,
            message="Bot started and connected to Telegram",
            context={"event": "startup", "connection": "telegram"}
        )
        print(f"{CHECK} Logged INFO event: {log_1_id}")

        log_2_id = ledger_mem.log_event(
            log_level=LogLevel.DEBUG,
            message="Memory systems initialized",
            context={"components": ["VectorMemory", "LedgerMemory"]}
        )
        print(f"{CHECK} Logged DEBUG event: {log_2_id}")

        log_3_id = ledger_mem.log_event(
            log_level=LogLevel.WARNING,
            message="High task queue depth detected",
            context={"queue_size": 3, "recommendation": "increase_processing_rate"}
        )
        print(f"{CHECK} Logged WARNING event: {log_3_id}")

        # Test 5: Retrieve logs
        print("\n--- Test 5: Retrieving System Logs ---")
        all_logs = ledger_mem.get_logs(limit=10)
        print(f"{CHECK} Retrieved {len(all_logs)} recent log entries:")
        for i, log in enumerate(all_logs, 1):
            print(f"\n  Log {i}:")
            print(f"    ID: {log['id']}")
            print(f"    Level: {log['log_level']}")
            print(f"    Message: {log['message']}")
            print(f"    Timestamp: {log['timestamp']}")
            if log['context']:
                print(f"    Context: {log['context']}")

        # Test 6: Retrieve filtered logs
        print("\n--- Test 6: Filtering Logs by Level ---")
        info_logs = ledger_mem.get_logs(log_level=LogLevel.INFO, limit=5)
        print(f"{CHECK} Retrieved {len(info_logs)} INFO level logs")

        # Test 7: Close connection
        print("\n--- Test 7: Closing Database Connection ---")
        ledger_mem.close()
        print(f"{CHECK} LedgerMemory connection closed gracefully")

        print(f"\n{CHECK} LedgerMemory tests PASSED")
        return True

    except Exception as e:
        logger.error(f"LedgerMemory test failed: {e}")
        print(f"\n{FAIL} LedgerMemory test FAILED: {e}")
        return False


def main() -> int:
    """
    Run all integration tests for the memory architecture.

    Returns:
        int: Exit code (0 for success, 1 for failure).
    """
    print("\n" + "=" * 80)
    print("MEMORY ARCHITECTURE INTEGRATION TEST")
    print("Autonomous Biomimetic AI Agent - Sprint 2")
    print("=" * 80)

    try:
        # Run VectorMemory tests
        vector_passed = test_vector_memory()

        # Run LedgerMemory tests
        ledger_passed = test_ledger_memory()

        # Run concurrency stress test
        concurrency_passed = test_ledger_memory_concurrency()

        # Summary
        print("\n" + "=" * 80)
        print("TEST SUMMARY")
        print("=" * 80)
        print(f"VectorMemory (ChromaDB):  {'PASSED' if vector_passed else 'FAILED'}")
        print(f"LedgerMemory (SQLite):    {'PASSED' if ledger_passed else 'FAILED'}")
        print(f"LedgerMemory Concurrency: {'PASSED' if concurrency_passed else 'FAILED'}")

        if vector_passed and ledger_passed and concurrency_passed:
            print(f"\n{CHECK} All memory architecture tests PASSED!")
            print("The Hippocampus is ready for integration with the Telegram interface.")
            return 0
        else:
            print(f"\n{FAIL} Some tests FAILED. Please review the output above.")
            return 1

    except Exception as e:
        logger.error(f"Test suite failed: {e}")
        print(f"\n{FAIL} Test suite encountered a critical error: {e}")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
