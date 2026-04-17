# tests/conftest.py
# Shared fixtures and pytest-asyncio configuration.
import sys
import os

# Ensure the project root is on sys.path so `src.*` imports resolve.
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
