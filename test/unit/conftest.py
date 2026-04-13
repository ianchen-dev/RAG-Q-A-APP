"""Pytest configuration for unit tests."""

import sys
from unittest.mock import Mock

# Mock problematic modules that cause import errors
# This allows unit tests to run without installing all dependencies

# Mock langchain_mcp_adapters
sys.modules['langchain_mcp_adapters'] = Mock()
sys.modules['langchain_mcp_adapters.client'] = Mock()

# Mock unstructured and its dependencies (if needed)
try:
    import magic
except (ImportError, OSError):
    # magic module may fail to load on Windows
    sys.modules['magic'] = Mock()
    sys.modules['magic.compat'] = Mock()
