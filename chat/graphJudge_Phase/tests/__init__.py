"""
Test Suite for GraphJudge Phase Modular System

This package contains comprehensive unit tests for the modularized
GraphJudge system, ensuring compatibility with the original run_gj.py
functionality while testing each component independently.

Test Coverage:
- Core graph judgment functionality
- Perplexity API integration
- Gold label bootstrapping
- Processing pipeline
- Prompt engineering
- Data structures
- Utilities and validation
- Logging system
- Configuration management
"""

__version__ = "1.0.0"
__author__ = "GraphJudge Team"

# Test utilities and common fixtures
from .conftest import (
    PerplexityTestBase,
    mock_perplexity_response,
    sample_test_data
)

__all__ = [
    'PerplexityTestBase',
    'mock_perplexity_response', 
    'sample_test_data'
]
