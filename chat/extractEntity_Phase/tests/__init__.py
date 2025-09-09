"""
Test Suite Module

This module contains comprehensive tests for the ECTD pipeline:
- unit/: Unit tests for individual modules (95%+ coverage required)
- integration/: Integration tests for module interactions
- fixtures/: Test data and mock responses

All tests follow TDD principles and use realistic Chinese text examples
for comprehensive validation of Chinese language processing capabilities.

Test Modules:
- test_rate_limiter.py: Tests for rate limiting and token tracking
- test_cache_manager.py: Tests for disk caching functionality
- test_gpt5mini_client.py: Tests for GPT-5-mini API client
- test_config.py: Tests for configuration management
- test_logging.py: Tests for logging infrastructure
- test_entities.py: Tests for entity data models
- test_chinese_text.py: Tests for Chinese text utilities
"""

# Test suite initialization
# Tests are implemented following TDD principles with comprehensive coverage

# Import test modules for discovery
# Commented out to avoid module import issues during pytest discovery
# from . import test_rate_limiter
# from . import test_cache_manager
# from . import test_gpt5mini_client
# from . import test_config
# from . import test_logging
# from . import test_entities
# from . import test_chinese_text

__all__ = [
    'test_rate_limiter',
    'test_cache_manager', 
    'test_gpt5mini_client',
    'test_config',
    'test_logging',
    'test_entities',
    'test_chinese_text'
]
