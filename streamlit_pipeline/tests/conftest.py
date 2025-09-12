"""
Global pytest configuration for GraphJudge Streamlit Pipeline tests.

This module provides shared fixtures, configuration, and test utilities
that are available to all test modules.

Following TDD principles from docs/Testing_Demands.md:
- Provide consistent test environment setup
- Enable proper mocking and isolation 
- Support async test execution
- Facilitate performance testing
"""

import pytest
import asyncio
import sys
import os
from pathlib import Path
from unittest.mock import Mock, AsyncMock, patch
from typing import Generator, Dict, Any

# Add project root to path for proper imports
project_root = Path(__file__).parent.parent
tests_dir = Path(__file__).parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(tests_dir))

from fixtures.api_fixtures import (
    ScenarioFixtures, GPT5MiniFixtures, PerplexityFixtures,
    create_mock_response, STANDARD_TEST_SCENARIOS
)
from test_utils import (
    MockAPIClient, create_mock_gpt5_client, create_mock_perplexity_client,
    TestDataGenerator, ErrorSimulator, PerformanceTestUtils
)


# =============================================================================
# PYTEST CONFIGURATION
# =============================================================================

def pytest_configure(config):
    """Configure pytest with custom markers and settings."""
    config.addinivalue_line(
        "markers", "unit: Unit tests for individual components"
    )
    config.addinivalue_line(
        "markers", "integration: Integration tests for component interactions"
    )
    config.addinivalue_line(
        "markers", "api: Tests involving API calls (use with mocks)"
    )
    config.addinivalue_line(
        "markers", "slow: Tests that take longer to run"
    )
    config.addinivalue_line(
        "markers", "smoke: Basic functionality smoke tests"
    )
    config.addinivalue_line(
        "markers", "e2e: End-to-end tests with real API calls"
    )
    config.addinivalue_line(
        "markers", "mock_api: Tests using mocked API responses"
    )
    config.addinivalue_line(
        "markers", "performance: Performance and load tests"
    )


def pytest_collection_modifyitems(config, items):
    """Modify test collection to handle markers and dependencies."""
    # Skip slow tests by default unless explicitly requested
    if not config.getoption("--slow"):
        skip_slow = pytest.mark.skip(reason="Slow test - use --slow to run")
        for item in items:
            if "slow" in item.keywords:
                item.add_marker(skip_slow)
    
    # Skip E2E tests by default unless explicitly requested
    if not config.getoption("--e2e"):
        skip_e2e = pytest.mark.skip(reason="E2E test - use --e2e to run")
        for item in items:
            if "e2e" in item.keywords:
                item.add_marker(skip_e2e)


def pytest_addoption(parser):
    """Add custom command line options."""
    parser.addoption(
        "--slow", action="store_true", default=False, help="Run slow tests"
    )
    parser.addoption(
        "--e2e", action="store_true", default=False, help="Run end-to-end tests"
    )
    parser.addoption(
        "--performance", action="store_true", default=False, help="Run performance tests"
    )
    parser.addoption(
        "--api-key", action="store", default=None, help="API key for E2E tests"
    )


# =============================================================================
# EVENT LOOP AND ASYNC FIXTURES
# =============================================================================

@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
async def async_test_client():
    """Provide async test client for testing async operations."""
    client = AsyncMock()
    return client


# =============================================================================
# MOCK API FIXTURES
# =============================================================================

@pytest.fixture
def mock_gpt5_client():
    """Provide mock GPT-5-mini client with realistic responses."""
    return create_mock_gpt5_client("DREAM_OF_RED_CHAMBER")


@pytest.fixture
def mock_perplexity_client():
    """Provide mock Perplexity client with realistic responses."""
    return create_mock_perplexity_client("DREAM_OF_RED_CHAMBER")


@pytest.fixture
def mock_api_clients():
    """Provide both mock API clients."""
    return {
        "gpt5": create_mock_gpt5_client(),
        "perplexity": create_mock_perplexity_client()
    }


@pytest.fixture(params=[s[0] for s in STANDARD_TEST_SCENARIOS])
def scenario_fixture(request):
    """Parametrized fixture providing different test scenarios."""
    scenario_name = request.param
    return getattr(ScenarioFixtures, scenario_name)


@pytest.fixture
def api_error_responses():
    """Provide various API error response fixtures."""
    return {
        "rate_limit": GPT5MiniFixtures.rate_limit_error(),
        "auth_error": GPT5MiniFixtures.invalid_api_key_error(),
        "server_error": PerplexityFixtures.server_error(),
        "timeout": PerplexityFixtures.timeout_error(),
        "malformed": GPT5MiniFixtures.malformed_response()
    }


# =============================================================================
# TEST DATA FIXTURES
# =============================================================================

@pytest.fixture
def test_data_generator():
    """Provide test data generator utility."""
    return TestDataGenerator()


@pytest.fixture
def sample_entities():
    """Provide sample entity list for testing."""
    return ["林黛玉", "賈寶玉", "榮國府", "賈母", "碧紗櫥"]


@pytest.fixture
def sample_triples(test_data_generator):
    """Provide sample triples for testing."""
    return [
        test_data_generator.create_test_triple("林黛玉", "居住於", "榮國府"),
        test_data_generator.create_test_triple("賈寶玉", "相遇", "林黛玉"),
        test_data_generator.create_test_triple("賈母", "疼愛", "林黛玉")
    ]


@pytest.fixture
def large_test_dataset(test_data_generator):
    """Provide large dataset for performance testing."""
    return test_data_generator.create_large_test_dataset(size=50)


# =============================================================================
# PERFORMANCE AND ERROR TESTING FIXTURES
# =============================================================================

@pytest.fixture
def performance_tracker():
    """Provide performance testing utilities."""
    return PerformanceTestUtils()


@pytest.fixture
def error_simulator():
    """Provide error simulation utilities."""
    return ErrorSimulator()


@pytest.fixture
def timing_context():
    """Provide timing context for performance tests."""
    import time
    
    class TimingContext:
        def __init__(self):
            self.start_time = None
            self.end_time = None
        
        def __enter__(self):
            self.start_time = time.time()
            return self
        
        def __exit__(self, exc_type, exc_val, exc_tb):
            self.end_time = time.time()
        
        @property
        def elapsed(self):
            if self.start_time is None or self.end_time is None:
                return 0.0
            return self.end_time - self.start_time
    
    return TimingContext()


# =============================================================================
# ENVIRONMENT AND CONFIGURATION FIXTURES
# =============================================================================

@pytest.fixture
def clean_environment():
    """Provide clean environment for config testing."""
    # Store original environment
    original_env = os.environ.copy()
    
    # Clear API-related environment variables
    api_vars = [
        'OPENAI_API_KEY', 'AZURE_OPENAI_KEY', 'AZURE_OPENAI_ENDPOINT',
        'OPENAI_API_BASE', 'PERPLEXITY_API_KEY'
    ]
    
    for var in api_vars:
        if var in os.environ:
            del os.environ[var]
    
    yield
    
    # Restore original environment
    os.environ.clear()
    os.environ.update(original_env)


@pytest.fixture
def temp_config_file():
    """Provide temporary configuration file for testing."""
    import tempfile
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.env', delete=False) as f:
        f.write("""
# Test configuration
OPENAI_API_KEY=test-key-12345
AZURE_OPENAI_ENDPOINT=https://test.openai.azure.com/
PERPLEXITY_API_KEY=test-perplexity-key
""")
        temp_path = Path(f.name)
    
    yield temp_path
    
    # Cleanup
    if temp_path.exists():
        temp_path.unlink()


# =============================================================================
# VALIDATION AND CONSISTENCY FIXTURES
# =============================================================================

@pytest.fixture
def validation_test_cases():
    """Provide validation test cases for input testing."""
    return [
        # Valid cases
        {"input": "這是一個有效的中文測試文本。", "expected": "valid"},
        {"input": "John works at Google in San Francisco.", "expected": "valid"},
        
        # Edge cases
        {"input": "短", "expected": "too_short"},
        {"input": "A" * 60000, "expected": "too_long"},
        {"input": "   \n\t   ", "expected": "empty"},
        {"input": "", "expected": "empty"},
        
        # Special cases
        {"input": "重複" * 1000, "expected": "repetitive"},
        {"input": "Line 1\n" + "X" * 2000 + "\nLine 3", "expected": "long_lines"}
    ]


@pytest.fixture
def integration_test_pipeline():
    """Provide pipeline components for integration testing."""
    
    class MockPipeline:
        def __init__(self):
            self.entity_processor = Mock()
            self.triple_generator = Mock()
            self.graph_judge = Mock()
            self.results = {}
        
        async def run_entity_extraction(self, text: str):
            """Mock entity extraction."""
            result = Mock()
            result.success = True
            result.entities = ["實體1", "實體2"]
            result.denoised_text = "處理後的文本"
            result.processing_time = 1.0
            self.results["entities"] = result
            return result
        
        async def run_triple_generation(self, entities, text):
            """Mock triple generation."""
            result = Mock()
            result.success = True
            result.triples = [Mock(), Mock()]
            result.metadata = {"count": 2}
            result.processing_time = 2.0
            self.results["triples"] = result
            return result
        
        async def run_graph_judgment(self, triples):
            """Mock graph judgment."""
            result = Mock()
            result.success = True
            result.judgments = [True, False]
            result.confidence = [0.9, 0.7]
            result.processing_time = 1.5
            self.results["judgments"] = result
            return result
    
    return MockPipeline()


# =============================================================================
# CLEANUP AND TEARDOWN
# =============================================================================

@pytest.fixture(autouse=True)
def cleanup_after_test():
    """Auto-cleanup after each test."""
    yield
    
    # Clear any lingering mocks or patches
    patch.stopall()
    
    # Reset asyncio event loop policy if needed
    try:
        asyncio.set_event_loop_policy(None)
    except:
        pass


# =============================================================================
# TEST REPORTING AND LOGGING
# =============================================================================

@pytest.fixture(scope="session", autouse=True)
def test_session_setup():
    """Set up test session with reporting."""
    print("\n=== GraphJudge Streamlit Pipeline Test Session ===")
    print("Test framework: pytest with asyncio support")
    print("Mock API responses: enabled")
    print("Performance tracking: enabled")
    print("Error simulation: enabled")
    print("=" * 50)
    
    yield
    
    print("\n=== Test Session Complete ===")


def pytest_report_teststatus(report, config):
    """Customize test status reporting."""
    if report.when == "call":
        if report.outcome == "passed":
            return report.outcome, "✓", f"PASSED: {report.nodeid}"
        elif report.outcome == "failed":
            return report.outcome, "✗", f"FAILED: {report.nodeid}"
        elif report.outcome == "skipped":
            return report.outcome, "↷", f"SKIPPED: {report.nodeid}"