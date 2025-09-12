"""
Test utilities for GraphJudge Streamlit Pipeline testing.

This module provides common testing patterns, utilities, and helpers
following the TDD principles outlined in docs/Testing_Demands.md.

Key features:
- Mock API client creation
- Async test utilities
- Performance testing helpers
- Error simulation utilities
- Test data generation
"""

import asyncio
import time
import json
from typing import Dict, Any, List, Optional, Callable, Union
from unittest.mock import Mock, AsyncMock, patch, MagicMock
import pytest
from dataclasses import dataclass
from contextlib import asynccontextmanager

from core.models import Triple, EntityResult, TripleResult, JudgmentResult
from fixtures.api_fixtures import (
    GPT5MiniFixtures, PerplexityFixtures, ScenarioFixtures,
    create_mock_response, STANDARD_TEST_SCENARIOS
)


# =============================================================================
# MOCK API CLIENT UTILITIES
# =============================================================================

class MockAPIClient:
    """Mock API client for testing pipeline components."""
    
    def __init__(self, responses: Optional[Dict[str, Any]] = None):
        """
        Initialize mock API client.
        
        Args:
            responses: Dictionary mapping method names to response data
        """
        self.responses = responses or {}
        self.call_history = []
        self.call_count = 0
        self.should_fail = False
        self.failure_reason = None
    
    async def make_request(self, method: str, **kwargs) -> Dict[str, Any]:
        """Mock API request method."""
        self.call_count += 1
        self.call_history.append({
            'method': method,
            'kwargs': kwargs,
            'timestamp': time.time()
        })
        
        if self.should_fail:
            if isinstance(self.failure_reason, Exception):
                raise self.failure_reason
            else:
                raise Exception(self.failure_reason or "Mock API failure")
        
        if method in self.responses:
            response = self.responses[method]
            if callable(response):
                return response(**kwargs)
            return response
        
        return {"mock": True, "method": method, "args": kwargs}
    
    def set_response(self, method: str, response: Union[Dict, Callable]):
        """Set response for a specific method."""
        self.responses[method] = response
    
    def set_failure(self, reason: Union[str, Exception]):
        """Make the client fail on next request."""
        self.should_fail = True
        self.failure_reason = reason
    
    def reset(self):
        """Reset the mock client state."""
        self.call_history = []
        self.call_count = 0
        self.should_fail = False
        self.failure_reason = None


def create_mock_gpt5_client(scenario: str = "DREAM_OF_RED_CHAMBER") -> MockAPIClient:
    """Create a mock GPT-5-mini client with predefined responses."""
    scenario_data = getattr(ScenarioFixtures, scenario)
    
    client = MockAPIClient()
    client.set_response(
        "extract_entities",
        lambda **kwargs: GPT5MiniFixtures.successful_entity_response(scenario_data["entities"])
    )
    client.set_response(
        "denoise_text", 
        lambda **kwargs: GPT5MiniFixtures.successful_denoising_response(scenario_data["denoised_text"])
    )
    
    return client


def create_mock_perplexity_client(scenario: str = "DREAM_OF_RED_CHAMBER") -> MockAPIClient:
    """Create a mock Perplexity client with predefined responses."""
    scenario_data = getattr(ScenarioFixtures, scenario)
    
    client = MockAPIClient()
    judgments = [
        {
            "judgment": j["judgment"],
            "confidence": j["confidence"], 
            "triple_index": j["triple_index"]
        }
        for j in scenario_data["judgments"]
    ]
    
    client.set_response(
        "judge_triples",
        lambda **kwargs: PerplexityFixtures.successful_judgment_response(judgments)
    )
    
    return client


# =============================================================================
# ASYNC TEST UTILITIES  
# =============================================================================

class AsyncTestUtils:
    """Utilities for testing async operations."""
    
    @staticmethod
    async def run_with_timeout(coro, timeout: float = 5.0):
        """Run coroutine with timeout."""
        try:
            return await asyncio.wait_for(coro, timeout=timeout)
        except asyncio.TimeoutError:
            pytest.fail(f"Operation timed out after {timeout} seconds")
    
    @staticmethod
    @asynccontextmanager
    async def time_operation():
        """Context manager to time async operations."""
        start = time.time()
        try:
            yield
        finally:
            end = time.time()
            # Store timing in context for assertions
            time_operation.elapsed = end - start
    
    @staticmethod
    def create_async_mock(return_value=None, side_effect=None):
        """Create an async mock with proper await behavior."""
        mock = AsyncMock()
        if return_value is not None:
            mock.return_value = return_value
        if side_effect is not None:
            mock.side_effect = side_effect
        return mock


# =============================================================================
# PERFORMANCE TESTING UTILITIES
# =============================================================================

@dataclass
class PerformanceMetrics:
    """Performance metrics for testing."""
    execution_time: float
    memory_usage: Optional[int] = None
    api_calls: int = 0
    cache_hits: int = 0
    cache_misses: int = 0


class PerformanceTestUtils:
    """Utilities for performance testing."""
    
    @staticmethod
    def time_function(func: Callable, *args, **kwargs) -> tuple:
        """Time function execution and return result with metrics."""
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        
        metrics = PerformanceMetrics(
            execution_time=end_time - start_time
        )
        
        return result, metrics
    
    @staticmethod
    async def time_async_function(coro) -> tuple:
        """Time async function execution."""
        start_time = time.time()
        result = await coro
        end_time = time.time()
        
        metrics = PerformanceMetrics(
            execution_time=end_time - start_time
        )
        
        return result, metrics
    
    @staticmethod
    def assert_performance(metrics: PerformanceMetrics, max_time: float):
        """Assert performance meets requirements."""
        assert metrics.execution_time < max_time, (
            f"Execution took {metrics.execution_time:.3f}s, "
            f"expected < {max_time}s"
        )


# =============================================================================
# ERROR SIMULATION UTILITIES
# =============================================================================

class ErrorSimulator:
    """Utilities for simulating various error conditions."""
    
    @staticmethod
    def simulate_api_errors():
        """Return various API error scenarios."""
        return {
            "rate_limit": GPT5MiniFixtures.rate_limit_error(),
            "auth_error": GPT5MiniFixtures.invalid_api_key_error(),
            "server_error": PerplexityFixtures.server_error(),
            "timeout": PerplexityFixtures.timeout_error(),
            "malformed": GPT5MiniFixtures.malformed_response()
        }
    
    @staticmethod
    def create_failing_mock(error_type: str = "generic"):
        """Create a mock that fails with specific error."""
        error_map = {
            "generic": Exception("Generic test error"),
            "timeout": asyncio.TimeoutError("Request timed out"),
            "connection": ConnectionError("Unable to connect to API"),
            "json": json.JSONDecodeError("Invalid JSON", "", 0)
        }
        
        mock = Mock()
        mock.side_effect = error_map.get(error_type, Exception("Unknown error"))
        return mock
    
    @staticmethod
    def create_intermittent_mock(success_responses: List[Any], 
                                failure_responses: List[Exception],
                                pattern: str = "SFFF"):
        """
        Create a mock that alternates between success and failure.
        
        Args:
            success_responses: List of successful responses
            failure_responses: List of exceptions to raise
            pattern: Pattern like "SFFF" (Success, Fail, Fail, Fail)
        """
        responses = []
        success_idx = 0
        failure_idx = 0
        
        for char in pattern:
            if char.upper() == 'S':
                responses.append(success_responses[success_idx % len(success_responses)])
                success_idx += 1
            else:  # 'F' or any other char
                responses.append(failure_responses[failure_idx % len(failure_responses)])
                failure_idx += 1
        
        mock = Mock()
        mock.side_effect = responses
        return mock


# =============================================================================
# TEST DATA GENERATION
# =============================================================================

class TestDataGenerator:
    """Generate test data for various scenarios."""
    
    @staticmethod
    def create_test_triple(subject: str = "测试主语", 
                          predicate: str = "测试谓语",
                          object: str = "测试宾语",
                          confidence: Optional[float] = 0.85) -> Triple:
        """Create a test Triple object."""
        return Triple(
            subject=subject,
            predicate=predicate,
            object=object,
            confidence=confidence
        )
    
    @staticmethod
    def create_test_entity_result(entities: Optional[List[str]] = None,
                                 denoised_text: str = "测试去噪文本",
                                 success: bool = True,
                                 processing_time: float = 1.5,
                                 error: Optional[str] = None) -> EntityResult:
        """Create a test EntityResult object."""
        if entities is None:
            entities = ["实体1", "实体2", "实体3"]
        
        return EntityResult(
            entities=entities,
            denoised_text=denoised_text,
            success=success,
            processing_time=processing_time,
            error=error
        )
    
    @staticmethod
    def create_test_triple_result(triples: Optional[List[Triple]] = None,
                                 success: bool = True,
                                 processing_time: float = 2.0,
                                 error: Optional[str] = None) -> TripleResult:
        """Create a test TripleResult object."""
        if triples is None:
            triples = [
                TestDataGenerator.create_test_triple("A", "关系1", "B"),
                TestDataGenerator.create_test_triple("B", "关系2", "C")
            ]
        
        return TripleResult(
            triples=triples,
            metadata={"total_chunks": 1, "validation_passed": True},
            success=success,
            processing_time=processing_time,
            error=error
        )
    
    @staticmethod
    def create_test_judgment_result(num_triples: int = 2,
                                   success: bool = True,
                                   processing_time: float = 1.8,
                                   error: Optional[str] = None,
                                   include_explanations: bool = False) -> JudgmentResult:
        """Create a test JudgmentResult object."""
        judgments = [True] * num_triples
        confidence = [0.9, 0.8, 0.85, 0.92, 0.88][:num_triples]
        explanations = None
        
        if include_explanations:
            explanations = [f"解释 {i+1}" for i in range(num_triples)]
        
        return JudgmentResult(
            judgments=judgments,
            confidence=confidence,
            explanations=explanations,
            success=success,
            processing_time=processing_time,
            error=error
        )
    
    @staticmethod
    def create_large_test_dataset(size: int = 100) -> Dict[str, List]:
        """Create large test dataset for performance testing."""
        return {
            "texts": [f"测试文本 {i}" for i in range(size)],
            "entities": [[f"实体{i}-{j}" for j in range(3)] for i in range(size)],
            "triples": [
                [TestDataGenerator.create_test_triple(f"主语{i}", f"谓语{i}", f"宾语{i}") 
                 for _ in range(2)]
                for i in range(size)
            ]
        }


# =============================================================================
# COMMON TEST PATTERNS
# =============================================================================

class CommonTestPatterns:
    """Common test patterns following Testing_Demands.md guidelines."""
    
    @staticmethod
    def test_successful_operation(operation_func: Callable, 
                                 expected_result_type: type,
                                 *args, **kwargs):
        """Standard pattern for testing successful operations."""
        result = operation_func(*args, **kwargs)
        
        # Basic type checking
        assert isinstance(result, expected_result_type), (
            f"Expected {expected_result_type}, got {type(result)}"
        )
        
        # Success flag checking (if applicable)
        if hasattr(result, 'success'):
            assert result.success is True, f"Operation failed: {getattr(result, 'error', 'Unknown error')}"
        
        return result
    
    @staticmethod
    def test_error_handling(operation_func: Callable,
                           expected_error_type: type = Exception,
                           *args, **kwargs):
        """Standard pattern for testing error handling."""
        try:
            result = operation_func(*args, **kwargs)
            
            # If function returns result objects with error fields
            if hasattr(result, 'success') and hasattr(result, 'error'):
                assert result.success is False, "Expected operation to fail"
                assert result.error is not None, "Expected error message"
                return result
            else:
                pytest.fail("Expected exception or error result")
                
        except expected_error_type as e:
            # Exception was expected
            return e
        except Exception as e:
            pytest.fail(f"Unexpected exception type: {type(e)}, message: {str(e)}")
    
    @staticmethod  
    def test_edge_cases(operation_func: Callable, edge_cases: List[Dict]):
        """Standard pattern for testing edge cases."""
        results = []
        
        for case in edge_cases:
            description = case.get('description', 'Edge case')
            inputs = case.get('inputs', {})
            expected_behavior = case.get('expected', 'success')
            
            try:
                result = operation_func(**inputs)
                
                if expected_behavior == 'success':
                    assert hasattr(result, 'success') and result.success, (
                        f"{description} failed: {getattr(result, 'error', 'Unknown')}"
                    )
                elif expected_behavior == 'error':
                    assert hasattr(result, 'success') and not result.success, (
                        f"{description} should have failed"
                    )
                
                results.append(result)
                
            except Exception as e:
                if expected_behavior != 'exception':
                    pytest.fail(f"{description} raised unexpected exception: {e}")
                results.append(e)
        
        return results


# =============================================================================
# PYTEST FIXTURES AND DECORATORS
# =============================================================================

@pytest.fixture
def mock_gpt5_client():
    """Pytest fixture for mock GPT-5-mini client."""
    return create_mock_gpt5_client()


@pytest.fixture
def mock_perplexity_client():
    """Pytest fixture for mock Perplexity client."""
    return create_mock_perplexity_client()


@pytest.fixture
def performance_tracker():
    """Pytest fixture for tracking performance metrics."""
    return PerformanceTestUtils()


@pytest.fixture
def error_simulator():
    """Pytest fixture for error simulation."""
    return ErrorSimulator()


@pytest.fixture
def test_data_generator():
    """Pytest fixture for test data generation."""
    return TestDataGenerator()


def parametrize_scenarios(scenarios: List[str] = None):
    """Decorator for parametrizing tests with standard scenarios."""
    if scenarios is None:
        scenarios = [s[0] for s in STANDARD_TEST_SCENARIOS]
    
    return pytest.mark.parametrize("scenario_name", scenarios)


def mark_api_test(api_type: str = "both"):
    """Mark test as requiring API mocking."""
    return pytest.mark.api


def mark_integration_test():
    """Mark test as integration test."""
    return pytest.mark.integration


def mark_performance_test(max_time: float = 1.0):
    """Mark test as performance test with time limit."""
    def decorator(func):
        func._max_time = max_time
        return pytest.mark.performance(func)
    return decorator