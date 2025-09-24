"""
Test suite for timeout handling and error recovery enhancements.

This test module specifically validates the new progressive timeout and
error recovery mechanisms added to fix the API timeout issues.

Test Categories:
1. Progressive timeout functionality
2. Fallback mechanism testing
3. Enhanced error messages
4. Partial success scenarios
5. API client timeout progression
"""

import pytest
import time
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, Any

# Import the modules under test
try:
    from streamlit_pipeline.utils.api_client import APIClient
    from streamlit_pipeline.core.triple_generator import generate_triples
    from streamlit_pipeline.core.config import get_model_config
    from streamlit_pipeline.core.models import TripleResult, Triple
except ImportError:
    import sys
    import os
    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
    from utils.api_client import APIClient
    from core.triple_generator import generate_triples
    from core.config import get_model_config
    from core.models import TripleResult, Triple


class TestProgressiveTimeouts:
    """Test progressive timeout functionality in API client."""

    @patch('streamlit_pipeline.utils.api_client.completion')
    @patch('streamlit_pipeline.utils.api_client.get_model_config')
    def test_progressive_timeout_sequence(self, mock_config, mock_completion):
        """Test that API calls use progressive timeouts on retries."""
        # Mock configuration with progressive timeouts
        mock_config.return_value = {
            "timeout": 180,
            "progressive_timeouts": [120, 180, 240],
            "reasoning_efforts": ["minimal", "medium", None],
            "max_retries": 3,
            "temperature": 1.0,
            "max_tokens": 12000
        }

        # Mock timeout errors for first two attempts, success on third
        timeout_error = Exception("Request timed out")
        mock_completion.side_effect = [timeout_error, timeout_error, self._create_mock_response("Success")]

        client = APIClient()

        # Should succeed on third attempt with longest timeout
        result = client.call_gpt5_mini("Test prompt", "Test system prompt")

        assert result == "Success"
        assert mock_completion.call_count == 3

        # Verify timeout progression: 120s, 180s, 240s
        call_args_list = mock_completion.call_args_list
        assert call_args_list[0][1]["timeout"] == 120
        assert call_args_list[1][1]["timeout"] == 180
        assert call_args_list[2][1]["timeout"] == 240

    @patch('streamlit_pipeline.utils.api_client.completion')
    @patch('streamlit_pipeline.utils.api_client.get_model_config')
    def test_reasoning_effort_progression(self, mock_config, mock_completion):
        """Test that reasoning effort progresses through attempts."""
        mock_config.return_value = {
            "timeout": 180,
            "progressive_timeouts": [120, 180, 240],
            "reasoning_efforts": ["minimal", "medium", None],
            "max_retries": 3,
            "temperature": 1.0,
            "max_tokens": 12000
        }

        # Mock empty responses for first two attempts, success on third
        empty_response = self._create_mock_response("")
        success_response = self._create_mock_response("Valid content")
        mock_completion.side_effect = [empty_response, empty_response, success_response]

        client = APIClient()
        result = client.call_gpt5_mini("Test prompt", "Test system prompt")

        assert result == "Valid content"
        assert mock_completion.call_count == 3

        # Verify reasoning effort progression: minimal, medium, None (default)
        call_args_list = mock_completion.call_args_list
        assert call_args_list[0][1]["reasoning_effort"] == "minimal"
        assert call_args_list[1][1]["reasoning_effort"] == "medium"
        assert "reasoning_effort" not in call_args_list[2][1]  # None means parameter omitted

    @patch('streamlit_pipeline.utils.api_client.completion')
    @patch('streamlit_pipeline.utils.api_client.get_model_config')
    def test_timeout_error_detection(self, mock_config, mock_completion):
        """Test that timeout errors are properly detected and handled."""
        mock_config.return_value = {
            "timeout": 180,
            "progressive_timeouts": [120, 180, 240],
            "reasoning_efforts": ["minimal", "medium", None],
            "max_retries": 3,
            "temperature": 1.0,
            "max_tokens": 12000
        }

        # Mock timeout error messages for all 3 retries (expecting failure)
        timeout_errors = [
            Exception("Request timed out"),
            Exception("APITimeoutError - Request timed out"),
            Exception("litellm.Timeout: APITimeoutError - Request timed out")
        ]
        mock_completion.side_effect = timeout_errors

        client = APIClient()

        with patch('time.sleep') as mock_sleep:  # Mock sleep to speed up test
            # Should raise exception after all retries fail
            with pytest.raises(Exception) as exc_info:
                client.call_gpt5_mini("Test prompt", "Test system prompt")

            # Verify it's a timeout-related error with progressive timeouts info
            error_msg = str(exc_info.value)
            assert "API call failed after 3 attempts" in error_msg
            assert "Timeouts used: [120, 180, 240]" in error_msg
            assert "Reasoning efforts tried: ['minimal', 'medium', None]" in error_msg

            # Should have waited with fixed 2.0s intervals for timeout errors
            sleep_calls = [call[0][0] for call in mock_sleep.call_args_list]
            assert all(wait_time == 2.0 for wait_time in sleep_calls)

    def _create_mock_response(self, content: str):
        """Create a mock API response with the given content."""
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message = Mock()
        mock_response.choices[0].message.content = content
        return mock_response


class TestFallbackMechanisms:
    """Test fallback mechanisms in triple generator."""

    @patch('streamlit_pipeline.core.triple_generator.call_gpt5_mini')
    def test_chunk_fallback_on_timeout(self, mock_api_call):
        """Test that large chunks are split when primary API call times out."""
        # Mock timeout on primary call, success on fallback
        timeout_error = Exception("Request timed out")
        success_response = '{"triples": [["entity1", "relation", "entity2"]]}'

        mock_api_call.side_effect = [timeout_error, success_response]

        # Create a large chunk that should trigger fallback
        large_text = "這是一段很長的古典中文文本，包含很多複雜的內容。" * 50  # >500 chars
        entities = ["entity1", "entity2"]

        result = generate_triples(entities, large_text)

        assert result.success
        assert len(result.triples) > 0
        assert mock_api_call.call_count == 2  # Primary + fallback

    @patch('streamlit_pipeline.core.triple_generator.call_gpt5_mini')
    def test_skip_chunk_on_repeated_failures(self, mock_api_call):
        """Test that chunks are skipped when both primary and fallback fail."""
        # Mock failures for both primary and fallback calls
        timeout_error = Exception("Request timed out")
        mock_api_call.side_effect = [timeout_error, timeout_error]

        large_text = "這是一段很長的古典中文文本，包含很多複雜的內容。" * 50
        entities = ["entity1", "entity2"]

        result = generate_triples(entities, large_text)

        # Should not succeed but should not crash
        assert not result.success
        assert "timeout" in result.error.lower() or "api" in result.error.lower()
        assert mock_api_call.call_count == 2  # Primary + fallback

    @patch('streamlit_pipeline.core.triple_generator.call_gpt5_mini')
    def test_no_fallback_for_small_chunks(self, mock_api_call):
        """Test that small chunks don't trigger fallback on timeout."""
        timeout_error = Exception("Request timed out")
        mock_api_call.side_effect = [timeout_error]

        # Small text that won't trigger fallback
        small_text = "簡短文本"
        entities = ["entity1"]

        result = generate_triples(entities, small_text)

        assert not result.success
        assert mock_api_call.call_count == 1  # Only primary call, no fallback


class TestPartialSuccessHandling:
    """Test enhanced partial success handling."""

    @patch('streamlit_pipeline.core.triple_generator.call_gpt5_mini')
    def test_partial_success_acceptance(self, mock_api_call):
        """Test that partial success (some chunks succeed) is accepted."""
        # Mock alternating success/failure for chunks
        success_response = '{"triples": [["entity1", "relation", "entity2"]]}'
        timeout_error = Exception("Request timed out")

        mock_api_call.side_effect = [success_response, timeout_error, success_response, timeout_error]

        # Create text that will be split into multiple chunks
        text = "文本chunk1。" * 100 + "文本chunk2。" * 100 + "文本chunk3。" * 100 + "文本chunk4。" * 100
        entities = ["entity1", "entity2"]

        result = generate_triples(entities, text)

        # Should succeed with partial results (≥3 triples from successful chunks)
        assert result.success or len(result.triples) >= 3
        assert len(result.triples) > 0

    @patch('streamlit_pipeline.core.triple_generator.call_gpt5_mini')
    def test_enhanced_error_messages(self, mock_api_call):
        """Test that error messages provide better context for failures."""
        # Mock complete failure scenario
        timeout_error = Exception("Request timed out")
        mock_api_call.side_effect = [timeout_error] * 10

        text = "測試文本"
        entities = ["實體1", "實體2"]

        result = generate_triples(entities, text)

        assert not result.success
        assert result.error is not None

        # Check for enhanced error message content
        error_lower = result.error.lower()
        assert any(phrase in error_lower for phrase in [
            "timeout", "api", "connectivity", "no chunks could be processed"
        ])

    @patch('streamlit_pipeline.core.triple_generator.call_gpt5_mini')
    def test_processing_rate_calculation(self, mock_api_call):
        """Test that processing success rate is calculated correctly."""
        # Mock 2 successes out of 3 attempts (66% success rate)
        success_response = '{"triples": [["entity1", "relation", "entity2"]]}'
        timeout_error = Exception("Request timed out")

        mock_api_call.side_effect = [success_response, timeout_error, success_response]

        text = "文本1。" * 50 + "文本2。" * 50 + "文本3。" * 50  # Should create 3 chunks
        entities = ["entity1", "entity2"]

        result = generate_triples(entities, text)

        # Should succeed with 66% processing rate (>30% threshold)
        assert result.success

        # Check metadata includes processing statistics
        assert 'text_processing' in result.metadata
        text_meta = result.metadata['text_processing']
        assert text_meta['chunks_processed'] >= 2
        assert text_meta['chunks_with_triples'] >= 2


class TestConfigurationUpdates:
    """Test that configuration updates are properly applied."""

    def test_updated_timeout_values(self):
        """Test that new timeout configuration is loaded correctly."""
        config = get_model_config()

        # Verify updated timeout settings
        assert config["timeout"] == 180  # Updated from 60
        assert "progressive_timeouts" in config
        assert config["progressive_timeouts"] == [120, 180, 240]
        assert "reasoning_efforts" in config
        assert config["reasoning_efforts"] == ["minimal", "medium", None]

    def test_updated_model_parameters(self):
        """Test that model parameters are updated correctly."""
        config = get_model_config()

        # Verify updated model parameters
        assert config["temperature"] == 1.0  # GPT-5-mini requirement
        assert config["max_tokens"] == 12000  # Updated from 8000


@pytest.mark.integration
class TestEndToEndRecovery:
    """Integration tests for complete timeout recovery workflow."""

    @patch('streamlit_pipeline.utils.api_client.completion')
    def test_complete_timeout_recovery_workflow(self, mock_completion):
        """Test the complete workflow from timeout to recovery."""
        # Simulate timeout on first attempts, then success
        timeout_error = Exception("Request timed out")

        def side_effect(*args, **kwargs):
            # Fail on short timeouts, succeed on longer ones
            timeout_val = kwargs.get('timeout', 60)
            if timeout_val < 180:
                raise timeout_error
            else:
                return self._create_mock_response('{"triples": [["測試主體", "測試關係", "測試客體"]]}')

        mock_completion.side_effect = side_effect

        entities = ["測試主體", "測試客體"]
        text = "測試主體與測試客體之間有測試關係。"

        result = generate_triples(entities, text)

        assert result.success
        assert len(result.triples) > 0
        assert result.triples[0].subject == "測試主體"
        assert result.triples[0].predicate == "測試關係"
        assert result.triples[0].object == "測試客體"

    def _create_mock_response(self, content: str):
        """Create a mock API response with the given content."""
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message = Mock()
        mock_response.choices[0].message.content = content
        return mock_response