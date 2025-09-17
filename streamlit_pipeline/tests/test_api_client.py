"""
Comprehensive unit tests for APIClient module.

This test suite validates the unified API client wrapper that provides
simplified access to OpenAI and Perplexity APIs with rate limiting,
error handling, and retry logic.

Test Categories:
1. Basic functionality tests (successful API calls)
2. Error handling tests (API failures, timeouts)
3. Rate limiting tests (request throttling)
4. Configuration tests (model parameters, API keys)
5. Integration tests (litellm integration)

Following TDD principles per docs/Testing_Demands.md.
"""

import pytest
import time
from unittest.mock import Mock, patch, MagicMock
from typing import Optional

# Import the module under test
try:
    from streamlit_pipeline.utils.api_client import (
        APIClient,
        get_api_client,
        call_gpt5_mini,
        call_perplexity
    )
    from streamlit_pipeline.core.config import GPT5_MINI_MODEL, PERPLEXITY_MODEL
except ImportError:
    import sys
    import os
    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
    from utils.api_client import (
        APIClient,
        get_api_client,
        call_gpt5_mini,
        call_perplexity
    )
    from core.config import GPT5_MINI_MODEL, PERPLEXITY_MODEL


class TestAPIClientInitialization:
    """Test API client initialization and configuration."""
    
    def test_api_client_initialization(self):
        """Test that APIClient initializes with proper configuration."""
        # Mock the imported function in the actual module location
        with patch('streamlit_pipeline.utils.api_client.get_model_config') as mock_config:
            mock_config.return_value = {
                "entity_model": "gpt-5-mini",
                "temperature": 0.0,
                "max_tokens": 4000,
                "timeout": 60,
                "max_retries": 3
            }
            
            client = APIClient()
            
            assert client.config is not None
            assert client._last_request_time == 0
            assert client._min_request_interval == 0.1
            mock_config.assert_called_once()
    
    def test_get_api_client_singleton(self):
        """Test that get_api_client returns singleton instance."""
        client1 = get_api_client()
        client2 = get_api_client()
        
        assert client1 is client2
        assert isinstance(client1, APIClient)


class TestRateLimiting:
    """Test rate limiting functionality."""
    
    def test_rate_limiting_delay(self):
        """Test that rate limiting adds proper delays between requests."""
        with patch('streamlit_pipeline.utils.api_client.get_model_config') as mock_config:
            mock_config.return_value = {
                "temperature": 0.0,
                "max_tokens": 4000,
                "timeout": 60,
                "max_retries": 3
            }
            
            client = APIClient()
            
            # First call should not be delayed
            start_time = time.time()
            client._rate_limit()
            first_duration = time.time() - start_time
            
            # Second call should be delayed
            start_time = time.time()
            client._rate_limit()
            second_duration = time.time() - start_time
            
            # Second call should take longer due to rate limiting
            assert second_duration >= client._min_request_interval
    
    def test_rate_limiting_updates_last_request_time(self):
        """Test that rate limiting updates the last request time."""
        with patch('streamlit_pipeline.utils.api_client.get_model_config') as mock_config:
            mock_config.return_value = {"max_retries": 3}
            
            client = APIClient()
            initial_time = client._last_request_time
            
            client._rate_limit()
            
            assert client._last_request_time > initial_time


class TestGPT5MiniAPICalls:
    """Test GPT-5-mini API calls."""
    
    @patch('streamlit_pipeline.utils.api_client.completion')
    def test_call_gpt5_mini_success(self, mock_completion):
        """Test successful GPT-5-mini API call."""
        # Setup mock response
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = "Test response"
        mock_completion.return_value = mock_response
        
        with patch('streamlit_pipeline.utils.api_client.get_model_config') as mock_config:
            mock_config.return_value = {
                "temperature": 0.0,
                "max_tokens": 4000,
                "timeout": 60,
                "max_retries": 3
            }
            
            client = APIClient()
            result = client.call_gpt5_mini("Test prompt")
            
            assert result == "Test response"
            mock_completion.assert_called_once()
            
            # Verify correct model was used
            call_args = mock_completion.call_args
            assert call_args[1]['model'] == GPT5_MINI_MODEL
    
    @patch('streamlit_pipeline.utils.api_client.completion')
    def test_call_gpt5_mini_with_system_prompt(self, mock_completion):
        """Test GPT-5-mini call with system prompt."""
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = "Test response"
        mock_completion.return_value = mock_response
        
        with patch('streamlit_pipeline.utils.api_client.get_model_config') as mock_config:
            mock_config.return_value = {
                "temperature": 0.0,
                "max_tokens": 4000,
                "timeout": 60,
                "max_retries": 3
            }
            
            client = APIClient()
            result = client.call_gpt5_mini(
                "Test prompt",
                system_prompt="You are a helpful assistant"
            )
            
            assert result == "Test response"
            
            # Check messages structure
            call_args = mock_completion.call_args
            messages = call_args[1]['messages']
            assert len(messages) == 2
            assert messages[0]['role'] == 'system'
            assert messages[0]['content'] == "You are a helpful assistant"
            assert messages[1]['role'] == 'user'
            assert messages[1]['content'] == "Test prompt"
    
    @patch('streamlit_pipeline.utils.api_client.completion')
    def test_call_gpt5_mini_with_custom_parameters(self, mock_completion):
        """Test GPT-5-mini call with custom temperature and max_tokens."""
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = "Test response"
        mock_completion.return_value = mock_response
        
        with patch('streamlit_pipeline.utils.api_client.get_model_config') as mock_config:
            mock_config.return_value = {
                "temperature": 0.0,
                "max_tokens": 4000,
                "timeout": 60,
                "max_retries": 3
            }
            
            client = APIClient()
            result = client.call_gpt5_mini(
                "Test prompt",
                temperature=0.5,
                max_tokens=2000
            )
            
            assert result == "Test response"
            
            # Check custom parameters were used
            call_args = mock_completion.call_args
            assert call_args[1]['temperature'] == 0.5
            assert call_args[1]['max_completion_tokens'] == 2000


class TestPerplexityAPICalls:
    """Test Perplexity API calls."""
    
    @patch('streamlit_pipeline.utils.api_client.completion')
    def test_call_perplexity_success(self, mock_completion):
        """Test successful Perplexity API call."""
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = "Perplexity response"
        mock_completion.return_value = mock_response
        
        with patch('streamlit_pipeline.utils.api_client.get_model_config') as mock_config:
            mock_config.return_value = {
                "temperature": 0.0,
                "max_tokens": 4000,
                "timeout": 60,
                "max_retries": 3
            }
            
            client = APIClient()
            result = client.call_perplexity("Test judgment prompt")
            
            assert result == "Perplexity response"
            mock_completion.assert_called_once()
            
            # Verify correct model was used
            call_args = mock_completion.call_args
            assert call_args[1]['model'] == PERPLEXITY_MODEL
    
    @patch('streamlit_pipeline.utils.api_client.completion')
    def test_call_perplexity_with_system_prompt(self, mock_completion):
        """Test Perplexity call with system prompt."""
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = "Judgment response"
        mock_completion.return_value = mock_response
        
        with patch('streamlit_pipeline.utils.api_client.get_model_config') as mock_config:
            mock_config.return_value = {
                "temperature": 0.0,
                "max_tokens": 4000,
                "timeout": 60,
                "max_retries": 3
            }
            
            client = APIClient()
            result = client.call_perplexity(
                "Judge these triples",
                system_prompt="You are a knowledge graph judge"
            )
            
            assert result == "Judgment response"
            
            # Check messages structure
            call_args = mock_completion.call_args
            messages = call_args[1]['messages']
            assert len(messages) == 2
            assert messages[0]['role'] == 'system'
            assert messages[1]['role'] == 'user'


class TestErrorHandling:
    """Test error handling and retry logic."""
    
    @patch('streamlit_pipeline.utils.api_client.completion')
    @patch('streamlit_pipeline.utils.api_client.time.sleep')  # Mock sleep to speed up tests
    def test_retry_on_api_failure(self, mock_sleep, mock_completion):
        """Test that API calls are retried on failure."""
        # Configure mock to fail twice, then succeed
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = "Success after retries"
        
        mock_completion.side_effect = [
            Exception("First failure"),
            Exception("Second failure"), 
            mock_response  # Third attempt succeeds
        ]
        
        with patch('streamlit_pipeline.utils.api_client.get_model_config') as mock_config:
            mock_config.return_value = {
                "temperature": 0.0,
                "max_tokens": 4000,
                "timeout": 60,
                "max_retries": 3
            }
            
            client = APIClient()
            result = client.call_gpt5_mini("Test prompt")
            
            assert result == "Success after retries"
            assert mock_completion.call_count == 3
            assert mock_sleep.call_count == 2  # Two sleeps between 3 attempts
    
    @patch('streamlit_pipeline.utils.api_client.completion')
    @patch('streamlit_pipeline.utils.api_client.time.sleep')
    def test_exhaust_all_retries(self, mock_sleep, mock_completion):
        """Test that exception is raised when all retries are exhausted."""
        mock_completion.side_effect = Exception("Persistent failure")
        
        with patch('streamlit_pipeline.utils.api_client.get_model_config') as mock_config:
            mock_config.return_value = {
                "temperature": 0.0,
                "max_tokens": 4000,
                "timeout": 60,
                "max_retries": 3
            }
            
            client = APIClient()
            
            with pytest.raises(Exception) as exc_info:
                client.call_gpt5_mini("Test prompt")
            
            assert "API call failed after 3 attempts" in str(exc_info.value)
            assert mock_completion.call_count == 3
            assert mock_sleep.call_count == 2
    
    @patch('streamlit_pipeline.utils.api_client.completion')
    @patch('streamlit_pipeline.utils.api_client.time.sleep')
    def test_exponential_backoff(self, mock_sleep, mock_completion):
        """Test that retry delays follow exponential backoff pattern."""
        mock_completion.side_effect = [
            Exception("Failure 1"),
            Exception("Failure 2"),
            Exception("Failure 3")
        ]
        
        with patch('streamlit_pipeline.utils.api_client.get_model_config') as mock_config:
            mock_config.return_value = {
                "temperature": 0.0,
                "max_tokens": 4000,
                "timeout": 60,
                "max_retries": 3
            }
            
            client = APIClient()
            
            with pytest.raises(Exception):
                client.call_gpt5_mini("Test prompt")
            
            # Check that sleep was called with increasing delays
            sleep_calls = [call[0][0] for call in mock_sleep.call_args_list]
            assert sleep_calls == [0.5, 1.0]  # Exponential backoff: 0.5, 1.0
    
    def test_api_configuration_error_handling(self):
        """Test handling of configuration errors."""
        with patch('streamlit_pipeline.utils.api_client.get_model_config') as mock_config:
            mock_config.side_effect = Exception("Configuration error")
            
            with pytest.raises(Exception):
                APIClient()


class TestConvenienceFunctions:
    """Test module-level convenience functions."""
    
    @patch('streamlit_pipeline.utils.api_client.get_api_client')
    def test_call_gpt5_mini_convenience(self, mock_get_client):
        """Test call_gpt5_mini convenience function."""
        mock_client = Mock()
        mock_client.call_gpt5_mini.return_value = "GPT response"
        mock_get_client.return_value = mock_client
        
        result = call_gpt5_mini("Test prompt", system_prompt="Test system")
        
        assert result == "GPT response"
        mock_client.call_gpt5_mini.assert_called_once_with(
            "Test prompt", "Test system"
        )
    
    @patch('streamlit_pipeline.utils.api_client.get_api_client')
    def test_call_perplexity_convenience(self, mock_get_client):
        """Test call_perplexity convenience function."""
        mock_client = Mock()
        mock_client.call_perplexity.return_value = "Perplexity response"
        mock_get_client.return_value = mock_client
        
        result = call_perplexity(
            "Test prompt", 
            system_prompt="Test system",
            temperature=0.5
        )
        
        assert result == "Perplexity response"
        mock_client.call_perplexity.assert_called_once_with(
            "Test prompt", "Test system", temperature=0.5
        )


class TestAPIClientIntegration:
    """Integration tests with actual configuration."""
    
    @patch('streamlit_pipeline.utils.api_client.completion')
    def test_full_api_call_flow(self, mock_completion):
        """Test complete API call flow with real configuration loading."""
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = "Integration test response"
        mock_completion.return_value = mock_response
        
        # This test uses actual config loading (no mocking)
        client = APIClient()
        result = client.call_gpt5_mini("Integration test prompt")
        
        assert result == "Integration test response"
        
        # Verify proper parameters were passed
        call_args = mock_completion.call_args
        assert call_args[1]['model'] == GPT5_MINI_MODEL
        assert 'messages' in call_args[1]
        assert 'temperature' in call_args[1]
        assert 'max_completion_tokens' in call_args[1]
        assert 'timeout' in call_args[1]


class TestEdgeCases:
    """Test edge cases and boundary conditions."""
    
    @patch('streamlit_pipeline.utils.api_client.completion')
    def test_empty_prompt_handling(self, mock_completion):
        """Test handling of empty prompts."""
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = "Empty prompt response"
        mock_completion.return_value = mock_response
        
        with patch('streamlit_pipeline.utils.api_client.get_model_config') as mock_config:
            mock_config.return_value = {
                "temperature": 0.0,
                "max_tokens": 4000,
                "timeout": 60,
                "max_retries": 3
            }
            
            client = APIClient()
            result = client.call_gpt5_mini("")
            
            assert result == "Empty prompt response"
            
            # Verify empty prompt was passed correctly
            call_args = mock_completion.call_args
            messages = call_args[1]['messages']
            assert messages[-1]['content'] == ""
    
    @patch('streamlit_pipeline.utils.api_client.completion')
    def test_very_long_prompt_handling(self, mock_completion):
        """Test handling of very long prompts."""
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = "Long prompt response"
        mock_completion.return_value = mock_response
        
        with patch('streamlit_pipeline.utils.api_client.get_model_config') as mock_config:
            mock_config.return_value = {
                "temperature": 0.0,
                "max_tokens": 4000,
                "timeout": 60,
                "max_retries": 3
            }
            
            client = APIClient()
            long_prompt = "Test " * 10000  # Very long prompt
            result = client.call_gpt5_mini(long_prompt)
            
            assert result == "Long prompt response"
    
    def test_zero_max_retries(self):
        """Test behavior when max_retries is set to 0."""
        with patch('streamlit_pipeline.utils.api_client.get_model_config') as mock_config:
            mock_config.return_value = {
                "temperature": 0.0,
                "max_tokens": 4000,
                "timeout": 60,
                "max_retries": 0
            }
            
            with patch('streamlit_pipeline.utils.api_client.completion') as mock_completion:
                mock_completion.side_effect = Exception("API failure")
                
                client = APIClient()
                
                with pytest.raises(Exception) as exc_info:
                    client.call_gpt5_mini("Test prompt")
                
                assert "API call failed after 0 attempts" in str(exc_info.value)
                # With 0 max_retries, range(0) makes no calls, so no API calls should happen
                assert mock_completion.call_count == 0


class TestGPT5MiniCompatibility:
    """Test GPT-5-mini specific compatibility and reasoning mode handling."""

    @patch('streamlit_pipeline.utils.api_client.completion')
    def test_gpt5_mini_reasoning_effort_parameter(self, mock_completion):
        """Test that reasoning_effort parameter is passed for GPT-5-mini calls."""
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = "Response with minimal reasoning"
        mock_completion.return_value = mock_response

        with patch('streamlit_pipeline.utils.api_client.get_model_config') as mock_config:
            mock_config.return_value = {
                "temperature": 1.0,
                "max_tokens": 8000,
                "timeout": 60,
                "max_retries": 3
            }

            client = APIClient()
            result = client.call_gpt5_mini("Test prompt")

            assert result == "Response with minimal reasoning"

            # Verify reasoning_effort parameter was passed
            call_args = mock_completion.call_args
            assert call_args[1]['reasoning_effort'] == 'minimal'

    @patch('streamlit_pipeline.utils.api_client.completion')
    def test_gpt5_mini_empty_response_recovery(self, mock_completion):
        """Test recovery mechanism when GPT-5-mini returns empty content."""
        # Create properly mocked empty responses without default Mock attributes
        def create_empty_response():
            choice = Mock()
            choice.message.content = ""
            choice.finish_reason = "length"
            # Remove Mock default attributes that might interfere
            del choice.reasoning
            del choice.delta

            response = Mock()
            response.choices = [choice]
            return response

        empty_response1 = create_empty_response()
        empty_response2 = create_empty_response()

        success_response = Mock()
        success_response.choices = [Mock()]
        success_response.choices[0].message.content = "Success after retry"

        mock_completion.side_effect = [
            empty_response1,  # First attempt with minimal reasoning
            empty_response2,  # Second attempt with medium reasoning
            success_response  # Third attempt succeeds
        ]

        with patch('streamlit_pipeline.utils.api_client.get_model_config') as mock_config:
            mock_config.return_value = {
                "temperature": 1.0,
                "max_tokens": 8000,
                "timeout": 60,
                "max_retries": 3
            }

            client = APIClient()
            result = client.call_gpt5_mini("Test prompt")

            assert result == "Success after retry"
            assert mock_completion.call_count == 3

            # Verify reasoning_effort progression: minimal -> medium -> None (not set)
            call_args_list = mock_completion.call_args_list
            assert call_args_list[0][1]['reasoning_effort'] == 'minimal'
            assert call_args_list[1][1]['reasoning_effort'] == 'medium'
            # Third call should not have reasoning_effort parameter since it was set to None
            assert 'reasoning_effort' not in call_args_list[2][1]

    @patch('streamlit_pipeline.utils.api_client.completion')
    def test_gpt5_mini_reasoning_response_extraction(self, mock_completion):
        """Test extraction of content from reasoning-mode responses."""
        # Mock response with reasoning but empty message content
        reasoning_response = Mock()
        reasoning_response.choices = [Mock()]
        reasoning_response.choices[0].message.content = ""
        reasoning_response.choices[0].reasoning = "This is content from reasoning field"
        reasoning_response.choices[0].finish_reason = "stop"

        mock_completion.return_value = reasoning_response

        with patch('streamlit_pipeline.utils.api_client.get_model_config') as mock_config:
            mock_config.return_value = {
                "temperature": 1.0,
                "max_tokens": 8000,
                "timeout": 60,
                "max_retries": 3
            }

            client = APIClient()
            result = client.call_gpt5_mini("Test prompt")

            # Should extract content from reasoning field
            assert result == "This is content from reasoning field"

    @patch('streamlit_pipeline.utils.api_client.completion')
    def test_gpt5_mini_token_usage_logging(self, mock_completion):
        """Test proper logging of reasoning tokens usage."""
        # Create properly mocked empty response
        choice = Mock()
        choice.message.content = ""
        choice.finish_reason = "length"
        # Remove Mock default attributes
        del choice.reasoning
        del choice.delta

        mock_response = Mock()
        mock_response.choices = [choice]

        # Mock usage details with reasoning tokens
        mock_usage = Mock()
        mock_usage.completion_tokens_details = Mock()
        mock_usage.completion_tokens_details.reasoning_tokens = 4000
        mock_response.usage = mock_usage

        mock_completion.return_value = mock_response

        with patch('streamlit_pipeline.utils.api_client.get_model_config') as mock_config:
            mock_config.return_value = {
                "temperature": 1.0,
                "max_tokens": 8000,
                "timeout": 60,
                "max_retries": 1  # Only one attempt to trigger empty response handling
            }

            client = APIClient()

            # Capture print output for debugging verification
            with patch('builtins.print') as mock_print:
                result = client.call_gpt5_mini("Test prompt")

                # Should log reasoning tokens usage
                print_calls = [str(call) for call in mock_print.call_args_list]
                reasoning_token_logged = any("Reasoning tokens used: 4000" in call for call in print_calls)
                assert reasoning_token_logged, f"Expected reasoning tokens log not found in: {print_calls}"

    @patch('streamlit_pipeline.utils.api_client.completion')
    def test_non_gpt5_model_no_reasoning_effort(self, mock_completion):
        """Test that reasoning_effort is not passed to non-GPT-5 models."""
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = "Standard response"
        mock_completion.return_value = mock_response

        with patch('streamlit_pipeline.utils.api_client.get_model_config') as mock_config:
            mock_config.return_value = {
                "temperature": 1.0,
                "max_tokens": 8000,
                "timeout": 60,
                "max_retries": 3
            }

            client = APIClient()

            # Call perplexity (non-GPT-5 model)
            result = client.call_perplexity("Test prompt")

            assert result == "Standard response"

            # Verify reasoning_effort parameter was NOT passed
            call_args = mock_completion.call_args
            assert 'reasoning_effort' not in call_args[1]

    @patch('streamlit_pipeline.utils.api_client.completion')
    def test_gpt5_mini_max_tokens_increased(self, mock_completion):
        """Test that GPT-5-mini uses increased token limit."""
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = "Response with high token limit"
        mock_completion.return_value = mock_response

        with patch('streamlit_pipeline.utils.api_client.get_model_config') as mock_config:
            mock_config.return_value = {
                "temperature": 1.0,
                "max_tokens": 8000,  # Updated from 4000
                "timeout": 60,
                "max_retries": 3
            }

            client = APIClient()
            result = client.call_gpt5_mini("Test prompt")

            assert result == "Response with high token limit"

            # Verify increased token limit was used
            call_args = mock_completion.call_args
            assert call_args[1]['max_completion_tokens'] == 8000

    @patch('streamlit_pipeline.utils.api_client.completion')
    def test_gpt5_mini_finish_reason_length_handling(self, mock_completion):
        """Test proper handling when finish_reason is 'length' (token limit hit)."""
        # Create properly mocked empty response
        length_choice = Mock()
        length_choice.message.content = ""
        length_choice.finish_reason = "length"
        # Remove Mock default attributes
        del length_choice.reasoning
        del length_choice.delta

        length_response = Mock()
        length_response.choices = [length_choice]

        success_response = Mock()
        success_response.choices = [Mock()]
        success_response.choices[0].message.content = "Success with different reasoning effort"
        success_response.choices[0].finish_reason = "stop"

        mock_completion.side_effect = [length_response, success_response]

        with patch('streamlit_pipeline.utils.api_client.get_model_config') as mock_config:
            mock_config.return_value = {
                "temperature": 1.0,
                "max_tokens": 8000,
                "timeout": 60,
                "max_retries": 3
            }

            client = APIClient()
            result = client.call_gpt5_mini("Test prompt")

            assert result == "Success with different reasoning effort"
            assert mock_completion.call_count == 2

            # First call should have minimal reasoning effort
            assert mock_completion.call_args_list[0][1]['reasoning_effort'] == 'minimal'
            # Second call should have medium reasoning effort
            assert mock_completion.call_args_list[1][1]['reasoning_effort'] == 'medium'


class TestReasoningModeScenarios:
    """Test specific reasoning mode scenarios that cause empty content."""

    @patch('streamlit_pipeline.utils.api_client.completion')
    def test_all_reasoning_no_content_scenario(self, mock_completion):
        """Test scenario where model uses all tokens for reasoning, produces no content."""
        mock_response = Mock()
        mock_choice = Mock()
        mock_message = Mock()
        mock_message.content = ""  # Ensure this is a proper string
        mock_choice.message = mock_message
        mock_choice.finish_reason = "length"
        # Ensure reasoning attributes return empty/None to avoid len() issues
        mock_choice.reasoning = None
        mock_choice.delta = None
        mock_response.choices = [mock_choice]

        # Mock usage showing all tokens went to reasoning
        mock_usage = Mock()
        mock_usage.completion_tokens_details = Mock()
        mock_usage.completion_tokens_details.reasoning_tokens = 7500  # Most tokens used for reasoning
        mock_response.usage = mock_usage

        mock_completion.return_value = mock_response

        with patch('streamlit_pipeline.utils.api_client.get_model_config') as mock_config:
            mock_config.return_value = {
                "temperature": 1.0,
                "max_tokens": 8000,
                "timeout": 60,
                "max_retries": 1
            }

            client = APIClient()
            result = client.call_gpt5_mini("Complex reasoning prompt")

            # Should return empty string gracefully
            assert result == ""

    @patch('streamlit_pipeline.utils.api_client.completion')
    def test_mixed_reasoning_and_content_scenario(self, mock_completion):
        """Test scenario with both reasoning tokens and content."""
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = "Final answer after reasoning"
        mock_response.choices[0].finish_reason = "stop"

        # Mock balanced token usage
        mock_usage = Mock()
        mock_usage.completion_tokens_details = Mock()
        mock_usage.completion_tokens_details.reasoning_tokens = 2000
        mock_response.usage = mock_usage

        mock_completion.return_value = mock_response

        with patch('streamlit_pipeline.utils.api_client.get_model_config') as mock_config:
            mock_config.return_value = {
                "temperature": 1.0,
                "max_tokens": 8000,
                "timeout": 60,
                "max_retries": 3
            }

            client = APIClient()
            result = client.call_gpt5_mini("Reasoning prompt")

            assert result == "Final answer after reasoning"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])