"""
Simplified API client wrapper for GraphJudge Streamlit Pipeline.

This module provides a clean, simplified interface to external AI APIs,
extracting essential functionality from the complex original scripts while
maintaining compatibility with core features.

Key simplifications:
- Synchronous execution (Streamlit-compatible)
- Basic error handling without complex retry mechanisms
- Simple rate limiting
- No file-based caching
"""

import time
from typing import Optional, Dict, Any
import litellm
from litellm import completion

# Configure litellm to drop unsupported parameters for GPT-5 models
litellm.drop_params = True

try:
    from ..core.config import get_api_config, get_model_config, GPT5_MINI_MODEL, PERPLEXITY_MODEL
except ImportError:
    # For direct execution or testing
    import sys
    import os
    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
    from core.config import get_api_config, get_model_config, GPT5_MINI_MODEL, PERPLEXITY_MODEL


class APIClient:
    """
    Simplified API client for OpenAI and Perplexity APIs.
    
    This client provides a clean interface for making API calls without
    the complexity of the original scripts' caching and rate limiting systems.
    """
    
    def __init__(self):
        """Initialize the API client with configuration."""
        self.config = get_model_config()
        self._last_request_time = 0
        self._min_request_interval = 0.1  # Minimum time between requests (seconds)
    
    def _rate_limit(self) -> None:
        """Simple rate limiting to avoid overwhelming APIs."""
        elapsed = time.time() - self._last_request_time
        if elapsed < self._min_request_interval:
            time.sleep(self._min_request_interval - elapsed)
        self._last_request_time = time.time()
    
    def call_gpt5_mini(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None
    ) -> str:
        """
        Call GPT-5-mini model for entity extraction and triple generation.
        
        Args:
            prompt: The user prompt/question
            system_prompt: Optional system message to guide behavior
            temperature: Temperature setting (defaults to config value)
            max_tokens: Maximum tokens (defaults to config value)
            
        Returns:
            Generated response text
            
        Raises:
            Exception: If API call fails after retries
        """
        return self._make_api_call(
            model=GPT5_MINI_MODEL,
            prompt=prompt,
            system_prompt=system_prompt,
            temperature=temperature or self.config["temperature"],
            max_tokens=max_tokens or self.config["max_tokens"],
            reasoning_effort="minimal"  # Use minimal reasoning to prioritize content generation
        )
    
    def call_perplexity(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None
    ) -> str:
        """
        Call Perplexity model for graph judgment.
        
        Args:
            prompt: The user prompt/question
            system_prompt: Optional system message to guide behavior
            temperature: Temperature setting (defaults to config value)
            max_tokens: Maximum tokens (defaults to config value)
            
        Returns:
            Generated response text
            
        Raises:
            Exception: If API call fails after retries
        """
        return self._make_api_call(
            model=PERPLEXITY_MODEL,
            prompt=prompt,
            system_prompt=system_prompt,
            temperature=temperature or self.config["temperature"],
            max_tokens=max_tokens or self.config["max_tokens"]
        )
    
    def _make_api_call(
        self,
        model: str,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 1.0,
        max_tokens: int = 4000,
        reasoning_effort: Optional[str] = None
    ) -> str:
        """
        Internal method to make API calls with basic retry logic.
        
        Args:
            model: Model identifier
            prompt: User prompt
            system_prompt: Optional system prompt
            temperature: Temperature setting
            max_tokens: Maximum tokens
            
        Returns:
            Generated response text
            
        Raises:
            Exception: If all retry attempts fail
        """
        # Apply rate limiting
        self._rate_limit()
        
        # Build messages
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        # Enhanced retry logic with progressive timeouts and reasoning effort progression
        last_exception = None
        current_reasoning_effort = reasoning_effort
        progressive_timeouts = self.config.get("progressive_timeouts", [self.config["timeout"]])
        reasoning_efforts = self.config.get("reasoning_efforts", ["minimal", "medium", None])

        for attempt in range(self.config["max_retries"]):
            try:
                # Use progressive timeouts: try shorter timeouts first, longer ones for retries
                current_timeout = progressive_timeouts[min(attempt, len(progressive_timeouts) - 1)]

                # Progressive reasoning effort for GPT-5 models
                if "gpt-5" in model.lower() and attempt < len(reasoning_efforts):
                    current_reasoning_effort = reasoning_efforts[attempt]

                # Build completion parameters with progressive timeout
                completion_params = {
                    "model": model,
                    "messages": messages,
                    "temperature": temperature,
                    "max_completion_tokens": max_tokens,
                    "timeout": current_timeout
                }

                # Add reasoning_effort for GPT-5 models to control reasoning behavior
                if "gpt-5" in model.lower() and current_reasoning_effort:
                    completion_params["reasoning_effort"] = current_reasoning_effort

                print(f"DEBUG: API call attempt {attempt + 1}/{self.config['max_retries']}")
                print(f"DEBUG: Timeout: {current_timeout}s, Reasoning effort: {current_reasoning_effort}")
                print(f"DEBUG: Completion params: {completion_params}")

                response = completion(**completion_params)

                # Enhanced content extraction for GPT-5 reasoning models
                content = self._extract_content_from_response(response, model, attempt)

                # Success criteria: non-empty content
                if content and content.strip():
                    print(f"DEBUG: API call successful on attempt {attempt + 1}")
                    return content

                # Handle empty content - continue to retry if attempts remain
                if attempt < self.config["max_retries"] - 1:
                    print(f"DEBUG: Empty content received, will retry with timeout {progressive_timeouts[min(attempt + 1, len(progressive_timeouts) - 1)]}s")
                    # Create exception for empty content to trigger retry
                    last_exception = Exception("Empty content received from API")
                    continue
                else:
                    print(f"DEBUG: Empty content on final attempt, returning empty string")
                    return ""

            except Exception as e:
                error_type = type(e).__name__
                error_msg = str(e)

                print(f"DEBUG: API call failed on attempt {attempt + 1}: {error_type} - {error_msg}")

                # Check if it's a timeout error specifically
                is_timeout = "timeout" in error_msg.lower() or "timed out" in error_msg.lower()

                last_exception = e

                if attempt < self.config["max_retries"] - 1:
                    # Enhanced exponential backoff with special handling for timeouts
                    if is_timeout:
                        wait_time = 2.0  # Fixed wait for timeouts to prevent overwhelming
                        print(f"DEBUG: Timeout detected, waiting {wait_time}s before retry with longer timeout")
                    else:
                        wait_time = (2 ** attempt) * 0.5  # 0.5, 1.0, 2.0 seconds for other errors
                        print(f"DEBUG: Non-timeout error, waiting {wait_time}s before retry")

                    time.sleep(wait_time)
                    continue

        # All retries failed - provide detailed error information
        timeout_info = f"Timeouts used: {progressive_timeouts[:self.config['max_retries']]}"
        reasoning_info = f"Reasoning efforts tried: {reasoning_efforts[:self.config['max_retries']]}" if "gpt-5" in model.lower() else ""

        detailed_error = f"API call failed after {self.config['max_retries']} attempts. {timeout_info}. {reasoning_info}. Last error: {last_exception}"
        raise Exception(detailed_error)

    def _extract_content_from_response(self, response, model: str, attempt: int) -> str:
        """
        Extract content from API response with enhanced handling for GPT-5 reasoning models.

        Args:
            response: API response object
            model: Model name for debugging
            attempt: Current retry attempt number

        Returns:
            Extracted content string or empty string
        """
        if not hasattr(response, 'choices') or len(response.choices) == 0:
            print(f"DEBUG: No choices in response. Response structure: {response}")
            return ""

        choice = response.choices[0]
        extracted_content = ""

        # Primary extraction: Standard message content
        if hasattr(choice, 'message') and hasattr(choice.message, 'content'):
            content = choice.message.content
            if content:
                print(f"DEBUG: Content received via message.content, length: {len(content)}")
                extracted_content = content
            else:
                print(f"DEBUG: Empty content in message.content")

        # Secondary extraction: Try to get content from reasoning responses
        if not extracted_content:
            content_from_reasoning = self._extract_from_reasoning_response(choice, model)
            if content_from_reasoning:
                print(f"DEBUG: Content extracted from reasoning response, length: {len(content_from_reasoning)}")
                extracted_content = content_from_reasoning

        # If still no content, perform final debugging
        if not extracted_content:
            print(f"DEBUG: Empty content received from API!")
            print(f"DEBUG: Model: {model} - GPT-5-mini reasoning mode issue suspected")
            print(f"DEBUG: Finish reason: {getattr(choice, 'finish_reason', 'unknown')}")
            print(f"DEBUG: Attempt: {attempt + 1}")

            # Check for reasoning tokens in response (GPT-5-mini specific debugging)
            if hasattr(response, 'usage') and hasattr(response.usage, 'completion_tokens_details'):
                details = response.usage.completion_tokens_details
                if hasattr(details, 'reasoning_tokens'):
                    print(f"DEBUG: Reasoning tokens used: {details.reasoning_tokens}")

        return extracted_content

    def _extract_from_reasoning_response(self, choice, model: str) -> str:
        """
        Attempt to extract content from reasoning-mode responses.

        Args:
            choice: API response choice object
            model: Model name

        Returns:
            Extracted content or empty string
        """
        # For GPT-5 models, try to extract from various possible response structures
        if "gpt-5" not in model.lower():
            return ""

        # Check if there's a reasoning attribute with content
        if hasattr(choice, 'reasoning'):
            reasoning = choice.reasoning
            if isinstance(reasoning, str) and reasoning.strip():
                # Sometimes reasoning contains the actual content
                return reasoning.strip()

        # Check for alternative content structures in GPT-5 responses
        if hasattr(choice, 'delta') and hasattr(choice.delta, 'content'):
            delta_content = choice.delta.content
            if delta_content:
                return delta_content

        return ""

    def test_api_connection(self) -> Dict[str, Any]:
        """
        Test API connectivity and return status information.

        Returns:
            Dictionary with test results for each model
        """
        results = {}

        # Test GPT-5-mini
        try:
            test_response = self.call_gpt5_mini(
                "Hello, this is a connection test.",
                "Respond with 'Connection successful' if you receive this message.",
                max_tokens=50
            )
            results["gpt5_mini"] = {
                "status": "success",
                "response_length": len(test_response),
                "model": GPT5_MINI_MODEL
            }
        except Exception as e:
            results["gpt5_mini"] = {
                "status": "failed",
                "error": str(e),
                "model": GPT5_MINI_MODEL
            }

        # Test Perplexity
        try:
            test_response = self.call_perplexity(
                "Hello, this is a connection test.",
                "Respond with 'Connection successful' if you receive this message.",
                max_tokens=50
            )
            results["perplexity"] = {
                "status": "success",
                "response_length": len(test_response),
                "model": PERPLEXITY_MODEL
            }
        except Exception as e:
            results["perplexity"] = {
                "status": "failed",
                "error": str(e),
                "model": PERPLEXITY_MODEL
            }

        return results


# Global client instance
_client_instance: Optional[APIClient] = None


def get_api_client() -> APIClient:
    """
    Get or create the global API client instance.
    
    Returns:
        Configured APIClient instance
    """
    global _client_instance
    if _client_instance is None:
        _client_instance = APIClient()
    return _client_instance


def call_gpt5_mini(prompt: str, system_prompt: Optional[str] = None, **kwargs) -> str:
    """
    Convenience function for GPT-5-mini calls.
    
    Args:
        prompt: User prompt
        system_prompt: Optional system prompt
        **kwargs: Additional parameters
        
    Returns:
        Generated response text
    """
    client = get_api_client()
    return client.call_gpt5_mini(prompt, system_prompt, **kwargs)


def call_perplexity(prompt: str, system_prompt: Optional[str] = None, **kwargs) -> str:
    """
    Convenience function for Perplexity calls.
    
    Args:
        prompt: User prompt
        system_prompt: Optional system prompt
        **kwargs: Additional parameters
        
    Returns:
        Generated response text
    """
    client = get_api_client()
    return client.call_perplexity(prompt, system_prompt, **kwargs)