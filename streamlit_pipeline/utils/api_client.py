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
            max_tokens=max_tokens or self.config["max_tokens"]
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
        max_tokens: int = 4000
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
        
        # Retry logic
        last_exception = None
        for attempt in range(self.config["max_retries"]):
            try:
                response = completion(
                    model=model,
                    messages=messages,
                    temperature=temperature,
                    max_completion_tokens=max_tokens,
                    timeout=self.config["timeout"]
                )

                # Debug: Check response structure
                print(f"DEBUG: API call successful. Model: {model}, Temperature: {temperature}")
                if hasattr(response, 'choices') and len(response.choices) > 0:
                    choice = response.choices[0]
                    if hasattr(choice, 'message') and hasattr(choice.message, 'content'):
                        content = choice.message.content
                        # Debug log for empty content
                        if not content:
                            print(f"DEBUG: Empty content received from API!")
                            print(f"DEBUG: Model: {model} - GPT-5-mini reasoning mode issue suspected")
                            print(f"DEBUG: Response object: {response}")
                            print(f"DEBUG: Choice object: {choice}")
                            print(f"DEBUG: Message object: {choice.message}")
                            print(f"DEBUG: Finish reason: {getattr(choice, 'finish_reason', 'unknown')}")
                            # Check for reasoning tokens in response (GPT-5-mini specific issue)
                            if hasattr(response, 'choices') and hasattr(response.choices[0], 'reasoning'):
                                print(f"DEBUG: Reasoning tokens present: {bool(getattr(response.choices[0], 'reasoning', None))}")
                            if hasattr(choice, 'reasoning'):
                                print(f"DEBUG: Choice has reasoning: {bool(getattr(choice, 'reasoning', None))}")
                        else:
                            print(f"DEBUG: Content received, length: {len(content)}")
                        return content or ""  # Return empty string instead of None
                    else:
                        print(f"DEBUG: No message.content in choice. Choice structure: {choice}")
                        return ""
                else:
                    print(f"DEBUG: No choices in response. Response structure: {response}")
                    return ""
                
            except Exception as e:
                last_exception = e
                if attempt < self.config["max_retries"] - 1:
                    # Wait before retry with exponential backoff
                    wait_time = (2 ** attempt) * 0.5  # 0.5, 1.0, 2.0 seconds
                    time.sleep(wait_time)
                    continue
        
        # All retries failed
        raise Exception(f"API call failed after {self.config['max_retries']} attempts: {last_exception}")

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