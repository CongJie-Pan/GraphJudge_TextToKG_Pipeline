"""
GPT-5-mini API client using LiteLLM.

This module provides a high-level client for interacting with GPT-5-mini
via LiteLLM, with integrated rate limiting, caching, and retry logic.
"""

import asyncio
import json
from typing import Dict, List, Optional, Any, Union, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import time

try:
    from litellm import completion, acompletion
    LITELLM_AVAILABLE = True
except ImportError:
    LITELLM_AVAILABLE = False
    # Mock for testing
    def completion(*args, **kwargs):
        raise ImportError("LiteLLM not available")
    def acompletion(*args, **kwargs):
        raise ImportError("LiteLLM not available")

from extractEntity_Phase.infrastructure.logging import get_logger
from extractEntity_Phase.infrastructure.config import get_config
from extractEntity_Phase.api.rate_limiter import RateLimiter, RateLimitConfig, RetryStrategy
from extractEntity_Phase.api.cache_manager import CacheManager, CacheConfig


@dataclass
class GPT5MiniConfig:
    """Configuration for GPT-5-mini client."""
    api_key: str = ""
    base_url: str = "https://api.openai.com"
    model: str = "gpt-5-mini"
    temperature: float = 1.0  # GPT-5 models only support default temperature
    max_tokens: int = 4000
    timeout: int = 60
    max_retries: int = 5
    
    # Rate limiting
    rpm_limit: int = 60
    tpm_limit: int = 90000
    tpd_limit: int = 2000000
    concurrent_limit: int = 3
    
    # Caching
    cache_enabled: bool = True
    cache_dir: str = "./cache"
    cache_ttl_hours: int = 24
    cache_max_size_mb: int = 100


@dataclass
class APIRequest:
    """API request structure."""
    prompt: str
    system_prompt: Optional[str] = None
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    model: Optional[str] = None
    # Accept arbitrary keyword arguments for APIRequest.
    # Note: dataclasses do not support **kwargs directly as fields.
    # Instead, we add an extra field to store additional parameters.
    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass
class APIResponse:
    """API response structure."""
    content: str
    model: str
    usage: Dict[str, int]
    finish_reason: str
    response_time: float
    cached: bool = False
    error: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "content": self.content,
            "model": self.model,
            "usage": self.usage,
            "finish_reason": self.finish_reason,
            "response_time": self.response_time,
            "cached": self.cached,
            "error": self.error
        }


class GPT5MiniClient:
    """
    High-level client for GPT-5-mini API interactions.
    
    Features:
    - Integrated rate limiting and token tracking
    - Intelligent caching with TTL
    - Automatic retry with exponential backoff
    - Concurrent request management
    - Comprehensive error handling
    - Chinese text optimization
    """
    
    def __init__(self, config: Optional[GPT5MiniConfig] = None):
        """
        Initialize GPT-5-mini client.
        
        Args:
            config: Client configuration
        """
        if not LITELLM_AVAILABLE:
            raise ImportError("LiteLLM is required but not available. Install with: pip install litellm")
        
        self.config = config or GPT5MiniConfig()
        self.logger = get_logger()
        
        # Load configuration from environment if not provided
        if not self.config.api_key:
            self._load_config_from_env()
        
        # Initialize components
        self.rate_limiter = RateLimiter(RateLimitConfig(
            rpm_limit=self.config.rpm_limit,
            tpm_limit=self.config.tpm_limit,
            tpd_limit=self.config.tpd_limit,
            concurrent_limit=self.config.concurrent_limit
        ))
        
        self.cache_manager = CacheManager(CacheConfig(
            enabled=self.config.cache_enabled,
            cache_dir=self.config.cache_dir,
            ttl_hours=self.config.cache_ttl_hours,
            max_size_mb=self.config.cache_max_size_mb
        ))
        
        self.retry_strategy = RetryStrategy(self.config.max_retries)
        
        # Statistics
        self.total_requests = 0
        self.successful_requests = 0
        self.failed_requests = 0
        self.cache_hits = 0
        self.cache_misses = 0
    
    def _load_config_from_env(self) -> None:
        """Load configuration from environment variables."""
        import os
        
        self.config.api_key = os.getenv("OPENAI_API_KEY", "")
        if not self.config.api_key:
            self.logger.warning("No API key provided. Set OPENAI_API_KEY environment variable.")
        
        # Load other config from environment
        self.config.base_url = os.getenv("OPENAI_BASE_URL", self.config.base_url)
        self.config.rpm_limit = int(os.getenv("OPENAI_RPM_LIMIT", str(self.config.rpm_limit)))
        self.config.tpm_limit = int(os.getenv("OPENAI_TPM_LIMIT", str(self.config.tpm_limit)))
        self.config.tpd_limit = int(os.getenv("OPENAI_TPD_LIMIT", str(self.config.tpd_limit)))
        self.config.concurrent_limit = int(os.getenv("OPENAI_CONCURRENT_LIMIT", str(self.config.concurrent_limit)))
    
    async def chat_completion(self, request: APIRequest) -> APIResponse:
        """
        Send a chat completion request to GPT-5-mini.
        
        Args:
            request: API request structure
            
        Returns:
            API response structure
        """
        start_time = time.time()
        self.total_requests += 1
        
        try:
            # Check cache first
            cache_key = self.cache_manager.generate_cache_key(
                request.prompt,
                request.system_prompt,
                temperature=request.temperature or self.config.temperature,
                max_tokens=request.max_tokens or self.config.max_tokens,
                model=request.model or self.config.model
            )
            
            cached_response = self.cache_manager.get(cache_key)
            if cached_response:
                self.cache_hits += 1
                self.logger.info(f"Cache hit for request: {request.prompt[:50]}...")
                
                return APIResponse(
                    content=cached_response,
                    model=request.model or self.config.model,
                    usage={"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
                    finish_reason="cached",
                    response_time=time.time() - start_time,
                    cached=True
                )
            
            self.cache_misses += 1
            
            # Wait for rate limits if necessary
            estimated_tokens = self._estimate_tokens(request.prompt, request.system_prompt)
            await self.rate_limiter.wait_if_needed(estimated_tokens)
            
            # Make API call with retry logic
            async with self.rate_limiter:
                response = await self.retry_strategy.execute_with_retry(
                    self._make_api_call, request
                )
            
            # Record successful request
            self.rate_limiter.record_request(response.usage.get("total_tokens", estimated_tokens))
            self.successful_requests += 1
            
            # Cache the response
            if self.config.cache_enabled:
                self.cache_manager.set(
                    cache_key,
                    request.prompt,
                    response.content,
                    request.system_prompt,
                    model=request.model or self.config.model,
                    temperature=request.temperature or self.config.temperature,
                    max_tokens=request.max_tokens or self.config.max_tokens,
                    token_count=response.usage.get("total_tokens", 0)
                )
            
            return response
            
        except Exception as e:
            self.failed_requests += 1
            self.logger.error(f"API request failed: {e}")
            
            return APIResponse(
                content="",
                model=request.model or self.config.model,
                usage={"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
                finish_reason="error",
                response_time=time.time() - start_time,
                error=str(e)
            )
    
    async def _make_api_call(self, request: APIRequest) -> APIResponse:
        """
        Make the actual API call to GPT-5-mini.
        
        Args:
            request: API request structure
            
        Returns:
            API response structure
        """
        start_time = time.time()
        
        # Build messages
        messages = []
        if request.system_prompt:
            messages.append({"role": "system", "content": request.system_prompt})
        messages.append({"role": "user", "content": request.prompt})
        
        # Prepare API parameters
        api_params = {
            "model": request.model or self.config.model,
            "messages": messages,
            "max_tokens": request.max_tokens or self.config.max_tokens,
            "timeout": self.config.timeout
        }
        
        # Note: GPT-5 models only support default temperature (1.0)
        # Custom temperature values are not supported
        
        # Make API call
        response = await acompletion(**api_params)
        
        # Extract response content
        content = response.choices[0].message.content
        finish_reason = response.choices[0].finish_reason
        
        # Extract usage information
        usage = {}
        if hasattr(response, 'usage') and response.usage:
            usage = {
                "prompt_tokens": getattr(response.usage, 'prompt_tokens', 0),
                "completion_tokens": getattr(response.usage, 'completion_tokens', 0),
                "total_tokens": getattr(response.usage, 'total_tokens', 0)
            }
        
        response_time = time.time() - start_time
        
        return APIResponse(
            content=content,
            model=request.model or self.config.model,
            usage=usage,
            finish_reason=finish_reason,
            response_time=response_time
        )
    
    def _estimate_tokens(self, prompt: str, system_prompt: Optional[str] = None) -> int:
        """
        Estimate token count for a request.
        
        Args:
            prompt: User prompt
            system_prompt: System prompt
            
        Returns:
            Estimated token count
        """
        # Rough estimation: 1 token ≈ 4 characters for English, 1 token ≈ 2 characters for Chinese
        total_chars = len(prompt)
        if system_prompt:
            total_chars += len(system_prompt)
        
        # Assume mixed content, use average of 3 characters per token
        estimated_tokens = total_chars // 3
        
        return min(estimated_tokens, self.config.max_tokens)
    
    async def batch_completion(self, requests: List[APIRequest], 
                             max_concurrent: Optional[int] = None) -> List[APIResponse]:
        """
        Process multiple requests in batch with concurrency control.
        
        Args:
            requests: List of API requests
            max_concurrent: Maximum concurrent requests (default: from config)
            
        Returns:
            List of API responses
        """
        if not requests:
            return []
        
        max_concurrent = max_concurrent or self.config.concurrent_limit
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def process_request(request: APIRequest) -> APIResponse:
            async with semaphore:
                return await self.chat_completion(request)
        
        # Process requests with progress tracking
        self.logger.info(f"Processing {len(requests)} requests with max {max_concurrent} concurrent")
        
        tasks = [process_request(req) for req in requests]
        responses = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Handle any exceptions
        processed_responses = []
        for i, response in enumerate(responses):
            if isinstance(response, Exception):
                self.logger.error(f"Request {i} failed: {response}")
                processed_responses.append(APIResponse(
                    content="",
                    model=requests[i].model or self.config.model,
                    usage={"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
                    finish_reason="error",
                    response_time=0.0,
                    error=str(response)
                ))
            else:
                processed_responses.append(response)
        
        self.logger.info(f"Batch processing completed: {len(processed_responses)} responses")
        return processed_responses
    
    async def complete(self, request: APIRequest) -> APIResponse:
        """
        Complete method for backward compatibility with core modules.
        
        Args:
            request: API request structure
            
        Returns:
            API response structure
        """
        return await self.chat_completion(request)
    
    async def entity_extraction(self, text: str, system_prompt: Optional[str] = None) -> APIResponse:
        """
        Extract entities from text using GPT-5-mini.
        
        Args:
            text: Text to extract entities from
            system_prompt: Custom system prompt for entity extraction
            
        Returns:
            API response with extracted entities
        """
        if not system_prompt:
            system_prompt = self._get_default_entity_extraction_prompt()
        
        request = APIRequest(
            prompt=text,
            system_prompt=system_prompt,
            max_tokens=self.config.max_tokens
        )
        
        return await self.chat_completion(request)
    
    async def text_denoising(self, text: str, entities: str, 
                           system_prompt: Optional[str] = None) -> APIResponse:
        """
        Denoise text using extracted entities.
        
        Args:
            text: Original text
            entities: Extracted entities
            system_prompt: Custom system prompt for denoising
            
        Returns:
            API response with denoised text
        """
        if not system_prompt:
            system_prompt = self._get_default_denoising_prompt()
        
        prompt = f"Original Text:\n{text}\n\nExtracted Entities:\n{entities}\n\nPlease denoise and restructure the original text based on the extracted entities."
        
        request = APIRequest(
            prompt=prompt,
            system_prompt=system_prompt,
            max_tokens=self.config.max_tokens
        )
        
        return await self.chat_completion(request)
    
    def _get_default_entity_extraction_prompt(self) -> str:
        """Get default system prompt for entity extraction."""
        return """You are an expert in classical Chinese literature and entity extraction. 
Your task is to identify and extract key entities from the given text.

Please extract the following types of entities:
- PERSON: Names of people, characters, historical figures
- LOCATION: Places, buildings, geographical locations
- ORGANIZATION: Institutions, groups, families
- OBJECT: Important objects, artifacts, items
- CONCEPT: Abstract concepts, themes, ideas
- EVENT: Historical events, story events
- TIME: Time periods, dates, temporal references

For each entity, provide:
1. The entity text
2. The entity type
3. A brief description or context
4. Confidence level (high/medium/low)

Format your response as a structured list with clear entity identification."""

    def _get_default_denoising_prompt(self) -> str:
        """Get default system prompt for text denoising."""
        return """You are an expert in classical Chinese literature and text processing.
Your task is to denoise and restructure the original text based on the extracted entities.

Please:
1. Use the extracted entities to identify the most important content
2. Remove noise, redundancy, and irrelevant information
3. Restructure the text for better clarity and readability
4. Maintain the classical Chinese style and tone
5. Preserve all important narrative elements
6. Ensure the output is coherent and well-organized

The result should be a clean, structured version that maintains the essence and beauty of the original text while being more accessible and organized."""

    def get_stats(self) -> Dict[str, Any]:
        """
        Get client statistics.
        
        Returns:
            Dictionary with client statistics
        """
        rate_limit_stats = self.rate_limiter.get_usage_stats()
        cache_stats = self.cache_manager.get_stats()
        
        return {
            "requests": {
                "total": self.total_requests,
                "successful": self.successful_requests,
                "failed": self.failed_requests,
                "success_rate": (self.successful_requests / self.total_requests * 100) if self.total_requests > 0 else 0
            },
            "cache": {
                "hits": self.cache_hits,
                "misses": self.cache_misses,
                "hit_rate": (self.cache_hits / (self.cache_hits + self.cache_misses) * 100) if (self.cache_hits + self.cache_misses) > 0 else 0
            },
            "rate_limiting": rate_limit_stats,
            "caching": cache_stats,
            "config": {
                "model": self.config.model,
                "max_tokens": self.config.max_tokens,
                "rpm_limit": self.config.rpm_limit,
                "tpm_limit": self.config.tpm_limit,
                "concurrent_limit": self.config.concurrent_limit
            }
        }
    
    def get_rate_limit_info(self) -> Dict[str, Any]:
        """
        Get current rate limit information.
        
        Returns:
            Dictionary with rate limit status
        """
        return self.rate_limiter.get_rate_limit_info()
    
    def clear_cache(self) -> None:
        """Clear all cached responses."""
        self.cache_manager.clear()
        self.logger.info("Cache cleared")


# Convenience functions for backward compatibility
def create_gpt5mini_client(config: Optional[GPT5MiniConfig] = None) -> GPT5MiniClient:
    """Create a GPT-5-mini client instance."""
    return GPT5MiniClient(config)


def get_default_gpt5mini_client() -> GPT5MiniClient:
    """Get a GPT-5-mini client with default configuration."""
    return GPT5MiniClient()


async def simple_chat_completion(prompt: str, system_prompt: Optional[str] = None, 
                               api_key: Optional[str] = None) -> str:
    """
    Simple chat completion function for backward compatibility.
    
    Args:
        prompt: User prompt
        system_prompt: System prompt
        api_key: API key (optional)
        
    Returns:
        Generated response
    """
    config = GPT5MiniConfig()
    if api_key:
        config.api_key = api_key
    
    client = GPT5MiniClient(config)
    request = APIRequest(prompt=prompt, system_prompt=system_prompt)
    response = await client.chat_completion(request)
    
    if response.error:
        raise Exception(f"API call failed: {response.error}")
    
    return response.content
