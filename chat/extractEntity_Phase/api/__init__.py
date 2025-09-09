"""
API layer for extractEntity_Phase package.

This package provides the API integration layer including:
- Rate limiting and token tracking
- Disk caching for API responses
- GPT-5-mini client with LiteLLM integration
"""

__version__ = "1.0.0"
__author__ = "extractEntity_Phase Team"
__description__ = "API layer for entity extraction and text denoising"

# Import core API components
try:
    from .rate_limiter import (
        RateLimiter, RateLimitConfig, TokenUsage, RetryStrategy,
        create_rate_limiter, get_default_rate_limiter, create_retry_strategy
    )
except ImportError as e:
    print(f"Warning: Could not import rate_limiter: {e}")

try:
    from .cache_manager import (
        CacheManager, CacheConfig, CacheEntry,
        create_cache_manager, get_default_cache_manager, generate_cache_key
    )
except ImportError as e:
    print(f"Warning: Could not import cache_manager: {e}")

try:
    from .gpt5mini_client import (
        GPT5MiniClient, GPT5MiniConfig, APIRequest, APIResponse,
        create_gpt5mini_client, get_default_gpt5mini_client, simple_chat_completion
    )
except ImportError as e:
    print(f"Warning: Could not import gpt5mini_client: {e}")

# Convenience imports
__all__ = [
    # Rate limiting
    "RateLimiter", "RateLimitConfig", "TokenUsage", "RetryStrategy",
    "create_rate_limiter", "get_default_rate_limiter", "create_retry_strategy",
    
    # Caching
    "CacheManager", "CacheConfig", "CacheEntry",
    "create_cache_manager", "get_default_cache_manager", "generate_cache_key",
    
    # GPT-5-mini client
    "GPT5MiniClient", "GPT5MiniConfig", "APIRequest", "APIResponse",
    "create_gpt5mini_client", "get_default_gpt5mini_client", "simple_chat_completion"
]
