"""
GPT-5-mini API Configuration Module

This module contains configuration settings for the GPT-5-mini API integration via OpenAI,
specifically focused on rate limiting and concurrency control to ensure
compliance with API usage limits.

Key Configuration Options:
- OPENAI_RPM_LIMIT: Requests per minute limit for your API plan
- OPENAI_CONCURRENT_LIMIT: Maximum concurrent requests allowed
- OPENAI_RETRY_ATTEMPTS: Maximum retry attempts for failed requests
- OPENAI_BASE_DELAY: Base delay between requests in seconds
- OPENAI_TPM_LIMIT: Tokens per minute limit
- OPENAI_TPD_LIMIT: Tokens per day limit

OpenAI Free Tier Limits:
- Concurrency: 3
- TPM: 90,000
- RPM: 3500
- TPD: 2,000,000

Usage:
    from openai_config import OPENAI_RPM_LIMIT, OPENAI_CONCURRENT_LIMIT
    # Use these values in your API calls
"""

# Rate limiting configuration for OpenAI API
# Using original settings from KIMI config for consistency

import time
import random

# OpenAI rate limits - Using more reasonable limits for GPT-5-mini
OPENAI_RPM_LIMIT = 60  # Requests per minute limit (reasonable for development)
OPENAI_TPM_LIMIT = 90000  # Tokens per minute limit (OpenAI standard)
OPENAI_TPD_LIMIT = 2000000  # Tokens per day limit (OpenAI standard)

# Concurrency control - Allow more concurrent requests
OPENAI_CONCURRENT_LIMIT = 3  # Maximum concurrent requests (OpenAI allows 3)

# Retry configuration - Optimized for OpenAI
OPENAI_RETRY_ATTEMPTS = 5  # Retry attempts for rate limit handling

# Delay configuration - Optimized for higher rate limits
OPENAI_BASE_DELAY = 5  # Base delay for rate limit compliance (seconds)

# Advanced configuration options
# Note: GPT-5 models only support default temperature (1.0)
# Custom temperature values are not supported
OPENAI_TEMPERATURE = 1.0  # Default temperature for GPT-5 models
OPENAI_MAX_TOKENS = 4000  # Adjusted tokens for GPT-5-mini
GPT5_MINI_MODEL = "gpt-5-mini"  # GPT-5-mini model identifier

# Token tracking for TPM/TPD limits
_token_usage_minute = []
_token_usage_day = []
_last_reset_minute = time.time()
_last_reset_day = time.time()

def calculate_rate_limit_delay():
    """
    Calculate the appropriate delay between requests based on RPM limit.
    Optimized for OpenAI's higher rate limits.
    
    Returns:
        int: Delay in seconds between requests
    """
    # Ensure at least 60/RPM_LIMIT seconds between requests
    # For OpenAI's high RPM, this will be very small, so we use a minimum value
    base_delay = max(OPENAI_BASE_DELAY, int(60 / min(OPENAI_RPM_LIMIT, 100)))
    
    # Add random jitter (±20%) to avoid synchronized requests
    jitter = random.uniform(0.8, 1.2)
    
    return int(base_delay * jitter)

def track_token_usage(token_count):
    """
    Track token usage for TPM and TPD limits.
    
    Args:
        token_count (int): Number of tokens used in the request
        
    Returns:
        bool: True if within limits, False if would exceed limits
    """
    global _token_usage_minute, _token_usage_day, _last_reset_minute, _last_reset_day
    
    current_time = time.time()
    
    # Reset minute counter if more than 60 seconds have passed
    if current_time - _last_reset_minute >= 60:
        _token_usage_minute = []
        _last_reset_minute = current_time
    
    # Reset day counter if more than 24 hours have passed
    if current_time - _last_reset_day >= 86400:  # 24 * 60 * 60 = 86400 seconds
        _token_usage_day = []
        _last_reset_day = current_time
    
    # Check current usage
    minute_tokens = sum(_token_usage_minute)
    day_tokens = sum(_token_usage_day)
    
    # Check if adding this request would exceed limits
    if minute_tokens + token_count > OPENAI_TPM_LIMIT:
        return False
    if day_tokens + token_count > OPENAI_TPD_LIMIT:
        return False
    
    # Record the token usage
    _token_usage_minute.append(token_count)
    _token_usage_day.append(token_count)
    
    return True

def get_token_usage_stats():
    """
    Get current token usage statistics.
    
    Returns:
        dict: Token usage statistics
    """
    minute_tokens = sum(_token_usage_minute)
    day_tokens = sum(_token_usage_day)
    
    return {
        'minute_tokens': minute_tokens,
        'day_tokens': day_tokens,
        'minute_remaining': OPENAI_TPM_LIMIT - minute_tokens,
        'day_remaining': OPENAI_TPD_LIMIT - day_tokens,
        'minute_percentage': (minute_tokens / OPENAI_TPM_LIMIT) * 100,
        'day_percentage': (day_tokens / OPENAI_TPD_LIMIT) * 100
    }

def get_api_config_summary():
    """
    Get a summary of the current API configuration.
    
    Returns:
        dict: Configuration summary
    """
    token_stats = get_token_usage_stats()
    return {
        "rpm_limit": OPENAI_RPM_LIMIT,
        "tpm_limit": OPENAI_TPM_LIMIT,
        "tpd_limit": OPENAI_TPD_LIMIT,
        "concurrent_limit": OPENAI_CONCURRENT_LIMIT,
        "retry_attempts": OPENAI_RETRY_ATTEMPTS,
        "base_delay": OPENAI_BASE_DELAY,
        "calculated_delay": calculate_rate_limit_delay(),
        "temperature": OPENAI_TEMPERATURE,
        "max_tokens": OPENAI_MAX_TOKENS,
        "model": GPT5_MINI_MODEL,
        "token_usage": token_stats
    }

def print_config_summary():
    """
    Print the current configuration summary to console.
    """
    config = get_api_config_summary()
    token_usage = config['token_usage']
    
    print("=== GPT-5-mini API Configuration ===")
    print(f"RPM Limit: {config['rpm_limit']}")
    print(f"TPM Limit: {config['tpm_limit']:,}")
    print(f"TPD Limit: {config['tpd_limit']:,}")
    print(f"Concurrent Requests: {config['concurrent_limit']}")
    print(f"Retry Attempts: {config['retry_attempts']}")
    print(f"Base Delay: {config['base_delay']}s")
    print(f"Calculated Delay: {config['calculated_delay']}s")
    print(f"Temperature: {config['temperature']}")
    print(f"Max Tokens: {config['max_tokens']}")
    print(f"Model: {config['model']}")
    print("=" * 40)
    print("=== Token Usage Statistics ===")
    print(f"Minute Usage: {token_usage['minute_tokens']:,}/{config['tpm_limit']:,} ({token_usage['minute_percentage']:.1f}%)")
    print(f"Day Usage: {token_usage['day_tokens']:,}/{config['tpd_limit']:,} ({token_usage['day_percentage']:.1f}%)")
    print("=" * 40)

# Configuration presets for different API plans
PRESETS = {
    "development": {
        "OPENAI_RPM_LIMIT": 60,
        "OPENAI_TPM_LIMIT": 90000,
        "OPENAI_TPD_LIMIT": 2000000,
        "OPENAI_CONCURRENT_LIMIT": 3,
        "OPENAI_RETRY_ATTEMPTS": 5,
        "OPENAI_BASE_DELAY": 5,
        "OPENAI_MAX_TOKENS": 4000
    },
    "conservative": {
        "OPENAI_RPM_LIMIT": 30,
        "OPENAI_TPM_LIMIT": 60000,
        "OPENAI_TPD_LIMIT": 1500000,
        "OPENAI_CONCURRENT_LIMIT": 2,
        "OPENAI_RETRY_ATTEMPTS": 5,
        "OPENAI_BASE_DELAY": 10,
        "OPENAI_MAX_TOKENS": 3000
    },
    "high_throughput": {
        "OPENAI_RPM_LIMIT": 120,
        "OPENAI_TPM_LIMIT": 150000,
        "OPENAI_TPD_LIMIT": 3000000,
        "OPENAI_CONCURRENT_LIMIT": 5,
        "OPENAI_RETRY_ATTEMPTS": 3,
        "OPENAI_BASE_DELAY": 2,
        "OPENAI_MAX_TOKENS": 5000
    }
}

def apply_preset(preset_name):
    """
    Apply a configuration preset.
    
    Args:
        preset_name (str): Name of the preset to apply
        
    Raises:
        ValueError: If preset name is not found
    """
    if preset_name not in PRESETS:
        raise ValueError(f"Preset '{preset_name}' not found. Available presets: {list(PRESETS.keys())}")
    
    preset = PRESETS[preset_name]
    global OPENAI_RPM_LIMIT, OPENAI_TPM_LIMIT, OPENAI_TPD_LIMIT, OPENAI_CONCURRENT_LIMIT, OPENAI_RETRY_ATTEMPTS, OPENAI_BASE_DELAY, OPENAI_MAX_TOKENS
    
    OPENAI_RPM_LIMIT = preset["OPENAI_RPM_LIMIT"]
    OPENAI_TPM_LIMIT = preset["OPENAI_TPM_LIMIT"]
    OPENAI_TPD_LIMIT = preset["OPENAI_TPD_LIMIT"]
    OPENAI_CONCURRENT_LIMIT = preset["OPENAI_CONCURRENT_LIMIT"]
    OPENAI_RETRY_ATTEMPTS = preset["OPENAI_RETRY_ATTEMPTS"]
    OPENAI_BASE_DELAY = preset["OPENAI_BASE_DELAY"]
    OPENAI_MAX_TOKENS = preset["OPENAI_MAX_TOKENS"]
    
    print(f"Applied preset: {preset_name}")
    print_config_summary()

if __name__ == "__main__":
    # When run as a script, display current configuration
    print_config_summary()
    
    print("\nAvailable presets:")
    for preset_name, preset_config in PRESETS.items():
        print(f"  - {preset_name}: RPM={preset_config['OPENAI_RPM_LIMIT']}, TPM={preset_config['OPENAI_TPM_LIMIT']:,}, Concurrency={preset_config['OPENAI_CONCURRENT_LIMIT']}")
    
    print("\nTo apply a preset, use:")
    print("  from openai_config import apply_preset")
    print("  apply_preset('free_tier')  # Apply free tier limits")
    
    print("\nRecommended for most users:")
    print("  apply_preset('free_tier')  # Optimized for standard OpenAI limits")
