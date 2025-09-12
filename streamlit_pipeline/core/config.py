"""
Simplified configuration management for GraphJudge Streamlit Pipeline.

This module provides a clean, simplified configuration system that extracts
essential settings from the original complex configuration files while
maintaining compatibility with the core functionality.

Key simplifications:
- Unified API key management
- Essential model parameters only
- Streamlit-optimized settings
- No complex file-based caching
"""

import os
from typing import Tuple, Optional
from pathlib import Path


# Model configuration - simplified from openai_config.py
GPT5_MINI_MODEL = "gpt-5-mini"
PERPLEXITY_MODEL = "perplexity/sonar-reasoning"

# Essential API parameters
OPENAI_TEMPERATURE = 0.0  # Deterministic output
OPENAI_MAX_TOKENS = 4000  # Conservative token limit

# Simplified rate limiting (no complex tracking)
DEFAULT_TIMEOUT = 60  # Seconds
MAX_RETRIES = 3


def load_env_file() -> None:
    """
    Load environment variables from .env file if it exists.
    Simplified version of the original config.py function.
    """
    # Look for .env in multiple possible locations
    possible_paths = [
        Path(__file__).parent / '.env',  # streamlit_pipeline/core/.env
        Path(__file__).parent.parent / '.env',  # streamlit_pipeline/.env
        Path(__file__).parent.parent.parent / '.env',  # project root/.env
        Path(__file__).parent.parent.parent / 'chat' / '.env',  # chat/.env
    ]
    
    for env_path in possible_paths:
        if env_path.exists():
            _load_env_from_path(env_path)
            break


def _load_env_from_path(env_path: Path) -> None:
    """Load environment variables from a specific .env file path."""
    try:
        with open(env_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    key = key.strip()
                    value = value.strip().strip('"').strip("'")
                    if key and value:
                        os.environ[key] = value
    except IOError:
        pass  # Silently continue if file can't be read


def get_api_config(load_env: bool = True) -> Tuple[Optional[str], Optional[str]]:
    """
    Get API configuration with simplified logic.
    
    Args:
        load_env: Whether to load environment file (can be disabled for testing)
    
    Returns:
        Tuple of (api_key, api_base) where api_base may be None for default OpenAI endpoint
        
    Raises:
        ValueError: If no valid API key is found
    """
    # Load environment variables (optional for testing)
    if load_env:
        load_env_file()
    
    # Try Azure OpenAI first (priority from original config)
    azure_key = os.getenv('AZURE_OPENAI_KEY')
    azure_endpoint = os.getenv('AZURE_OPENAI_ENDPOINT')
    
    # Check for valid non-empty values
    if azure_key and azure_key.strip() and azure_endpoint and azure_endpoint.strip():
        return azure_key, azure_endpoint
    
    # Fall back to standard OpenAI
    openai_key = os.getenv('OPENAI_API_KEY')
    openai_base = os.getenv('OPENAI_API_BASE')  # Usually None for default endpoint
    
    if openai_key and openai_key.strip():
        return openai_key, openai_base
    
    # No valid configuration found
    raise ValueError(
        "No valid API configuration found. Please set either:\n"
        "- AZURE_OPENAI_KEY and AZURE_OPENAI_ENDPOINT for Azure OpenAI, or\n"  
        "- OPENAI_API_KEY for standard OpenAI"
    )


def get_api_key(load_env: bool = True) -> str:
    """
    Get just the API key, compatible with original chat/config.py interface.
    
    Args:
        load_env: Whether to load environment file (can be disabled for testing)
    
    Returns:
        API key string
        
    Raises:
        ValueError: If no valid API key is found
    """
    api_key, _ = get_api_config(load_env=load_env)
    if not api_key:
        raise ValueError("No API key found in configuration")
    return api_key


def get_model_config() -> dict:
    """
    Get model configuration parameters.
    
    Returns:
        Dictionary with model configuration
    """
    return {
        "entity_model": GPT5_MINI_MODEL,
        "triple_model": GPT5_MINI_MODEL, 
        "judgment_model": PERPLEXITY_MODEL,
        "temperature": OPENAI_TEMPERATURE,
        "max_tokens": OPENAI_MAX_TOKENS,
        "timeout": DEFAULT_TIMEOUT,
        "max_retries": MAX_RETRIES
    }