"""
Configuration Module for API Settings

This module handles the loading and validation of AI API credentials from environment variables.
It provides a centralized way to manage API configuration across all chat-related scripts while
maintaining security best practices by avoiding hardcoded credentials.

Supports Azure OpenAI and standard OpenAI APIs.

Environment Variables (Priority: Azure > Standard OpenAI):
- AZURE_OPENAI_KEY: Your Azure OpenAI API key for authentication  
- AZURE_OPENAI_ENDPOINT: Azure OpenAI endpoint URL (e.g., https://your-resource.openai.azure.com/)
- OPENAI_API_KEY: Your standard OpenAI API key for authentication (fallback)
- OPENAI_API_BASE: Base URL for OpenAI API endpoint (optional, defaults to OpenAI's standard endpoint)

Usage:
    from config import get_api_config
    api_key, api_base = get_api_config()
"""

"""
These imports bring in OS access for reading environment variables (os), 
type hints for clearer function signatures and return values (Tuple, Optional), 
and a cross-platform file-path utility for locating/reading the local .env file (Path)—written this way to make configuration loading safe, 
self-documenting, and portable.
"""
import os
from typing import Tuple, Optional
from pathlib import Path


def load_env_file():
    """
    Load environment variables from .env file if it exists.
    This allows storing API credentials securely in a local file.
    """
    env_path = Path(__file__).parent / '.env'
    if env_path.exists():
        with open(env_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    key = key.strip()
                    value = value.strip().strip('"').strip("'")
                    if key and value:
                        os.environ[key] = value


def get_api_config() -> Tuple[str, str]:
    """
    Retrieve OpenAI API configuration from environment variables or .env file.
    Supports both Azure OpenAI and standard OpenAI APIs.
    
    Returns:
        Tuple[str, str]: A tuple containing (api_key, api_base)
        
    Raises:
        ValueError: If neither AZURE_OPENAI_KEY nor OPENAI_API_KEY is set
        
    This function loads the API configuration from environment variables and provides
    appropriate defaults. It prioritizes Azure OpenAI if available, then falls back
    to standard OpenAI API. It also loads from .env file if present.
    """
    # Load from .env file first
    load_env_file()
    
    # Try Azure OpenAI first
    azure_key = os.getenv('AZURE_OPENAI_KEY', '').strip()
    azure_endpoint = os.getenv('AZURE_OPENAI_ENDPOINT', '').strip()
    
    if azure_key and azure_endpoint:
        return azure_key, azure_endpoint
    
    # Fall back to standard OpenAI
    api_key = os.getenv('OPENAI_API_KEY', '').strip()
    
    # Validate that at least one API key is provided
    if not api_key:
        raise ValueError(
            "API key not found. Please set either:\n"
            "- AZURE_OPENAI_KEY and AZURE_OPENAI_ENDPOINT for Azure OpenAI, or\n"
            "- OPENAI_API_KEY for standard OpenAI API\n"
            "- Create a .env file in the chat/ directory with your credentials"
        )
    
    # Get API base URL from environment variable with fallback to default
    api_base = os.getenv('OPENAI_API_BASE', 'https://api.openai.com/v1').strip()
    
    # Ensure API base URL ends with proper format
    if not api_base.endswith('/v1'):
        if not api_base.endswith('/'):
            api_base += '/'
        api_base += 'v1'
    
    return api_key, api_base


def validate_api_config() -> bool:
    """
    Validate that the API configuration is properly set up.
    
    Returns:
        bool: True if configuration is valid, False otherwise
        
    This function performs a basic validation check on the API configuration
    without raising exceptions, making it useful for testing and configuration
    verification scenarios.
    """
    try:
        api_key, api_base = get_api_config()
        
        # Basic validation checks
        if len(api_key) < 10:  # API keys should be reasonably long
            return False
            
        if not api_base.startswith(('http://', 'https://')):
            return False
            
        return True
        
    except (ValueError, TypeError):
        return False


def get_api_key() -> str:
    """
    Convenience function to get only the API key.
    
    Returns:
        str: OpenAI API key
        
    Raises:
        ValueError: If OPENAI_API_KEY is not set or is empty
    """
    api_key, _ = get_api_config()
    return api_key


def get_api_base() -> str:
    """
    Convenience function to get only the API base URL.
    
    Returns:
        str: OpenAI API base URL
    """
    _, api_base = get_api_config()
    return api_base


# Optional: Load configuration on module import for validation
if __name__ == "__main__":
    # When run as a script, perform configuration validation
    print("=== API Configuration Validation ===")
    
    # Test OpenAI API configuration
    try:
        api_key, api_base = get_api_config()
        print(f"✓ OpenAI API Key loaded: {api_key[:8]}...")
        print(f"✓ OpenAI API Base: {api_base}")
        print("✓ OpenAI Configuration is valid!")
    except ValueError as e:
        print(f"✗ OpenAI Configuration error: {e}")
    
    print("\n=== Validation Complete ===")
    
    # Exit with error code only if OpenAI config is missing
    try:
        get_api_config()
    except ValueError:
        exit(1) 