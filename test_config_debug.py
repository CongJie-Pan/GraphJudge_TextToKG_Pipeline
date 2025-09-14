#!/usr/bin/env python3
"""
Test script to check if config changes are being loaded correctly.
"""

import sys
from pathlib import Path

# Add streamlit_pipeline to path
sys.path.insert(0, str(Path(__file__).parent / 'streamlit_pipeline'))

def test_config():
    """Test if config values are correct."""
    try:
        from streamlit_pipeline.core.config import get_model_config, OPENAI_MAX_TOKENS
        from streamlit_pipeline.utils.api_client import APIClient

        print("=== Config Debug Test ===")
        print(f"OPENAI_MAX_TOKENS constant: {OPENAI_MAX_TOKENS}")

        config = get_model_config()
        print(f"get_model_config()['max_tokens']: {config['max_tokens']}")
        print(f"get_model_config()['max_retries']: {config['max_retries']}")

        client = APIClient()
        print(f"APIClient.config['max_tokens']: {client.config['max_tokens']}")
        print(f"APIClient.config['max_retries']: {client.config['max_retries']}")

        # Check if the values match expectations
        expected_tokens = 8000
        if client.config['max_tokens'] == expected_tokens:
            print(f"✅ Config correctly shows {expected_tokens} tokens")
        else:
            print(f"❌ Config shows {client.config['max_tokens']} tokens, expected {expected_tokens}")

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_config()