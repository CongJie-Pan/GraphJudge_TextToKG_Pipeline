#!/usr/bin/env python3
"""
Quick debug script to test GPT-5-mini retry logic.
"""

from streamlit_pipeline.utils.api_client import APIClient

def test_gpt5_retry_debug():
    """Test if the retry logic is working correctly."""
    print("=== GPT-5-mini Retry Logic Debug Test ===")

    try:
        client = APIClient()
        print(f"Client max_retries: {client.config['max_retries']}")
        print(f"Client max_tokens: {client.config['max_tokens']}")

        # Try a simple call that should trigger the empty content issue
        response = client.call_gpt5_mini(
            "Extract entities from this text: 女媧氏於大荒山無稽崖煉石補天",
            "You are a helpful assistant that extracts entities.",
            max_tokens=100  # Small token limit to trigger the issue
        )

        print(f"Response received: '{response}'")
        print(f"Response length: {len(response)}")

    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    test_gpt5_retry_debug()