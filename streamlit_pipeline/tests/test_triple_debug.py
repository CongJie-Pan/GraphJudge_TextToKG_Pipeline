#!/usr/bin/env python3
"""
Test script to debug GPT-5-mini API client with actual API call.
"""

import sys
import os
from pathlib import Path

# Add streamlit_pipeline to path
sys.path.insert(0, str(Path(__file__).parent / 'streamlit_pipeline'))

def test_gpt5_with_api():
    """Test GPT-5-mini API call with debug output."""
    try:
        from streamlit_pipeline.utils.api_client import APIClient
        from streamlit_pipeline.core.config import load_env_file

        # Load environment variables
        load_env_file()

        print("=== GPT-5-mini Debug Test with Real API ===")

        client = APIClient()
        print(f"Config: max_retries={client.config['max_retries']}, max_tokens={client.config['max_tokens']}")

        # Test with a prompt that should trigger GPT-5-mini reasoning mode issue
        prompt = """
        任務：分析古典中文文本，提取實體間的語義關係，輸出標準JSON格式的三元組。

        ## 輸出格式要求：
        ```json
        {
          "triples": [
            ["主體", "關係", "客體"]
          ]
        }
        ```

        ## 文本：
        女媧氏於大荒山無稽崖煉石補天，三萬六千五百塊石盡用，唯留一塊棄於青埂峰下。

        ## 實體列表：
        ["女媧氏", "大荒山", "無稽崖", "青埂峰", "石頭"]

        請提取三元組：
        """

        print("Making API call...")

        response = client.call_gpt5_mini(
            prompt,
            "你是一個專業的中文文本分析助手，專門提取語義關係。"
            # Use default max_tokens from config (should be 8000)
        )

        print(f"\n=== RESULT ===")
        print(f"Response: '{response}'")
        print(f"Length: {len(response)}")

        if not response:
            print("❌ Empty response - GPT-5-mini bug confirmed!")
        else:
            print("✅ Non-empty response - GPT-5-mini working!")

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_gpt5_with_api()