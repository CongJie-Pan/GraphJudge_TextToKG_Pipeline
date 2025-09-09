#!/usr/bin/env python3
"""
Simple test script to verify the strict Yes/No parsing logic implementation.

This script directly tests the parsing logic without the complexities of the full module.
It can be run standalone to verify that the implementation works correctly.
"""

import re

def test_strict_parsing_logic():
    """Test the strict Yes/No parsing logic as implemented in the improved run_kimi_gj.py"""
    
    def parse_response(cleaned_response):
        """Simplified version of the parsing logic from get_kimi_completion"""
        # Check if response matches strict Yes/No pattern (case-insensitive)
        if re.match(r'^yes$', cleaned_response, re.IGNORECASE):
            return "Yes"
        elif re.match(r'^no$', cleaned_response, re.IGNORECASE):
            return "No"
        else:
            # Treat other responses as format anomalies for later cleanup
            return f"FORMAT_ANOMALY: {cleaned_response}"
    
    # Test cases for valid responses
    valid_cases = [
        ("Yes", "Yes"),
        ("yes", "Yes"), 
        ("YES", "Yes"),
        ("yEs", "Yes"),
        ("No", "No"),
        ("no", "No"),
        ("NO", "No"),
        ("nO", "No")
    ]
    
    print("Testing valid Yes/No responses:")
    for input_resp, expected in valid_cases:
        result = parse_response(input_resp)
        status = "✓ PASS" if result == expected else "✗ FAIL"
        print(f"  {status}: '{input_resp}' -> '{result}' (expected: '{expected}')")
    
    # Test cases for format anomalies
    anomaly_cases = [
        "Yes, it is true.",
        "No, it is not true.", 
        "Maybe",
        "I don't know",
        "True",
        "False",
        "Correct",
        "Incorrect",
        "YES, this is correct",
        "NO, this is wrong",
        "",
        "  yes  ",  # Should fail because of spaces
        "yes\n",   # Should fail because of newline
    ]
    
    print("\nTesting format anomaly detection:")
    for input_resp in anomaly_cases:
        result = parse_response(input_resp)
        is_anomaly = result.startswith("FORMAT_ANOMALY:")
        status = "✓ PASS" if is_anomaly else "✗ FAIL"
        print(f"  {status}: '{input_resp}' -> detected as {'anomaly' if is_anomaly else 'valid'}")

def test_triple_extraction():
    """Test the triple extraction logic from instruction format"""
    
    def extract_triple(instruction):
        """Extract triple from instruction format"""
        return instruction.replace("Is this true: ", "").replace(" ?", "")
    
    test_cases = [
        ("Is this true: Apple Founded by Steve Jobs ?", "Apple Founded by Steve Jobs"),
        ("Is this true: 曹雪芹 創作 紅樓夢 ?", "曹雪芹 創作 紅樓夢"),
        ("Is this true: Mark Zuckerberg Founded Facebook ?", "Mark Zuckerberg Founded Facebook"),
        ("Is this true: 作者 作品 石頭記 ?", "作者 作品 石頭記")
    ]
    
    print("\nTesting triple extraction from instructions:")
    for instruction, expected in test_cases:
        result = extract_triple(instruction)
        status = "✓ PASS" if result == expected else "✗ FAIL"
        print(f"  {status}: '{instruction}' -> '{result}'")

def test_chinese_prompt_generation():
    """Test the Chinese prompt format generation"""
    
    def generate_chinese_prompt(triple_part):
        """Generate the Chinese one-shot prompt format"""
        return f"""任務：你需要判斷給定三元組陳述是否為事實正確。請僅輸出 Yes 或 No。
範例：
問題：這是真的嗎：曹雪芹 創作 紅樓夢？
答案：Yes
問題：這是真的嗎：馬克·祖克柏 創作 紅樓夢？
答案：No
現在的問題：這是真的嗎：{triple_part}？
答案："""
    
    test_triple = "Apple Founded by Steve Jobs"
    prompt = generate_chinese_prompt(test_triple)
    
    print("\nTesting Chinese prompt generation:")
    
    # Check required elements are present
    required_elements = [
        "任務：你需要判斷給定三元組陳述是否為事實正確。請僅輸出 Yes 或 No。",
        "範例：",
        "問題：這是真的嗎：曹雪芹 創作 紅樓夢？",
        "答案：Yes",
        "問題：這是真的嗎：馬克·祖克柏 創作 紅樓夢？",
        "答案：No",
        "現在的問題：這是真的嗎：Apple Founded by Steve Jobs？",
        "答案："
    ]
    
    all_present = True
    for element in required_elements:
        if element in prompt:
            print(f"  ✓ PASS: Found '{element[:50]}...'")
        else:
            print(f"  ✗ FAIL: Missing '{element[:50]}...'")
            all_present = False
    
    if all_present:
        print("  ✓ All required elements present in prompt")
    else:
        print("  ✗ Some required elements missing in prompt")

if __name__ == "__main__":
    print("=" * 70)
    print("Testing Judge Prompt Upgrade Implementation")
    print("=" * 70)
    
    try:
        test_strict_parsing_logic()
        test_triple_extraction()
        test_chinese_prompt_generation()
        
        print("\n" + "=" * 70)
        print("All tests completed. Check results above for any failures.")
        print("=" * 70)
        
    except Exception as e:
        print(f"\nTest execution failed with error: {e}")
        import traceback
        traceback.print_exc()
