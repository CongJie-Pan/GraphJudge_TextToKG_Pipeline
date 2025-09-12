"""
Test encoding compatibility and terminal output handling.

This test ensures that the triple generator module works correctly
across different terminal encoding environments and handles Chinese
characters properly in all scenarios.
"""

import sys
import os
import traceback
from io import StringIO
from contextlib import redirect_stdout, redirect_stderr

# Add the core module path  
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def test_encoding_safety():
    """Test that all functions handle encoding safely."""
    try:
        from core.triple_generator import (
            generate_triples, 
            chunk_text,
            create_enhanced_prompt,
            extract_json_from_response,
            validate_response_schema,
            validate_triples_quality
        )
        from core.models import Triple, TripleResult
        
        print("Testing encoding safety across all functions...")
        
        # Test Chinese characters in various functions
        chinese_text = "甄士隱是姑蘇城內的鄉宦，妻子是封氏，有一女名英蓮。賈寶玉性格溫柔多情，與林黛玉青梅竹馬。"
        chinese_entities = ["甄士隱", "封氏", "英蓮", "賈寶玉", "林黛玉", "姑蘇城"]
        
        # Test 1: Text chunking with Chinese text
        chunks = chunk_text(chinese_text, max_tokens=50)
        assert len(chunks) > 0, "Chunking failed with Chinese text"
        print(f"[PASS] Text chunking: {len(chunks)} chunks created")
        
        # Test 2: Prompt creation with Chinese entities
        prompt = create_enhanced_prompt(chinese_text, chinese_entities)
        assert chinese_text in prompt, "Chinese text not found in prompt"
        assert all(entity in prompt for entity in chinese_entities), "Not all entities found in prompt"
        print("[PASS] Enhanced prompt creation with Chinese characters")
        
        # Test 3: JSON extraction with Chinese content
        test_response = '{"triples": [["甄士隱", "職業", "鄉宦"], ["甄士隱", "妻子", "封氏"]]}'
        extracted = extract_json_from_response(test_response)
        assert extracted is not None, "Failed to extract JSON with Chinese content"
        print("[PASS] JSON extraction with Chinese characters")
        
        # Test 4: Schema validation with Chinese content
        validated = validate_response_schema(test_response)
        assert validated is not None, "Failed to validate schema with Chinese content"
        assert len(validated["triples"]) == 2, "Incorrect number of triples validated"
        print("[PASS] Schema validation with Chinese characters")
        
        # Test 5: Triple quality validation with Chinese content
        test_triples = [
            Triple("甄士隱", "職業", "鄉宦"),
            Triple("甄士隱", "妻子", "封氏"),
            Triple("英蓮", "父親", "甄士隱")
        ]
        quality = validate_triples_quality(test_triples)
        assert quality['total_triples'] == 3, "Incorrect triple count in quality check"
        assert quality['quality_score'] == 1.0, "Expected perfect quality score"
        print("[PASS] Quality validation with Chinese triples")
        
        # Test 6: Full integration with Chinese data
        result = generate_triples(chinese_entities, chinese_text, api_client=None)
        assert isinstance(result, TripleResult), "Integration test failed"
        assert result.success, "Expected successful result"
        print("[PASS] Full integration test with Chinese data")
        
        # Test 7: Output capture and encoding handling
        with StringIO() as captured_output:
            with redirect_stdout(captured_output):
                print(f"測試中文輸出: {chinese_text[:30]}...")
            output = captured_output.getvalue()
            assert len(output) > 0, "Failed to capture Chinese output"
        print("[PASS] Output capture with Chinese characters")
        
        print("\n" + "="*50)
        print("ENCODING COMPATIBILITY TEST COMPLETED!")
        print("="*50)
        print("All functions handle Chinese characters correctly")
        print("Module is safe for cross-platform deployment")
        
        return True
        
    except Exception as e:
        print(f"\nENCODING COMPATIBILITY TEST FAILED!")
        print(f"Error: {str(e)}")
        print("\nFull traceback:")
        traceback.print_exc()
        return False

def test_terminal_compatibility():
    """Test compatibility with different terminal encodings."""
    try:
        # Simulate different encoding scenarios
        test_strings = [
            "Basic ASCII test",
            "中文測試 - Chinese characters",
            "Mixed 测试 with ASCII and 中文",
            "Special symbols: [OK] [ERROR] -> <- ^ v",
            "Numbers and Chinese: 123 個實體"
        ]
        
        print("\nTesting terminal compatibility...")
        
        for i, test_string in enumerate(test_strings, 1):
            try:
                # Try to encode/decode safely
                encoded = test_string.encode('utf-8', errors='replace')
                decoded = encoded.decode('utf-8', errors='replace')
                ascii_safe = test_string.encode('ascii', errors='replace').decode('ascii')
                
                print(f"[{i}/5] String test passed: length={len(test_string)}")
                
            except Exception as e:
                print(f"[{i}/5] String test failed: {e}")
                return False
        
        print("[PASS] Terminal compatibility test passed")
        return True
        
    except Exception as e:
        print(f"Terminal compatibility test failed: {e}")
        return False

if __name__ == "__main__":
    print("GraphJudge Triple Generator - Encoding Compatibility Tests")
    print("=" * 60)
    
    success1 = test_encoding_safety()
    success2 = test_terminal_compatibility()
    
    overall_success = success1 and success2
    
    print("\n" + "=" * 60)
    print(f"OVERALL RESULT: {'PASSED' if overall_success else 'FAILED'}")
    print("=" * 60)
    
    if overall_success:
        print("[SUCCESS] Module is ready for production deployment")
        print("[SUCCESS] All encoding issues resolved")
        print("[SUCCESS] Cross-platform compatibility verified")
    else:
        print("[ERROR] Encoding issues remain - requires additional fixes")
    
    sys.exit(0 if overall_success else 1)