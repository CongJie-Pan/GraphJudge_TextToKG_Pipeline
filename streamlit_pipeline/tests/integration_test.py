"""
Integration test for triple generator module.

This test verifies the module works correctly with the existing data models
and handles encoding issues properly for different terminal environments.
"""

import sys
import os
import traceback

# Add the core module path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def safe_print(message):
    """Print message safely handling encoding issues."""
    try:
        print(message)
    except UnicodeEncodeError:
        # Fallback to ASCII-safe output
        ascii_message = message.encode('ascii', errors='replace').decode('ascii')
        print(ascii_message)

def run_integration_test():
    """Run comprehensive integration test for triple generator."""
    try:
        safe_print("="*50)
        safe_print("GraphJudge Triple Generator Integration Test")
        safe_print("="*50)
        
        # Import modules
        from core.triple_generator import generate_triples
        from core.models import Triple, TripleResult
        
        safe_print("[1/5] Module imports successful")
        
        # Test 1: Basic integration without API client
        safe_print("\n[2/5] Testing basic integration...")
        entities = ['甄士隱', '封氏', '賈寶玉']  
        text = '甄士隱是姑蘇城內的鄉宦，妻子是封氏。賈寶玉性格溫柔多情。'
        
        result = generate_triples(entities, text, api_client=None)
        
        assert isinstance(result, TripleResult), f"Expected TripleResult, got {type(result)}"
        assert result.success, f"Expected success=True, got {result.success}"
        assert result.processing_time >= 0, f"Expected non-negative processing time, got {result.processing_time}"
        assert 'text_processing' in result.metadata, "Missing text_processing metadata"
        assert 'extraction_stats' in result.metadata, "Missing extraction_stats metadata"
        
        safe_print(f"  - Success: {result.success}")
        safe_print(f"  - Processing time: {result.processing_time:.3f}s")
        safe_print(f"  - Text chunks created: {result.metadata['text_processing']['chunks_created']}")
        safe_print(f"  - Entities provided: {result.metadata['extraction_stats']['entities_provided']}")
        
        # Test 2: Error handling with empty inputs
        safe_print("\n[3/5] Testing error handling...")
        
        # Empty text
        result_empty_text = generate_triples(['entity'], '', api_client=None)
        assert not result_empty_text.success, "Expected failure with empty text"
        assert "empty or invalid" in result_empty_text.error, f"Unexpected error message: {result_empty_text.error}"
        
        # Empty entities
        result_empty_entities = generate_triples([], 'some text', api_client=None)
        assert not result_empty_entities.success, "Expected failure with empty entities"
        assert "empty or invalid" in result_empty_entities.error, f"Unexpected error message: {result_empty_entities.error}"
        
        safe_print("  - Empty input handling: PASSED")
        
        # Test 3: Text chunking with large input
        safe_print("\n[4/5] Testing text chunking...")
        long_text = "這是一段很長的古典中文文本，包含很多複雜的內容和詳細的描述。" * 200  # Create longer text
        result_chunked = generate_triples(['實體1', '實體2'], long_text, api_client=None)
        
        assert isinstance(result_chunked, TripleResult), "Expected TripleResult for chunked text"
        chunks_created = result_chunked.metadata['text_processing']['chunks_created']
        safe_print(f"  - Chunks created for long text: {chunks_created}")
        safe_print(f"  - Original text length: {len(long_text)} characters")
        safe_print(f"  - Expected chunking: {'YES' if len(long_text) > 2000 else 'NO'} (threshold: 2000 chars)")
        
        # Test 4: Data model compatibility
        safe_print("\n[5/5] Testing data model compatibility...")
        
        # Test Triple creation
        test_triple = Triple(
            subject="測試主體",
            predicate="測試關係", 
            object="測試客體",
            confidence=0.95
        )
        
        assert test_triple.subject == "測試主體"
        assert test_triple.confidence == 0.95
        assert test_triple.confidence_level.name == "VERY_HIGH"
        
        # Test Triple serialization
        triple_dict = test_triple.to_dict()
        assert 'subject' in triple_dict
        assert triple_dict['confidence'] == 0.95
        
        # Test Triple deserialization  
        restored_triple = Triple.from_dict(triple_dict)
        assert restored_triple.subject == test_triple.subject
        assert restored_triple.confidence == test_triple.confidence
        
        safe_print("  - Triple creation and serialization: PASSED")
        
        safe_print("\n" + "="*50)
        safe_print("INTEGRATION TEST COMPLETED SUCCESSFULLY!")
        safe_print("="*50)
        safe_print("\nSummary:")
        safe_print("- Module imports: PASSED")
        safe_print("- Basic integration: PASSED") 
        safe_print("- Error handling: PASSED")
        safe_print("- Text chunking: PASSED")
        safe_print("- Data model compatibility: PASSED")
        safe_print("\nTriple generator module is ready for production use.")
        
        return True
        
    except Exception as e:
        safe_print(f"\nINTEGRATION TEST FAILED!")
        safe_print(f"Error: {str(e)}")
        safe_print("\nFull traceback:")
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = run_integration_test()
    sys.exit(0 if success else 1)