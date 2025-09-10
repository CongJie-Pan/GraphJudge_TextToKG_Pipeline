#!/usr/bin/env python3
"""
Test the fixed validation logic in stage_manager.py

This test verifies that the validation now correctly finds ECTD output files
using the centralized path_resolver instead of hardcoded relative paths.
"""

import os
import sys
from pathlib import Path

# Add the chat directory to Python path
current_dir = Path(__file__).parent
chat_dir = current_dir if current_dir.name == 'chat' else current_dir.parent
sys.path.insert(0, str(chat_dir))

def test_validation_fix():
    """Test that validation now works correctly"""
    
    print("🔧 Testing Fixed Validation Logic")
    print("=" * 50)
    
    # Set environment variable to point to the correct location
    actual_output_dir = r"d:\AboutCoding\AI_Research\GraphJudge_TextToKG_CLI\datasets\KIMI_result_DreamOf_RedChamber\Graph_Iteration3"
    
    # Test from chat directory (where validation might run)
    original_cwd = os.getcwd()
    
    try:
        os.chdir(chat_dir)
        print(f"Working directory: {os.getcwd()}")
        
        # Import and create a stage manager instance
        sys.path.insert(0, str(chat_dir / "cli"))
        from stage_manager import StageManager
        
        stage_manager = StageManager()
        
        # Set up environment similar to real execution
        env = {
            'PIPELINE_OUTPUT_DIR': actual_output_dir,
            'ECTD_OUTPUT_DIR': actual_output_dir,
            'PIPELINE_ITERATION': '3',
        }
        
        print(f"Testing validation with output dir: {actual_output_dir}")
        
        # Test the validation
        result = stage_manager._validate_stage_output("ectd", env)
        
        print(f"Validation result: {result}")
        
        if result:
            print("✅ SUCCESS: Validation correctly found ECTD output files!")
            print("The working directory dependency bug has been fixed!")
        else:
            print("❌ FAILURE: Validation still cannot find ECTD output files!")
            print("The bug may not be completely fixed.")
        
        return result
        
    except Exception as e:
        print(f"❌ Error during test: {e}")
        import traceback
        traceback.print_exc()
        return False
        
    finally:
        os.chdir(original_cwd)

if __name__ == "__main__":
    success = test_validation_fix()
    
    if success:
        print("\n🎉 Validation Fix Test PASSED!")
        print("The ECTD pipeline should now work end-to-end!")
    else:
        print("\n❌ Validation Fix Test FAILED!")
        print("Additional debugging may be needed.")
    
    sys.exit(0 if success else 1)
