#!/usr/bin/env python3
"""
Simple test for validation logic fix
"""

import os
import sys
from pathlib import Path

# Add the chat directory to Python path
current_dir = Path(__file__).parent
chat_dir = current_dir if current_dir.name == 'chat' else current_dir.parent
sys.path.insert(0, str(chat_dir))

def test_simple_validation():
    """Simple test of validation function"""
    
    print("🔧 Testing Validation Logic Fix")
    print("=" * 40)
    
    # Test the path resolution directly
    os.environ['PIPELINE_OUTPUT_DIR'] = r"d:\AboutCoding\AI_Research\GraphJudge_TextToKG_CLI\datasets\KIMI_result_DreamOf_RedChamber\Graph_Iteration3"
    
    try:
        import path_resolver
        
        # Test from chat directory
        original_cwd = os.getcwd()
        os.chdir(chat_dir)
        
        print(f"Working directory: {os.getcwd()}")
        
        # Test path resolution
        resolved_path = path_resolver.resolve_pipeline_output(3, create=False)
        print(f"Resolved path: {resolved_path}")
        
        # Check if files exist in resolved location
        expected_files = ["test_entity.txt", "test_denoised.target"]
        files_found = []
        
        for filename in expected_files:
            file_path = Path(resolved_path) / filename
            if file_path.exists() and file_path.stat().st_size > 0:
                files_found.append(filename)
                print(f"✅ Found: {filename} ({file_path.stat().st_size} bytes)")
            else:
                print(f"❌ Missing: {filename}")
        
        success = len(files_found) == len(expected_files)
        
        if success:
            print("\n✅ SUCCESS: All ECTD output files found!")
            print("The path resolution fix is working correctly!")
        else:
            print(f"\n❌ FAILURE: Only found {len(files_found)}/{len(expected_files)} files")
        
        return success
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return False
        
    finally:
        os.chdir(original_cwd)
        if 'PIPELINE_OUTPUT_DIR' in os.environ:
            del os.environ['PIPELINE_OUTPUT_DIR']

if __name__ == "__main__":
    success = test_simple_validation()
    print(f"\nTest result: {'PASSED' if success else 'FAILED'}")
    sys.exit(0 if success else 1)
