#!/usr/bin/env python3
"""
Test script to verify the JSON format compatibility between run_triple.py output
and run_kimi_gj.py input requirements.
"""

import json
import os
from datasets import load_dataset

def test_json_format():
    """Test the JSON format compatibility."""
    
    # Test file path
    json_file = "../datasets/KIMI_result_DreamOf_RedChamber/Graph_Iteration1/test_instructions_context_kimi.json"
    
    print("🧪 Testing JSON format compatibility...")
    print(f"📁 Testing file: {json_file}")
    
    # Test 1: Basic JSON loading
    try:
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        print(f"✅ JSON loading successful: {len(data)} instructions")
    except Exception as e:
        print(f"❌ JSON loading failed: {e}")
        return False
    
    # Test 2: Check JSON structure
    if not isinstance(data, list):
        print(f"❌ Expected list, got {type(data)}")
        return False
    
    if len(data) == 0:
        print("❌ Empty instruction list")
        return False
    
    # Test 3: Check required fields
    required_fields = ["instruction", "input", "output"]
    sample = data[0]
    
    for field in required_fields:
        if field not in sample:
            print(f"❌ Missing required field: {field}")
            return False
    
    print(f"✅ JSON structure validation passed")
    
    # Test 4: Test with datasets library (used by run_kimi_gj.py)
    try:
        total_input = load_dataset("json", data_files=json_file)
        data_eval = total_input["train"]
        print(f"✅ Datasets library loading successful: {len(data_eval)} entries")
    except Exception as e:
        print(f"❌ Datasets library loading failed: {e}")
        return False
    
    # Test 5: Check instruction format
    sample_instruction = data_eval[0]["instruction"]
    if not sample_instruction.startswith("Is this true:"):
        print(f"❌ Unexpected instruction format: {sample_instruction}")
        return False
    
    print(f"✅ Instruction format validation passed")
    print(f"📋 Sample instruction: {sample_instruction}")
    
    # Test 6: Compatibility check
    print("\n📊 Compatibility Summary:")
    print(f"   📄 Total instructions: {len(data)}")
    print(f"   🔧 Format: JSON with required fields")
    print(f"   ⚖️  Compatible with run_kimi_gj.py: ✅ YES")
    print(f"   📋 Sample format: {sample_instruction}")
    
    return True

if __name__ == "__main__":
    print("=" * 60)
    print("🧪 JSON Format Compatibility Test")
    print("=" * 60)
    
    success = test_json_format()
    
    if success:
        print("\n🎉 All tests passed! The JSON format is fully compatible.")
    else:
        print("\n❌ Tests failed! Please check the JSON format.")
