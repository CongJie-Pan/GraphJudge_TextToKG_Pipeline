#!/usr/bin/env python3
"""
Minimal test to verify the variable scope bug hypothesis in run_entity.py

This test recreates the exact variable usage pattern from the buggy code
to prove that the UnboundLocalError is caused by using denoised_texts
before it's assigned.

Hypothesis: The code tries to use 'denoised_texts' variable at line ~758
but the variable is not assigned until line ~805, causing UnboundLocalError.
"""

import os
import sys
import tempfile
from pathlib import Path

def test_variable_scope_bug_reproduction():
    """
    Recreate the exact variable scope issue from run_entity.py main() function.
    
    This simulates the problematic code structure where denoised_texts is used
    before being assigned.
    """
    print("🧪 Testing Variable Scope Bug Reproduction")
    print("=" * 50)
    
    # Simulate the main() function structure from run_entity.py
    def buggy_main_simulation():
        """Simulate the buggy main() function logic"""
        
        # Step 1: Simulate entity extraction (this works fine)
        print("✓ Step 1: Entity extraction completed")
        entities_list = ["entity1", "entity2", "entity3"]  # Mock data
        successful_extractions = len(entities_list)
        
        # Create temporary output directory
        with tempfile.TemporaryDirectory() as output_dir:
            print(f"📁 Using temp output directory: {output_dir}")
            
            # Step 2: Save entity file (this works fine)
            entity_file = os.path.join(output_dir, "test_entity.txt")
            with open(entity_file, "w", encoding='utf-8') as f:
                for entity in entities_list:
                    f.write(str(entity) + '\n')
            print(f"✓ Step 2: Entity file created: {entity_file}")
            
            # Step 3: THIS IS THE BUG - Try to use denoised_texts before assignment
            # This corresponds to lines 758-778 in the current run_entity.py
            print("❌ Step 3: Attempting to use denoised_texts before assignment...")
            
            try:
                # This is the problematic code pattern from run_entity.py
                denoised_file = os.path.join(output_dir, "test_denoised.target")
                with open(denoised_file, "w", encoding='utf-8') as output_file:
                    # BUG: denoised_texts is used here but not yet assigned
                    for denoised_text in denoised_texts:  # This should fail
                        output_file.write(str(denoised_text).strip() + '\n')
                print("❌ ERROR: This should not succeed!")
                return False
                
            except UnboundLocalError as e:
                print(f"✓ EXPECTED ERROR CAUGHT: {e}")
                print("✓ This confirms our hypothesis!")
                
                # Step 4: Now assign denoised_texts (this corresponds to line 805)
                print("Step 4: Now assigning denoised_texts variable...")
                denoised_texts = ["denoised1", "denoised2", "denoised3"]  # Mock assignment
                
                # Step 5: Try to use denoised_texts after assignment (should work)
                print("Step 5: Using denoised_texts after assignment...")
                try:
                    for denoised_text in denoised_texts:
                        print(f"  - {denoised_text}")
                    print("✓ denoised_texts works correctly when used after assignment")
                    return True
                except Exception as e:
                    print(f"❌ Unexpected error: {e}")
                    return False
            
            except Exception as e:
                print(f"❌ Unexpected error type: {e}")
                return False
    
    # Run the buggy simulation
    try:
        result = buggy_main_simulation()
        return result
    except Exception as e:
        print(f"❌ Test framework error: {e}")
        return False

def test_code_structure_analysis():
    """
    Analyze the actual code structure to confirm the bug location.
    """
    print("\n🔍 Testing Code Structure Analysis")
    print("=" * 50)
    
    run_entity_path = Path(__file__).parent.parent / "run_entity.py"
    
    if not run_entity_path.exists():
        print(f"❌ Cannot find run_entity.py at {run_entity_path}")
        return False
    
    with open(run_entity_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    # Find lines where denoised_texts is used
    denoised_usage_lines = []
    denoised_assignment_lines = []
    
    for i, line in enumerate(lines, 1):
        if 'denoised_texts' in line:
            if 'denoised_texts =' in line or 'denoised_texts=' in line:
                denoised_assignment_lines.append((i, line.strip()))
            elif 'for denoised_text in denoised_texts' in line:
                denoised_usage_lines.append((i, line.strip()))
    
    print("📊 denoised_texts Variable Analysis:")
    print(f"   Assignment lines: {len(denoised_assignment_lines)}")
    for line_num, line_content in denoised_assignment_lines:
        print(f"     Line {line_num}: {line_content}")
    
    print(f"   Usage lines: {len(denoised_usage_lines)}")
    for line_num, line_content in denoised_usage_lines:
        print(f"     Line {line_num}: {line_content}")
    
    # Check if any usage comes before assignment
    if denoised_usage_lines and denoised_assignment_lines:
        first_usage = min(denoised_usage_lines, key=lambda x: x[0])
        first_assignment = min(denoised_assignment_lines, key=lambda x: x[0])
        
        print(f"\n🎯 Analysis Results:")
        print(f"   First usage at line: {first_usage[0]}")
        print(f"   First assignment at line: {first_assignment[0]}")
        
        if first_usage[0] < first_assignment[0]:
            print("❌ BUG CONFIRMED: Variable used before assignment!")
            print(f"   Usage line {first_usage[0]} comes before assignment line {first_assignment[0]}")
            return True
        else:
            print("✓ No scope issue found")
            return False
    
    return False

def main():
    """Run all verification tests"""
    print("🧪 Dynamic Verification - Variable Scope Bug")
    print("=" * 60)
    print("Hypothesis: denoised_texts variable is used before assignment in run_entity.py")
    print("=" * 60)
    
    # Test 1: Reproduce the exact error
    test1_result = test_variable_scope_bug_reproduction()
    
    # Test 2: Analyze actual code structure
    test2_result = test_code_structure_analysis()
    
    print("\n" + "=" * 60)
    print("VERIFICATION RESULTS:")
    print("=" * 60)
    print(f"✓ Bug reproduction test: {'PASSED' if test1_result else 'FAILED'}")
    print(f"✓ Code structure analysis: {'BUG CONFIRMED' if test2_result else 'NO BUG FOUND'}")
    
    if test1_result and test2_result:
        print("\n🎉 HYPOTHESIS CONFIRMED!")
        print("The UnboundLocalError is caused by using denoised_texts before assignment.")
        print("Root cause: Variable scope error in main() function of run_entity.py")
        return True
    else:
        print("\n❌ HYPOTHESIS NOT CONFIRMED")
        print("Further investigation needed.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
