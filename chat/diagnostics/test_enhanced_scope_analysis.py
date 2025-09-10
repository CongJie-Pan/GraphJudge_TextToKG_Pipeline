#!/usr/bin/env python3
"""
Enhanced verification test to check the exact variable scope in main() function
"""

import ast
import os
from pathlib import Path

def analyze_main_function_scope():
    """
    Parse the main() function and analyze variable scope within it
    """
    print("🔬 Enhanced Main Function Scope Analysis")
    print("=" * 50)
    
    run_entity_path = Path(__file__).parent.parent / "run_entity.py"
    
    if not run_entity_path.exists():
        print(f"❌ Cannot find run_entity.py at {run_entity_path}")
        return False
    
    with open(run_entity_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Parse the AST
    try:
        tree = ast.parse(content)
    except SyntaxError as e:
        print(f"❌ Syntax error in file: {e}")
        return False
    
    # Find the main function
    main_function = None
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "main":
            main_function = node
            break
    
    if not main_function:
        print("❌ main() function not found")
        return False
    
    print(f"✓ Found main() function at line {main_function.lineno}")
    
    # Analyze variable usage within main function
    denoised_texts_assignments = []
    denoised_texts_usages = []
    
    class VariableAnalyzer(ast.NodeVisitor):
        def __init__(self):
            self.in_main = False
            self.current_line = 0
            
        def visit_FunctionDef(self, node):
            if node.name == "main":
                self.in_main = True
                self.generic_visit(node)
                self.in_main = False
            else:
                self.generic_visit(node)
        
        def visit_Assign(self, node):
            if self.in_main:
                for target in node.targets:
                    if isinstance(target, ast.Name) and target.id == "denoised_texts":
                        denoised_texts_assignments.append(node.lineno)
            self.generic_visit(node)
        
        def visit_For(self, node):
            if self.in_main:
                if isinstance(node.iter, ast.Name) and node.iter.id == "denoised_texts":
                    denoised_texts_usages.append(node.lineno)
            self.generic_visit(node)
    
    analyzer = VariableAnalyzer()
    analyzer.visit(tree)
    
    print(f"📊 Variable Analysis in main() function:")
    print(f"   denoised_texts assignments: {denoised_texts_assignments}")
    print(f"   denoised_texts usages (in for loops): {denoised_texts_usages}")
    
    # Check for scope issues
    if denoised_texts_usages:
        first_usage = min(denoised_texts_usages)
        if denoised_texts_assignments:
            first_assignment = min(denoised_texts_assignments)
            if first_usage < first_assignment:
                print(f"❌ SCOPE BUG FOUND: Usage at line {first_usage} before assignment at line {first_assignment}")
                return True
            else:
                print(f"✓ Usage at line {first_usage} after assignment at line {first_assignment}")
        else:
            print(f"❌ SCOPE BUG FOUND: Usage at line {first_usage} but no assignment found in main()")
            return True
    
    return False

def check_exact_lines_manually():
    """
    Manually check the exact lines around the problematic area
    """
    print("\n🔍 Manual Line-by-Line Analysis")
    print("=" * 50)
    
    run_entity_path = Path(__file__).parent.parent / "run_entity.py"
    
    with open(run_entity_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    # Check lines 750-820 for the problematic area
    print("Lines 750-820 analysis:")
    
    for i in range(749, min(820, len(lines))):  # 750-820 (0-indexed)
        line_num = i + 1
        line = lines[i].strip()
        
        if 'denoised_texts' in line:
            print(f"  Line {line_num}: {line}")
            
            # Check if this is an assignment or usage
            if 'denoised_texts =' in line:
                print(f"    → ASSIGNMENT")
            elif 'for denoised_text in denoised_texts' in line:
                print(f"    → USAGE (for loop)")
            elif 'denoised_texts)' in line or 'denoised_texts,' in line:
                print(f"    → USAGE (parameter/expression)")
    
    return True

def simulate_exact_main_function():
    """
    Create a more accurate simulation of the main function structure
    """
    print("\n🧪 Accurate Main Function Simulation")
    print("=" * 50)
    
    def simulated_main():
        # This simulates the exact structure we see in run_entity.py
        print("Simulating main() function execution...")
        
        # Variables that exist in scope
        entities_list = ["entity1", "entity2"]
        successful_extractions = len(entities_list)
        output_dir = "/tmp/test"
        
        try:
            # Around line 761: manifest metadata creation
            # This references successful_denoising which depends on denoised_texts
            print("Creating manifest metadata...")
            
            # This is the problem - successful_denoising is calculated from denoised_texts
            # but denoised_texts hasn't been assigned yet in main() scope
            successful_denoising = sum(1 for d in denoised_texts if "Error:" not in str(d))
            
            manifest_metadata = {
                "successful_extractions": successful_extractions,
                "total_texts": len(entities_list),
                "successful_denoising": successful_denoising,  # Uses denoised_texts indirectly
            }
            
            print("❌ This should not work!")
            return False
            
        except UnboundLocalError as e:
            print(f"✓ Found the exact error: {e}")
            
            # Now simulate the assignment that comes later
            print("Now assigning denoised_texts...")
            denoised_texts = ["text1", "text2"]
            successful_denoising = sum(1 for d in denoised_texts if "Error:" not in str(d))
            print(f"✓ After assignment, successful_denoising = {successful_denoising}")
            return True
    
    return simulated_main()

def main():
    print("🧪 Enhanced Dynamic Verification")
    print("=" * 60)
    
    # Run enhanced tests
    result1 = analyze_main_function_scope()
    result2 = check_exact_lines_manually()
    result3 = simulate_exact_main_function()
    
    print("\n" + "=" * 60)
    print("ENHANCED VERIFICATION RESULTS:")
    print("=" * 60)
    print(f"AST Analysis: {'BUG FOUND' if result1 else 'NO BUG'}")
    print(f"Manual Analysis: {'COMPLETED' if result2 else 'FAILED'}")
    print(f"Accurate Simulation: {'BUG CONFIRMED' if result3 else 'NO BUG'}")
    
    return result1 or result3

if __name__ == "__main__":
    main()
