#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
convert_Judge_To_jsonGraph.py

This module converts GraphJudge CSV evaluation results into JSON knowledge graph format
that is compatible with the kgGenShows visualization system.

Author: Assistant Engineer
Date: 2025-01-15
Purpose: Transform CSV triplet evaluation results to JSON knowledge graph format
         for visualization in the kgGenShows system
"""

import csv
import json
import re
import os
import sys
from datetime import datetime
from typing import Dict, List, Set, Tuple, Optional
from collections import defaultdict


class TripleExtractor:
    """
    Extract and parse knowledge graph triplets from natural language prompts.
    
    This class handles the parsing of "Is this true:" questions to extract
    subject-predicate-object triplets from the evaluation data.
    """
    
    def __init__(self):
        """Initialize the extractor with pattern matching rules."""
        # Pattern to match "Is this true: [subject] [predicate] [object] ?"
        self.pattern = re.compile(r'Is this true:\s*(.+?)\s+(.+?)\s+(.+?)\s*\?', re.UNICODE)
        
        # Common predicate patterns in Chinese
        self.action_patterns = ['行為', '地點', '作品', '作者', '職業', '妻子', '女兒', 
                               '父親', '岳丈', '主人', '談論內容', '囑咐', '包含',
                               '認為', '幫助', '贈送', '知道', '抱', '看見', '聽見']
    
    def extract_triple(self, prompt: str) -> Optional[Tuple[str, str, str]]:
        """
        Extract subject, predicate, object from a prompt string.
        
        Args:
            prompt (str): The evaluation prompt in format "Is this true: [triple] ?"
            
        Returns:
            Optional[Tuple[str, str, str]]: (subject, predicate, object) or None if parsing fails
            
        Example:
            Input: "Is this true: 士隱 地點 書房 ?"
            Output: ("士隱", "地點", "書房")
        """
        try:
            # Remove extra whitespace and normalize
            prompt = re.sub(r'\s+', ' ', prompt.strip())
            
            match = self.pattern.match(prompt)
            if not match:
                # Fallback: try to extract from the part after "Is this true:"
                if "Is this true:" in prompt:
                    content = prompt.split("Is this true:")[-1].strip()
                    content = content.rstrip("?").strip()
                    
                    # Try to split by known action patterns
                    for pattern in self.action_patterns:
                        if pattern in content:
                            parts = content.split(pattern, 1)
                            if len(parts) == 2:
                                subject = parts[0].strip()
                                obj = parts[1].strip()
                                return (subject, pattern, obj)
                    
                    # If no pattern matched, try simple split
                    parts = content.split()
                    if len(parts) >= 3:
                        # Take first as subject, last as object, middle as predicate
                        subject = parts[0]
                        obj = parts[-1]
                        predicate = ' '.join(parts[1:-1])
                        return (subject, predicate, obj)
                
                return None
            
            # Extract the three groups from regex match
            full_content = match.group(1) + " " + match.group(2) + " " + match.group(3)
            
            # Try to identify subject, predicate, object by patterns
            for pattern in self.action_patterns:
                if pattern in full_content:
                    parts = full_content.split(pattern, 1)
                    if len(parts) == 2:
                        subject = parts[0].strip()
                        obj = parts[1].strip()
                        return (subject, pattern, obj)
            
            # Fallback: split into three parts
            parts = full_content.split()
            if len(parts) >= 3:
                subject = parts[0]
                obj = parts[-1]
                predicate = ' '.join(parts[1:-1])
                return (subject, predicate, obj)
                
            return None
            
        except Exception as e:
            print(f"Error extracting triple from '{prompt}': {str(e)}")
            return None


class KnowledgeGraphConverter:
    """
    Convert CSV evaluation results to JSON knowledge graph format.
    
    This class processes the evaluation CSV file and converts valid triplets
    (those marked as "Yes") into a JSON format compatible with kgGenShows.
    """
    
    def __init__(self, csv_file_path: str, output_dir: str):
        """
        Initialize the converter.
        
        Args:
            csv_file_path (str): Path to the input CSV file
            output_dir (str): Directory to save the output JSON file
        """
        self.csv_file_path = csv_file_path
        self.output_dir = output_dir
        self.extractor = TripleExtractor()
        
        # Data storage
        self.entities: Set[str] = set()
        self.relationships: List[str] = []
        self.valid_triplets: List[Tuple[str, str, str]] = []
        self.invalid_triplets: List[Tuple[str, str, str]] = []
        
        # Statistics
        self.stats = {
            'total_rows': 0,
            'valid_triplets': 0,
            'invalid_triplets': 0,
            'parsing_errors': 0,
            'unique_entities': 0,
            'unique_relationships': 0
        }
    
    def load_and_parse_csv(self) -> bool:
        """
        Load and parse the CSV file.
        
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            print(f"Loading CSV file: {self.csv_file_path}")
            
            if not os.path.exists(self.csv_file_path):
                print(f"Error: CSV file not found: {self.csv_file_path}")
                return False
            
            with open(self.csv_file_path, 'r', encoding='utf-8') as file:
                reader = csv.DictReader(file)
                
                for row_num, row in enumerate(reader, 1):
                    self.stats['total_rows'] += 1
                    
                    prompt = row.get('prompt', '').strip()
                    generated = row.get('generated', '').strip()
                    
                    if not prompt or not generated:
                        print(f"Warning: Empty data in row {row_num}")
                        continue
                    
                    # Extract triple from prompt
                    triple = self.extractor.extract_triple(prompt)
                    
                    if triple is None:
                        print(f"Warning: Could not parse triple from row {row_num}: {prompt}")
                        self.stats['parsing_errors'] += 1
                        continue
                    
                    subject, predicate, obj = triple
                    
                    # Check if the relationship is valid (marked as "Yes")
                    if generated.lower() == 'yes':
                        self.valid_triplets.append((subject, predicate, obj))
                        self.stats['valid_triplets'] += 1
                        
                        # Add entities
                        self.entities.add(subject)
                        self.entities.add(obj)
                        
                        # Add relationship in the format "subject - predicate - object"
                        relationship = f"{subject} - {predicate} - {obj}"
                        self.relationships.append(relationship)
                        
                        print(f"✓ Valid: {relationship}")
                        
                    else:
                        self.invalid_triplets.append((subject, predicate, obj))
                        self.stats['invalid_triplets'] += 1
                        print(f"✗ Invalid: {subject} - {predicate} - {obj}")
            
            # Update final statistics
            self.stats['unique_entities'] = len(self.entities)
            self.stats['unique_relationships'] = len(self.relationships)
            
            print(f"\n=== Parsing Complete ===")
            print(f"Total rows processed: {self.stats['total_rows']}")
            print(f"Valid triplets: {self.stats['valid_triplets']}")
            print(f"Invalid triplets: {self.stats['invalid_triplets']}")
            print(f"Parsing errors: {self.stats['parsing_errors']}")
            print(f"Unique entities: {self.stats['unique_entities']}")
            print(f"Unique relationships: {self.stats['unique_relationships']}")
            
            return True
            
        except Exception as e:
            print(f"Error loading CSV file: {str(e)}")
            return False
    
    def generate_json_output(self) -> Dict:
        """
        Generate the JSON output in kgGenShows format.
        
        Returns:
            Dict: The knowledge graph data in JSON format
        """
        # Sort entities and relationships for consistent output
        sorted_entities = sorted(list(self.entities))
        
        # Create report section similar to the reference format
        report = {
            "summary": {
                "entities": len(sorted_entities),
                "relationships": len(self.relationships),
                "total_evaluated": self.stats['total_rows'],
                "valid_triplets": self.stats['valid_triplets'],
                "invalid_triplets": self.stats['invalid_triplets'],
                "parsing_errors": self.stats['parsing_errors'],
                "processing_time_sec": 0,  # Will be updated when saving
                "source_file": os.path.basename(self.csv_file_path)
            },
            "processing_summary": {
                "total_evaluated": self.stats['total_rows'],
                "valid_extracted": self.stats['valid_triplets'],
                "conversion_accuracy": round(self.stats['valid_triplets'] / max(1, self.stats['total_rows']) * 100, 2)
            },
            "quality_metrics": {
                "entity_count": len(sorted_entities),
                "relationship_count": len(self.relationships),
                "data_source": "GraphJudge evaluation results",
                "filter_criteria": "Only relationships marked as 'Yes' included"
            }
        }
        
        json_data = {
            "entities": sorted_entities,
            "relationships": self.relationships,
            "report": report,
            "metadata": {
                "source_type": "graphjudge_evaluation",
                "conversion_timestamp": datetime.now().isoformat(),
                "converter_version": "1.0.0",
                "input_file": os.path.basename(self.csv_file_path)
            }
        }
        
        return json_data
    
    def save_json_file(self, json_data: Dict) -> str:
        """
        Save the JSON data to a file.
        
        Args:
            json_data (Dict): The JSON data to save
            
        Returns:
            str: Path to the saved file
        """
        # Create output directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Generate output filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_filename = f"converted_kg_from_judge_{timestamp}.json"
        output_path = os.path.join(self.output_dir, output_filename)
        
        try:
            # Update processing time in the report
            start_time = datetime.now()
            
            with open(output_path, 'w', encoding='utf-8') as file:
                json.dump(json_data, file, ensure_ascii=False, indent=2)
            
            end_time = datetime.now()
            processing_time = (end_time - start_time).total_seconds()
            
            # Update the processing time in the saved file
            json_data['report']['summary']['processing_time_sec'] = processing_time
            
            with open(output_path, 'w', encoding='utf-8') as file:
                json.dump(json_data, file, ensure_ascii=False, indent=2)
            
            print(f"\n=== Output Saved ===")
            print(f"File saved to: {output_path}")
            print(f"File size: {os.path.getsize(output_path)} bytes")
            
            return output_path
            
        except Exception as e:
            print(f"Error saving JSON file: {str(e)}")
            return ""
    
    def run_conversion(self) -> bool:
        """
        Run the complete conversion process.
        
        Returns:
            bool: True if conversion successful, False otherwise
        """
        print("=== Knowledge Graph Converter ===")
        print("Converting GraphJudge CSV results to JSON knowledge graph format")
        print("Compatible with kgGenShows visualization system\n")
        
        # Step 1: Load and parse CSV
        if not self.load_and_parse_csv():
            return False
        
        # Step 2: Generate JSON output
        print("\nGenerating JSON output...")
        json_data = self.generate_json_output()
        
        # Step 3: Save to file
        print("Saving JSON file...")
        output_path = self.save_json_file(json_data)
        
        if output_path:
            print(f"\n✅ Conversion completed successfully!")
            print(f"Output file: {output_path}")
            print(f"Ready for use with kgGenShows visualization system")
            return True
        else:
            print(f"\n❌ Conversion failed!")
            return False


def main():
    """
    Main function to run the converter.
    """
    # File paths - support environment variables for pipeline integration
    iteration = int(os.environ.get('PIPELINE_ITERATION', '2'))
    csv_file_path = os.environ.get('PIPELINE_INPUT_FILE', 
                                 f"Miscellaneous/KgGen/GraphJudge/datasets/KIMI_result_DreamOf_RedChamber/Graph_Iteration{iteration}/pred_instructions_context_gemini_itr{iteration}.csv")
    output_dir = os.environ.get('PIPELINE_OUTPUT_DIR', 
                               f"Miscellaneous/KgGen/GraphJudge/datasets/KIMI_result_DreamOf_RedChamber/Graph_Iteration{iteration}")
    
    print(f"🔧 Using Iteration: {iteration}")
    print(f"📥 Input CSV file: {csv_file_path}")
    print(f"📤 Output directory: {output_dir}")
    
    # Check if running from correct directory
    if not os.path.exists(csv_file_path):
        # Try alternative path structures
        alternative_paths = [
            os.path.join(os.getcwd(), "datasets", "KIMI_result_DreamOf_RedChamber", "Graph_Iteration2", "pred_instructions_context_gemini_itr2.csv"),
            os.path.join(os.path.dirname(__file__), "..", "datasets", "KIMI_result_DreamOf_RedChamber", "Graph_Iteration2", "pred_instructions_context_gemini_itr2.csv"),
            os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", csv_file_path)
        ]
        
        found = False
        for alt_path in alternative_paths:
            if os.path.exists(alt_path):
                csv_file_path = alt_path
                output_dir = os.path.dirname(alt_path)
                found = True
                print(f"Found CSV file at: {csv_file_path}")
                break
        
        if not found:
            print(f"Error: CSV file not found at any of the following locations:")
            print(f"  - {csv_file_path}")
            for alt_path in alternative_paths:
                print(f"  - {alt_path}")
            print("Please ensure you're running this script from the correct directory")
            sys.exit(1)
    
    # Create and run converter
    converter = KnowledgeGraphConverter(csv_file_path, output_dir)
    success = converter.run_conversion()
    
    if success:
        print("\n=== Next Steps ===")
        print("1. Copy the generated JSON file to the kgGenShows/graphs/ directory")
        print("2. Open kgGenShows visualization system")
        print("3. Load the new graph file to visualize the knowledge graph")
        sys.exit(0)
    else:
        print("\nConversion failed. Please check the error messages above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
