"""
KIMI Triple Parser and Post-processor

This module provides comprehensive post-processing functionality for KIMI-generated triples.
It addresses the key issues identified in the improvement plan:
1. Strips wrapper phrases and explanatory text
2. Deduplicates triples within and across responses  
3. Enforces unified relation vocabulary mapping
4. Validates and normalizes triple structure
5. Generates multiple output formats for different downstream tasks

Key improvements over the raw KIMI output:
- Removes ~40% data loss during JSON conversion
- Standardizes inconsistent relation vocabulary
- Eliminates duplicate triples
- Provides clean, structured output for Graph Judge

Usage:
    parser = KIMITripleParser()
    cleaned_triples = parser.parse_file('test_generated_graphs.txt')
    parser.save_outputs(cleaned_triples, 'output_directory/')
"""

import os
import json
import re
import logging
from typing import List, Dict, Set, Tuple, Any, Optional
from pathlib import Path
from dataclasses import dataclass
from collections import defaultdict, Counter
import hashlib

# Configure logging for detailed processing information
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class Triple:
    """
    Represents a knowledge graph triple with validation and normalization capabilities.
    
    A triple consists of subject, relation, and object components that form
    a semantic relationship in the knowledge graph.
    """
    subject: str
    relation: str
    object: str
    source_line: int = 0  # Line number from original file for tracking
    confidence: float = 1.0  # Confidence score for the triple
    
    def __post_init__(self):
        """Normalize triple components after initialization."""
        self.subject = self.subject.strip()
        self.relation = self.relation.strip()
        self.object = self.object.strip()
    
    def __hash__(self) -> int:
        """Generate hash for deduplication based on normalized content."""
        content = f"{self.subject}|{self.relation}|{self.object}"
        return hash(content)
    
    def __eq__(self, other) -> bool:
        """Check equality for deduplication."""
        if not isinstance(other, Triple):
            return False
        return (self.subject == other.subject and 
                self.relation == other.relation and 
                self.object == other.object)
    
    def to_list(self) -> List[str]:
        """Convert to list format for JSON serialization."""
        return [self.subject, self.relation, self.object]
    
    def to_statement(self) -> str:
        """Convert to statement format for Graph Judge."""
        return f"{self.subject} {self.relation} {self.object}"
    
    def is_valid(self) -> bool:
        """Validate triple has non-empty components."""
        return (len(self.subject) > 0 and 
                len(self.relation) > 0 and 
                len(self.object) > 0)

class RelationMapper:
    """
    Handles relation vocabulary mapping and standardization.
    
    Uses the unified relation vocabulary from config/relation_map.json
    to standardize inconsistent relation terms across different outputs.
    """
    
    def __init__(self, mapping_file: str = None):
        """Initialize with relation mapping configuration."""
        if mapping_file is None:
            # Default path relative to tools directory
            mapping_file = os.path.join(os.path.dirname(__file__), '..', 'config', 'relation_map.json')
        
        self.mapping_file = mapping_file
        self.relation_map = self._load_mapping()
        self.stats = defaultdict(int)  # Track mapping statistics
        
    def _load_mapping(self) -> Dict[str, str]:
        """Load and flatten relation mapping from JSON configuration."""
        try:
            with open(self.mapping_file, 'r', encoding='utf-8') as f:
                config = json.load(f)
            
            # Flatten all relation categories into a single mapping
            flat_mapping = {}
            relation_mapping = config.get('relation_mapping', {})
            
            for category, mappings in relation_mapping.items():
                flat_mapping.update(mappings)
            
            # Add default mapping
            default_mapping = config.get('default_mapping', {})
            flat_mapping.update(default_mapping)
            
            logger.info(f"Loaded {len(flat_mapping)} relation mappings from {self.mapping_file}")
            return flat_mapping
            
        except FileNotFoundError:
            logger.warning(f"Relation mapping file not found: {self.mapping_file}")
            return {}
        except json.JSONDecodeError as e:
            logger.error(f"Error parsing relation mapping file: {e}")
            return {}
    
    def map_relation(self, relation: str) -> str:
        """
        Map a Chinese relation to its standardized English equivalent.
        
        Args:
            relation: Original Chinese relation term
            
        Returns:
            Standardized English relation term
        """
        # Track original relation usage
        self.stats[f"original_{relation}"] += 1
        
        # Direct mapping lookup
        if relation in self.relation_map:
            mapped = self.relation_map[relation]
            self.stats[f"mapped_{mapped}"] += 1
            return mapped
        
        # If no mapping found, use original but log for future mapping
        logger.debug(f"No mapping found for relation: {relation}")
        self.stats["unmapped_relations"] += 1
        return relation
    
    def get_stats(self) -> Dict[str, int]:
        """Get mapping statistics for analysis."""
        return dict(self.stats)

class KIMITripleParser:
    """
    Main parser class for processing KIMI-generated triple outputs.
    
    Handles the complete pipeline from raw KIMI output to clean, standardized triples
    suitable for knowledge graph construction and Graph Judge evaluation.
    """
    
    def __init__(self, relation_mapper: RelationMapper = None):
        """Initialize parser with optional custom relation mapper."""
        self.relation_mapper = relation_mapper or RelationMapper()
        self.parsing_stats = defaultdict(int)
        self.all_triples: Set[Triple] = set()
        
        # Patterns for cleaning wrapper text
        self.wrapper_patterns = [
            r'^根據文本內容.*?：',
            r'^根據文本與實體.*?：',
            r'^根據文本「.*?」.*?：',
            r'^語義圖：',
            r'^以下是提取出的語義圖.*?：',
            r'^可抽取以下語義圖.*?：',
            r'^我將提取.*?：',
            r'^我來分析.*?：',
            r'^分析如下：',
            r'^.*?生成以下語義圖：',
            r'^.*?語義圖（三元組列表）：'
        ]
        
        # Patterns for extracting JSON arrays
        self.json_patterns = [
            r'\[\[.*?\]\]',  # Main pattern for nested arrays
            r'\[.*?\]',      # Fallback for simple arrays
        ]
    
    def clean_wrapper_text(self, raw_response: str) -> str:
        """
        Remove explanatory wrapper text from KIMI response.
        
        Args:
            raw_response: Raw response from KIMI API
            
        Returns:
            Cleaned text with wrapper phrases removed
        """
        cleaned = raw_response.strip()
        
        # Remove common wrapper patterns
        for pattern in self.wrapper_patterns:
            cleaned = re.sub(pattern, '', cleaned, flags=re.MULTILINE)
        
        # Remove any remaining leading/trailing whitespace
        cleaned = cleaned.strip()
        
        self.parsing_stats["wrapper_removals"] += 1
        return cleaned
    
    def extract_json_array(self, cleaned_text: str) -> Optional[str]:
        """
        Extract JSON array from cleaned text using multiple patterns.
        
        Args:
            cleaned_text: Text with wrapper phrases removed
            
        Returns:
            Extracted JSON array string or None if not found
        """
        for pattern in self.json_patterns:
            matches = re.findall(pattern, cleaned_text, re.DOTALL)
            if matches:
                # Return the longest match (most complete)
                longest_match = max(matches, key=len)
                self.parsing_stats["json_extractions"] += 1
                return longest_match
        
        self.parsing_stats["json_extraction_failures"] += 1
        return None
    
    def parse_triple_array(self, json_str: str, source_line: int = 0) -> List[Triple]:
        """
        Parse JSON string into Triple objects with validation.
        
        Args:
            json_str: JSON array string containing triples
            source_line: Source line number for tracking
            
        Returns:
            List of validated Triple objects
        """
        triples = []
        
        try:
            # Try JSON parsing first
            try:
                raw_triples = json.loads(json_str)
            except json.JSONDecodeError:
                # Fallback to ast.literal_eval for Python list format
                import ast
                raw_triples = ast.literal_eval(json_str)
            
            # Process each triple
            for i, raw_triple in enumerate(raw_triples):
                if isinstance(raw_triple, list) and len(raw_triple) == 3:
                    subject, relation, obj = raw_triple
                    
                    # Map relation to standardized vocabulary
                    standardized_relation = self.relation_mapper.map_relation(relation)
                    
                    # Create and validate triple
                    triple = Triple(
                        subject=str(subject), 
                        relation=standardized_relation, 
                        object=str(obj),
                        source_line=source_line
                    )
                    
                    if triple.is_valid():
                        triples.append(triple)
                        self.parsing_stats["valid_triples"] += 1
                    else:
                        self.parsing_stats["invalid_triples"] += 1
                        logger.debug(f"Invalid triple: {raw_triple}")
                else:
                    self.parsing_stats["malformed_triples"] += 1
                    logger.debug(f"Malformed triple: {raw_triple}")
                    
        except Exception as e:
            self.parsing_stats["parsing_errors"] += 1
            logger.error(f"Error parsing triple array: {e}")
            logger.debug(f"Problematic JSON: {json_str[:100]}...")
        
        return triples
    
    def parse_line(self, line: str, line_number: int) -> List[Triple]:
        """
        Parse a single line from KIMI output into triples.
        
        Args:
            line: Single line of KIMI output
            line_number: Line number for tracking
            
        Returns:
            List of Triple objects extracted from the line
        """
        if not line.strip():
            return []
        
        # Clean wrapper text
        cleaned = self.clean_wrapper_text(line)
        
        # Extract JSON array
        json_array = self.extract_json_array(cleaned)
        if not json_array:
            self.parsing_stats["no_json_found"] += 1
            logger.debug(f"No JSON found in line {line_number}: {line[:50]}...")
            return []
        
        # Parse triples from JSON
        triples = self.parse_triple_array(json_array, line_number)
        return triples
    
    def parse_file(self, input_file: str) -> List[Triple]:
        """
        Parse entire KIMI output file into deduplicated triples.
        
        Args:
            input_file: Path to KIMI output file
            
        Returns:
            List of unique, validated Triple objects
        """
        logger.info(f"Starting to parse file: {input_file}")
        
        all_triples = []
        
        try:
            with open(input_file, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            self.parsing_stats["total_lines"] = len(lines)
            
            for line_num, line in enumerate(lines, 1):
                triples = self.parse_line(line, line_num)
                all_triples.extend(triples)
                
                # Progress logging
                if line_num % 10 == 0:
                    logger.info(f"Processed {line_num}/{len(lines)} lines, found {len(all_triples)} triples so far")
        
        except FileNotFoundError:
            logger.error(f"Input file not found: {input_file}")
            return []
        except Exception as e:
            logger.error(f"Error reading file {input_file}: {e}")
            return []
        
        # Deduplicate triples
        unique_triples = list(set(all_triples))
        duplicates_removed = len(all_triples) - len(unique_triples)
        
        self.parsing_stats["total_triples_found"] = len(all_triples)
        self.parsing_stats["unique_triples"] = len(unique_triples)
        self.parsing_stats["duplicates_removed"] = duplicates_removed
        
        logger.info(f"Parsing complete: {len(unique_triples)} unique triples from {len(all_triples)} total")
        logger.info(f"Removed {duplicates_removed} duplicate triples")
        
        return unique_triples
    
    def save_outputs(self, triples: List[Triple], output_dir: str) -> Dict[str, str]:
        """
        Save triples in multiple formats for different use cases.
        
        Args:
            triples: List of Triple objects to save
            output_dir: Output directory path
            
        Returns:
            Dictionary mapping output format to file path
        """
        os.makedirs(output_dir, exist_ok=True)
        output_files = {}
        
        # 1. Clean triples in JSON format
        json_triples = [triple.to_list() for triple in triples]
        json_file = os.path.join(output_dir, 'cleaned_triples.json')
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(json_triples, f, ensure_ascii=False, indent=2)
        output_files['json'] = json_file
        
        # 2. Statement format for Graph Judge
        statements_file = os.path.join(output_dir, 'statements_for_judge.txt')
        with open(statements_file, 'w', encoding='utf-8') as f:
            for triple in triples:
                f.write(triple.to_statement() + '\n')
        output_files['statements'] = statements_file
        
        # 3. Graph Judge instruction format
        instructions_file = os.path.join(output_dir, 'instructions_for_judge.json')
        instructions = []
        for triple in triples:
            instruction = {
                "instruction": f"Is this true: {triple.to_statement()} ?",
                "input": "",
                "output": ""
            }
            instructions.append(instruction)
        
        with open(instructions_file, 'w', encoding='utf-8') as f:
            json.dump(instructions, f, ensure_ascii=False, indent=2)
        output_files['instructions'] = instructions_file
        
        # 4. TSV format for analysis
        tsv_file = os.path.join(output_dir, 'triples.tsv')
        with open(tsv_file, 'w', encoding='utf-8') as f:
            f.write('subject\trelation\tobject\tsource_line\tconfidence\n')
            for triple in triples:
                f.write(f'{triple.subject}\t{triple.relation}\t{triple.object}\t{triple.source_line}\t{triple.confidence}\n')
        output_files['tsv'] = tsv_file
        
        # 5. Statistics report
        stats_file = os.path.join(output_dir, 'parsing_stats.json')
        combined_stats = {
            'parsing_stats': dict(self.parsing_stats),
            'relation_mapping_stats': self.relation_mapper.get_stats(),
            'summary': {
                'total_unique_triples': len(triples),
                'most_common_relations': self._get_relation_frequency(triples),
                'most_common_subjects': self._get_subject_frequency(triples)
            }
        }
        
        with open(stats_file, 'w', encoding='utf-8') as f:
            json.dump(combined_stats, f, ensure_ascii=False, indent=2)
        output_files['stats'] = stats_file
        
        logger.info(f"Saved {len(triples)} triples in {len(output_files)} formats to {output_dir}")
        return output_files
    
    def _get_relation_frequency(self, triples: List[Triple], top_n: int = 10) -> List[Tuple[str, int]]:
        """Get most frequent relations for analysis."""
        relations = [triple.relation for triple in triples]
        return Counter(relations).most_common(top_n)
    
    def _get_subject_frequency(self, triples: List[Triple], top_n: int = 10) -> List[Tuple[str, int]]:
        """Get most frequent subjects for analysis."""
        subjects = [triple.subject for triple in triples]
        return Counter(subjects).most_common(top_n)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive parsing statistics."""
        return {
            'parsing_stats': dict(self.parsing_stats),
            'relation_mapping_stats': self.relation_mapper.get_stats()
        }

def main():
    """
    Main function for command-line usage of the triple parser.
    
    Example usage:
        python parse_kimi_triples.py input.txt output_dir/
    """
    import sys
    
    if len(sys.argv) != 3:
        print("Usage: python parse_kimi_triples.py <input_file> <output_directory>")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_dir = sys.argv[2]
    
    # Initialize parser with default configuration
    parser = KIMITripleParser()
    
    # Parse input file
    logger.info(f"Processing KIMI triple file: {input_file}")
    triples = parser.parse_file(input_file)
    
    if not triples:
        logger.error("No valid triples found in input file")
        sys.exit(1)
    
    # Save outputs in multiple formats
    output_files = parser.save_outputs(triples, output_dir)
    
    # Print summary
    print(f"\n✅ Triple parsing completed successfully!")
    print(f"📊 Processed {len(triples)} unique triples")
    print(f"📁 Output files:")
    for format_name, file_path in output_files.items():
        print(f"   {format_name}: {file_path}")
    
    # Print key statistics
    stats = parser.get_stats()
    print(f"\n📈 Key Statistics:")
    print(f"   Total lines processed: {stats['parsing_stats'].get('total_lines', 0)}")
    print(f"   Valid triples found: {stats['parsing_stats'].get('valid_triples', 0)}")
    print(f"   Duplicates removed: {stats['parsing_stats'].get('duplicates_removed', 0)}")
    print(f"   Parsing errors: {stats['parsing_stats'].get('parsing_errors', 0)}")

if __name__ == "__main__":
    main()
