"""
Post-Extraction Entity Cleaner for ECTD Pipeline

This script implements the post-extraction cleaning functionality for the ECTD phase,
focusing on removing duplicates within entity lines and filtering out generic categories
that don't contribute meaningful information to knowledge graph construction.

Key Features:
1. Remove duplicate entities within the same line (e.g., multiple "神瑛侍者")
2. Filter out generic abstract categories using configurable stop-lists
3. Preserve entity order while removing duplicates
4. Support both list format and comma-separated format
5. Comprehensive logging and error handling
6. Validation of input/output data integrity

Usage:
    from tools.clean_entities import EntityCleaner
    
    cleaner = EntityCleaner()
    cleaned_entities = cleaner.clean_entity_file("path/to/test_entity.txt")
    cleaner.save_cleaned_entities(cleaned_entities, "path/to/output.txt")

Command Line Usage:
    python tools/clean_entities.py --input test_entity.txt --output test_entity_cleaned.txt
"""

import os
import re
import json
import argparse
import logging
from pathlib import Path
from typing import List, Dict, Set, Union, Tuple
from collections import OrderedDict


class EntityCleaner:
    """
    A comprehensive entity cleaning utility for ECTD pipeline.
    
    This class handles the post-extraction cleaning of entity lists by removing
    duplicates and filtering out generic categories that don't contribute to
    meaningful knowledge graph construction.
    """
    
    def __init__(self, config_path: str = None):
        """
        Initialize the EntityCleaner with configurable stop-lists.
        
        Args:
            config_path (str, optional): Path to custom configuration file.
                                       If None, uses default configuration.
        
        The configuration includes stop-lists for different types of generic
        categories that should be filtered out during cleaning.
        """
        self.setup_logging()
        self.stop_lists = self._load_configuration(config_path)
        self.statistics = {
            "total_lines_processed": 0,
            "duplicates_removed": 0,
            "generic_categories_removed": 0,
            "empty_lines_removed": 0,
            "average_entities_per_line_before": 0.0,
            "average_entities_per_line_after": 0.0
        }
        
    def setup_logging(self):
        """
        Configure logging for detailed tracking of cleaning operations.
        
        This method sets up comprehensive logging to track the cleaning process,
        including debug information about which entities are removed and why.
        """
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('logs/entity_cleaning.log', encoding='utf-8'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
        # Create logs directory if it doesn't exist
        os.makedirs('logs', exist_ok=True)
    
    def _load_configuration(self, config_path: str = None) -> Dict[str, Set[str]]:
        """
        Load stop-lists configuration for filtering generic categories.
        
        Args:
            config_path (str, optional): Path to custom configuration file
            
        Returns:
            Dict[str, Set[str]]: Dictionary containing different stop-lists
            
        The configuration includes multiple stop-lists for different types of
        generic categories that should be filtered out:
        - abstract_concepts: Generic abstract terms
        - temporal_terms: Time-related generic terms  
        - administrative_terms: Administrative/bureaucratic terms
        - literary_devices: Literary and narrative devices
        """
        if config_path and os.path.exists(config_path):
            with open(config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
                # Convert lists to sets for faster lookup
                return {k: set(v) for k, v in config.items()}
        
        # Default configuration optimized for classical Chinese literature
        return {
            "abstract_concepts": {
                "功名", "朝代", "年紀", "地輿", "邦國", "家庭", "閨閣", 
                "情性", "禮義", "詩書", "仕宦", "末世", "時運", "抱負",
                "風流", "冤家", "因果", "玄機", "緣分", "命運", "天意"
            },
            "temporal_terms": {
                "當日", "昨日", "今日", "明日", "前日", "後日", "往昔",
                "將來", "此時", "彼時", "時候", "年月", "歲月", "光陰"
            },
            "administrative_terms": {
                "官府", "衙門", "朝廷", "政事", "公務", "差事", "職位",
                "品級", "等第", "階級", "門第", "出身", "資格"
            },
            "literary_devices": {
                "比喻", "象徵", "暗示", "隱喻", "典故", "引用", "借代",
                "擬人", "對比", "反襯", "烘托", "渲染", "描寫", "敘述"
            }
        }
    
    def _parse_entity_line(self, line: str) -> List[str]:
        """
        Parse a single entity line into a list of entities.
        
        Args:
            line (str): Raw entity line from the input file
            
        Returns:
            List[str]: List of individual entities
            
        This method handles different formats:
        1. Python list format: ["entity1", "entity2", "entity3"]
        2. Comma-separated format: entity1, entity2, entity3
        3. Mixed formats with various brackets and quotation marks
        """
        line = line.strip()
        if not line:
            return []
        
        # Handle Python list format: ["entity1", "entity2"]
        if line.startswith('[') and line.endswith(']'):
            try:
                # Use eval safely for list parsing (only for known format)
                entities = eval(line)
                if isinstance(entities, list):
                    return [str(entity).strip() for entity in entities if str(entity).strip()]
            except (SyntaxError, ValueError) as e:
                self.logger.warning(f"Failed to parse list format: {line[:50]}... Error: {e}")
        
        # Handle comma-separated format
        # Remove various brackets and quotation marks
        cleaned_line = re.sub(r'[\[\]"\'""''「」『』]', '', line)
        # Split by both English and Chinese commas
        entities = re.split(r'[,，]', cleaned_line)
        entities = [entity.strip() for entity in entities if entity]  # Remove empty strings
        
        return entities
    
    def _remove_duplicates(self, entities: List[str]) -> List[str]:
        """
        Remove duplicate entities while preserving order.
        
        Args:
            entities (List[str]): List of entities that may contain duplicates
            
        Returns:
            List[str]: List with duplicates removed, order preserved
            
        This method uses OrderedDict to maintain the order of first occurrence
        while removing subsequent duplicates. It handles case-sensitive
        comparison to avoid removing legitimate variations.
        """
        if not entities:
            return []
        
        # Use OrderedDict to preserve order while removing duplicates
        unique_entities = list(OrderedDict.fromkeys(entities))
        
        duplicates_count = len(entities) - len(unique_entities)
        if duplicates_count > 0:
            self.statistics["duplicates_removed"] += duplicates_count
            removed_entities = [e for e in entities if entities.count(e) > 1]
            self.logger.debug(f"Removed {duplicates_count} duplicates: {set(removed_entities)}")
        
        return unique_entities
    
    def _filter_generic_categories(self, entities: List[str]) -> List[str]:
        """
        Filter out generic abstract categories using stop-lists.
        
        Args:
            entities (List[str]): List of entities to filter
            
        Returns:
            List[str]: List with generic categories removed
            
        This method checks each entity against multiple stop-lists to identify
        and remove generic categories that don't contribute meaningful information
        to knowledge graph construction. It maintains detailed statistics about
        what categories were removed.
        """
        if not entities:
            return []
        
        filtered_entities = []
        removed_categories = []
        
        # Combine all stop-lists for efficient checking
        all_stop_words = set()
        for stop_list in self.stop_lists.values():
            all_stop_words.update(stop_list)
        
        for entity in entities:
            entity_clean = entity.strip()
            if entity_clean and entity_clean not in all_stop_words:
                filtered_entities.append(entity_clean)
            elif entity_clean in all_stop_words:
                removed_categories.append(entity_clean)
                self.statistics["generic_categories_removed"] += 1
        
        if removed_categories:
            self.logger.debug(f"Filtered out generic categories: {removed_categories}")
        
        return filtered_entities
    
    def clean_entity_line(self, line: str) -> Tuple[List[str], Dict[str, int]]:
        """
        Clean a single entity line by removing duplicates and generic categories.
        
        Args:
            line (str): Raw entity line to clean
            
        Returns:
            Tuple[List[str], Dict[str, int]]: Cleaned entities and processing statistics
            
        This method orchestrates the complete cleaning process for a single line:
        1. Parse the line into individual entities
        2. Remove duplicates while preserving order
        3. Filter out generic categories
        4. Return cleaned entities with statistics
        """
        line_stats = {
            "original_count": 0,
            "duplicates_removed": 0,
            "generic_removed": 0,
            "final_count": 0
        }
        
        # Parse entities from the line
        entities = self._parse_entity_line(line)
        line_stats["original_count"] = len(entities)
        
        if not entities:
            return [], line_stats
        
        # Remove duplicates
        original_count = len(entities)
        entities = self._remove_duplicates(entities)
        line_stats["duplicates_removed"] = original_count - len(entities)
        
        # Filter generic categories
        pre_filter_count = len(entities)
        entities = self._filter_generic_categories(entities)
        line_stats["generic_removed"] = pre_filter_count - len(entities)
        line_stats["final_count"] = len(entities)
        
        return entities, line_stats
    
    def clean_entity_file(self, input_path: str) -> List[List[str]]:
        """
        Clean an entire entity file by processing each line.
        
        Args:
            input_path (str): Path to the input entity file
            
        Returns:
            List[List[str]]: List of cleaned entity lists, one per line
            
        This method processes the entire entity file line by line, applying
        cleaning operations and maintaining comprehensive statistics about
        the cleaning process. It handles file I/O errors gracefully.
        """
        if not os.path.exists(input_path):
            raise FileNotFoundError(f"Input file not found: {input_path}")
        
        self.logger.info(f"Starting entity cleaning for file: {input_path}")
        
        cleaned_entities = []
        total_original_entities = 0
        total_final_entities = 0
        
        try:
            with open(input_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            for line_num, line in enumerate(lines, 1):
                line = line.strip()
                if not line:
                    self.statistics["empty_lines_removed"] += 1
                    continue
                
                cleaned_line_entities, line_stats = self.clean_entity_line(line)
                cleaned_entities.append(cleaned_line_entities)
                
                total_original_entities += line_stats["original_count"]
                total_final_entities += line_stats["final_count"]
                
                self.logger.debug(f"Line {line_num}: {line_stats['original_count']} → "
                                f"{line_stats['final_count']} entities")
            
            # Update global statistics
            self.statistics["total_lines_processed"] = len(lines)
            self.statistics["average_entities_per_line_before"] = (
                total_original_entities / len(lines) if lines else 0
            )
            self.statistics["average_entities_per_line_after"] = (
                total_final_entities / len(cleaned_entities) if cleaned_entities else 0
            )
            
            self.logger.info(f"Entity cleaning completed. Processed {len(lines)} lines, "
                           f"reduced from {total_original_entities} to {total_final_entities} entities")
            
        except UnicodeDecodeError as e:
            self.logger.error(f"Unicode decode error reading file {input_path}: {e}")
            raise
        except Exception as e:
            self.logger.error(f"Unexpected error processing file {input_path}: {e}")
            raise
        
        return cleaned_entities
    
    def save_cleaned_entities(self, cleaned_entities: List[List[str]], output_path: str, 
                            format_type: str = "list") -> None:
        """
        Save cleaned entities to a file in the specified format.
        
        Args:
            cleaned_entities (List[List[str]]): Cleaned entity lists to save
            output_path (str): Path for the output file
            format_type (str): Output format - "list" for Python list format,
                             "csv" for comma-separated format
        
        This method saves the cleaned entities in the specified format while
        ensuring proper encoding and error handling. It supports multiple
        output formats for compatibility with downstream processing.
        """
        # Create output directory if it doesn't exist
        os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', 
                   exist_ok=True)
        
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                for entities in cleaned_entities:
                    if format_type == "list":
                        # Python list format for compatibility with existing scripts
                        f.write(str(entities) + '\n')
                    elif format_type == "csv":
                        # Comma-separated format for readability
                        f.write(', '.join(entities) + '\n')
                    else:
                        raise ValueError(f"Unsupported format type: {format_type}")
            
            self.logger.info(f"Cleaned entities saved to: {output_path}")
            
        except Exception as e:
            self.logger.error(f"Error saving cleaned entities to {output_path}: {e}")
            raise
    
    def get_cleaning_statistics(self) -> Dict[str, Union[int, float]]:
        """
        Get comprehensive statistics about the cleaning process.
        
        Returns:
            Dict[str, Union[int, float]]: Dictionary containing detailed statistics
            
        This method provides insights into what was removed during cleaning,
        helping to validate the cleaning process and tune parameters if needed.
        """
        return self.statistics.copy()
    
    def print_statistics(self) -> None:
        """
        Print a formatted summary of cleaning statistics.
        
        This method provides a human-readable summary of the cleaning process,
        including counts of removed duplicates, filtered categories, and
        efficiency metrics.
        """
        stats = self.get_cleaning_statistics()
        
        print("\n" + "="*50)
        print("ENTITY CLEANING STATISTICS")
        print("="*50)
        print(f"Total lines processed: {stats['total_lines_processed']}")
        print(f"Empty lines removed: {stats['empty_lines_removed']}")
        print(f"Duplicates removed: {stats['duplicates_removed']}")
        print(f"Generic categories removed: {stats['generic_categories_removed']}")
        print(f"Average entities per line (before): {stats['average_entities_per_line_before']:.2f}")
        print(f"Average entities per line (after): {stats['average_entities_per_line_after']:.2f}")
        print(f"Reduction rate: {((stats['average_entities_per_line_before'] - stats['average_entities_per_line_after']) / stats['average_entities_per_line_before'] * 100):.1f}%" if stats['average_entities_per_line_before'] > 0 else "N/A")
        print("="*50)


def main():
    """
    Command-line interface for the entity cleaner.
    
    This function provides a convenient command-line interface for running
    the entity cleaning process with customizable parameters.
    """
    parser = argparse.ArgumentParser(
        description="Clean entity files by removing duplicates and generic categories",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python clean_entities.py --input test_entity.txt --output test_entity_cleaned.txt
  python clean_entities.py --input data/entities.txt --format csv --config my_config.json
        """
    )
    
    parser.add_argument(
        '--input', '-i',
        required=True,
        help='Input entity file path'
    )
    
    parser.add_argument(
        '--output', '-o',
        required=True,
        help='Output cleaned entity file path'
    )
    
    parser.add_argument(
        '--format', '-f',
        choices=['list', 'csv'],
        default='list',
        help='Output format (default: list)'
    )
    
    parser.add_argument(
        '--config', '-c',
        help='Custom configuration file path for stop-lists'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose logging'
    )
    
    args = parser.parse_args()
    
    # Set logging level based on verbose flag
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    try:
        # Initialize cleaner with optional custom configuration
        cleaner = EntityCleaner(config_path=args.config)
        
        # Clean the entity file
        cleaned_entities = cleaner.clean_entity_file(args.input)
        
        # Save cleaned entities
        cleaner.save_cleaned_entities(cleaned_entities, args.output, args.format)
        
        # Print statistics
        cleaner.print_statistics()
        
        print(f"\n✅ Entity cleaning completed successfully!")
        print(f"📄 Input: {args.input}")
        print(f"📄 Output: {args.output}")
        print(f"📊 Format: {args.format}")
        
    except Exception as e:
        print(f"❌ Error during entity cleaning: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
