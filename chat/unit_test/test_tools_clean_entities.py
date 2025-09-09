"""
Unit Tests for Entity Cleaner Tool

This test suite validates the functionality of the tools/clean_entities.py module,
ensuring that entity cleaning, deduplication, and generic category filtering
work correctly across different scenarios and input formats.

Test Coverage:
- Entity line parsing with different formats
- Duplicate removal while preserving order
- Generic category filtering with configurable stop-lists
- File I/O operations with proper encoding
- Statistics tracking and reporting
- Error handling for malformed input
- Configuration loading and validation

Run with: pytest test_tools_clean_entities.py -v --json-report --json-report-file=test_clean_entities_report.json
"""

import pytest
import os
import json
import tempfile
import shutil
from pathlib import Path
from unittest.mock import patch, mock_open, MagicMock
import sys

# Add the parent directories to the path to import the module under test
# The tools directory is located at GraphJudge/tools, not chat/tools
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Import the tool modules
try:
    from tools.clean_entities import EntityCleaner, AlignmentIssueType
except ImportError:
    # Handle case where tools directory structure may be different
    sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "tools"))
    from clean_entities import EntityCleaner


class TestEntityCleaner:
    """Test cases for the EntityCleaner class."""
    
    def setup_method(self):
        """Set up method called before each test method."""
        # Create temporary directory for testing
        self.temp_dir = tempfile.mkdtemp()
        self.test_logs_dir = os.path.join(self.temp_dir, "logs")
        os.makedirs(self.test_logs_dir, exist_ok=True)
        
        # Sample entity data for testing
        self.sample_entity_lines = [
            '["甄士隱", "甄士隱", "書房"]',  # Contains duplicate
            '["閶門", "十里街", "仁清巷", "古廟", "葫蘆廟"]',  # No duplicates
            '["甄費", "甄士隱", "封氏", "功名", "朝代"]',  # Contains generic categories
            '[]',  # Empty list
            '',  # Empty line
            '甄士隱, 書房, 功名',  # Comma-separated format with generic
        ]
        
        # Expected results after cleaning
        self.expected_cleaned = [
            ["甄士隱", "書房"],  # Duplicate removed
            ["閶門", "十里街", "仁清巷", "古廟", "葫蘆廟"],  # No changes
            ["甄費", "甄士隱", "封氏"],  # Generic categories removed (功名, 朝代)
            [],  # Empty list unchanged
            [],  # Empty line becomes empty list
            ["甄士隱", "書房"],  # Generic removed from comma-separated
        ]
        
        # Patch logging to use temporary directory
        self.patcher = patch('tools.clean_entities.logging.FileHandler')
        self.mock_file_handler = self.patcher.start()
        
    def teardown_method(self):
        """Clean up method called after each test method."""
        # Stop all patches
        self.patcher.stop()
        
        # Remove temporary directory
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_init_default_config(self):
        """Test EntityCleaner initialization with default configuration."""
        cleaner = EntityCleaner()
        
        # Check that default stop lists are loaded
        assert "abstract_concepts" in cleaner.stop_lists
        assert "temporal_terms" in cleaner.stop_lists
        assert "administrative_terms" in cleaner.stop_lists
        assert "literary_devices" in cleaner.stop_lists
        
        # Check that specific stop words are included
        assert "功名" in cleaner.stop_lists["abstract_concepts"]
        assert "朝代" in cleaner.stop_lists["abstract_concepts"]
        
        # Check statistics initialization
        assert cleaner.statistics["total_lines_processed"] == 0
        assert cleaner.statistics["duplicates_removed"] == 0
    
    def test_init_custom_config(self):
        """Test EntityCleaner initialization with custom configuration."""
        # Create custom config file
        custom_config = {
            "test_category": ["test_word1", "test_word2"]
        }
        config_file = os.path.join(self.temp_dir, "custom_config.json")
        with open(config_file, 'w', encoding='utf-8') as f:
            json.dump(custom_config, f)
        
        cleaner = EntityCleaner(config_path=config_file)
        
        # Check that custom config is loaded
        assert "test_category" in cleaner.stop_lists
        assert "test_word1" in cleaner.stop_lists["test_category"]
        assert "test_word2" in cleaner.stop_lists["test_category"]
    
    def test_parse_entity_line_list_format(self):
        """Test parsing entity lines in Python list format."""
        cleaner = EntityCleaner()
        
        # Test valid list format
        line = '["甄士隱", "書房", "封氏"]'
        entities = cleaner._parse_entity_line(line)
        assert entities == ["甄士隱", "書房", "封氏"]
        
        # Test empty list
        line = '[]'
        entities = cleaner._parse_entity_line(line)
        assert entities == []
        
        # Test single entity
        line = '["甄士隱"]'
        entities = cleaner._parse_entity_line(line)
        assert entities == ["甄士隱"]
    
    def test_parse_entity_line_comma_separated(self):
        """Test parsing entity lines in comma-separated format."""
        cleaner = EntityCleaner()
        
        # Test comma-separated format
        line = '甄士隱, 書房, 封氏'
        entities = cleaner._parse_entity_line(line)
        assert entities == ["甄士隱", "書房", "封氏"]
        
        # Test with various quotes and brackets
        line = '"甄士隱", "書房", "封氏"'
        entities = cleaner._parse_entity_line(line)
        assert entities == ["甄士隱", "書房", "封氏"]
        
        # Test with Chinese quotes
        line = '"甄士隱"，"書房"，"封氏"'
        entities = cleaner._parse_entity_line(line)
        assert entities == ["甄士隱", "書房", "封氏"]
    
    def test_parse_entity_line_malformed(self):
        """Test parsing malformed entity lines."""
        cleaner = EntityCleaner()
        
        # Test malformed list (should fall back to comma-separated parsing)
        line = '["甄士隱", "書房"'  # Missing closing bracket
        entities = cleaner._parse_entity_line(line)
        # Should parse as comma-separated after removing brackets
        assert "甄士隱" in entities
        assert "書房" in entities
        
        # Test empty line
        line = ''
        entities = cleaner._parse_entity_line(line)
        assert entities == []
        
        # Test whitespace only
        line = '   '
        entities = cleaner._parse_entity_line(line)
        assert entities == []
    
    def test_remove_duplicates(self):
        """Test duplicate removal while preserving order."""
        cleaner = EntityCleaner()
        
        # Test with duplicates
        entities = ["甄士隱", "書房", "甄士隱", "封氏", "書房"]
        result = cleaner._remove_duplicates(entities)
        assert result == ["甄士隱", "書房", "封氏"]
        
        # Test with no duplicates
        entities = ["甄士隱", "書房", "封氏"]
        result = cleaner._remove_duplicates(entities)
        assert result == ["甄士隱", "書房", "封氏"]
        
        # Test empty list
        entities = []
        result = cleaner._remove_duplicates(entities)
        assert result == []
        
        # Test single entity
        entities = ["甄士隱"]
        result = cleaner._remove_duplicates(entities)
        assert result == ["甄士隱"]
    
    def test_filter_generic_categories(self):
        """Test filtering of generic categories using stop-lists."""
        cleaner = EntityCleaner()
        
        # Test filtering with default stop-lists
        entities = ["甄士隱", "功名", "朝代", "書房", "年紀"]
        result = cleaner._filter_generic_categories(entities)
        # Should remove "功名", "朝代", "年紀" (generic categories)
        assert result == ["甄士隱", "書房"]
        
        # Test with no generic categories
        entities = ["甄士隱", "書房", "封氏"]
        result = cleaner._filter_generic_categories(entities)
        assert result == ["甄士隱", "書房", "封氏"]
        
        # Test empty list
        entities = []
        result = cleaner._filter_generic_categories(entities)
        assert result == []
    
    def test_clean_entity_line(self):
        """Test complete entity line cleaning process."""
        cleaner = EntityCleaner()
        
        # Test line with duplicates and generic categories
        line = '["甄士隱", "甄士隱", "功名", "書房", "朝代"]'
        cleaned_entities, stats = cleaner.clean_entity_line(line)
        
        assert cleaned_entities == ["甄士隱", "書房"]
        assert stats["original_count"] == 5
        assert stats["duplicates_removed"] == 1
        assert stats["generic_removed"] == 2
        assert stats["final_count"] == 2
    
    def test_clean_entity_file(self):
        """Test cleaning an entire entity file."""
        # Create test input file
        input_file = os.path.join(self.temp_dir, "test_entities.txt")
        with open(input_file, 'w', encoding='utf-8') as f:
            for line in self.sample_entity_lines:
                f.write(line + '\n')
        
        cleaner = EntityCleaner()
        cleaned_entities = cleaner.clean_entity_file(input_file)
        
        # Check results (excluding empty lines)
        non_empty_results = [entities for entities in cleaned_entities if entities]
        non_empty_expected = [entities for entities in self.expected_cleaned[:4] if entities]  # First 4 non-empty
        
        # Verify each non-empty result
        for i, (result, expected) in enumerate(zip(non_empty_results, non_empty_expected)):
            assert result == expected, f"Mismatch at index {i}: {result} != {expected}"
        
        # Check statistics
        stats = cleaner.get_cleaning_statistics()
        assert stats["total_lines_processed"] == len(self.sample_entity_lines)
        assert stats["duplicates_removed"] > 0
        assert stats["generic_categories_removed"] > 0
    
    def test_save_cleaned_entities_list_format(self):
        """Test saving cleaned entities in list format."""
        cleaner = EntityCleaner()
        cleaned_entities = [
            ["甄士隱", "書房"],
            ["閶門", "十里街"],
            []
        ]
        
        output_file = os.path.join(self.temp_dir, "output_entities.txt")
        cleaner.save_cleaned_entities(cleaned_entities, output_file, format_type="list")
        
        # Verify output file
        assert os.path.exists(output_file)
        with open(output_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        assert lines[0].strip() == "['甄士隱', '書房']"
        assert lines[1].strip() == "['閶門', '十里街']"
        assert lines[2].strip() == "[]"
    
    def test_save_cleaned_entities_csv_format(self):
        """Test saving cleaned entities in CSV format."""
        cleaner = EntityCleaner()
        cleaned_entities = [
            ["甄士隱", "書房"],
            ["閶門", "十里街"],
            []
        ]
        
        output_file = os.path.join(self.temp_dir, "output_entities.csv")
        cleaner.save_cleaned_entities(cleaned_entities, output_file, format_type="csv")
        
        # Verify output file
        assert os.path.exists(output_file)
        with open(output_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        assert lines[0].strip() == "甄士隱, 書房"
        assert lines[1].strip() == "閶門, 十里街"
        assert lines[2].strip() == ""
    
    def test_get_cleaning_statistics(self):
        """Test statistics collection and retrieval."""
        cleaner = EntityCleaner()
        
        # Process some data to generate statistics
        line = '["甄士隱", "甄士隱", "功名", "書房"]'
        cleaner.clean_entity_line(line)
        
        stats = cleaner.get_cleaning_statistics()
        
        # Check that statistics are tracked
        assert "total_lines_processed" in stats
        assert "duplicates_removed" in stats
        assert "generic_categories_removed" in stats
        assert stats["duplicates_removed"] > 0
        assert stats["generic_categories_removed"] > 0
    
    def test_file_not_found_error(self):
        """Test error handling for non-existent input files."""
        cleaner = EntityCleaner()
        
        non_existent_file = os.path.join(self.temp_dir, "non_existent.txt")
        
        with pytest.raises(FileNotFoundError):
            cleaner.clean_entity_file(non_existent_file)
    
    def test_unicode_handling(self):
        """Test proper Unicode handling for Chinese text."""
        cleaner = EntityCleaner()
        
        # Test with complex Chinese characters
        entities = ["賈寶玉", "林黛玉", "薛寶釵", "史湘雲"]
        result = cleaner._remove_duplicates(entities)
        assert result == entities
        
        # Test with traditional Chinese punctuation
        line = '"賈寶玉"，"林黛玉"，"薛寶釵"'
        parsed = cleaner._parse_entity_line(line)
        assert "賈寶玉" in parsed
        assert "林黛玉" in parsed
        assert "薛寶釵" in parsed
    
    def test_edge_cases(self):
        """Test various edge cases and boundary conditions."""
        cleaner = EntityCleaner()
        
        # Test very long entity names
        long_entity = "非常長的實體名稱" * 10
        entities = [long_entity, "短名", long_entity]
        result = cleaner._remove_duplicates(entities)
        assert len(result) == 2
        assert long_entity in result
        assert "短名" in result
        
        # Test entities with special characters
        special_entities = ["實體@1", "實體#2", "實體$3"]
        result = cleaner._filter_generic_categories(special_entities)
        assert len(result) == 3  # Should not filter these
        
        # Test mixed language entities
        mixed_entities = ["甄士隱", "Smith", "書房", "House"]
        result = cleaner._remove_duplicates(mixed_entities)
        assert len(result) == 4


class TestEntityCleanerIntegration:
    """Integration tests for the EntityCleaner tool."""
    
    def setup_method(self):
        """Set up method called before each test method."""
        self.temp_dir = tempfile.mkdtemp()
        
        # Patch logging to use temporary directory
        self.patcher = patch('tools.clean_entities.logging.FileHandler')
        self.mock_file_handler = self.patcher.start()
    
    def teardown_method(self):
        """Clean up method called after each test method."""
        self.patcher.stop()
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_end_to_end_cleaning(self):
        """Test complete end-to-end entity cleaning workflow."""
        # Create realistic test data
        test_data = [
            '["甄士隱", "甄士隱", "書房", "功名"]',  # Duplicates + generic
            '["賈雨村", "胡州", "朝代", "詩書仕宦之族"]',  # Generic category
            '閶門, 十里街, 仁清巷, 古廟, 葫蘆廟',  # Comma-separated, no issues
            '["林黛玉", "賈寶玉", "太虛幻境"]',  # No issues
            '',  # Empty line
            '[]'  # Empty list
        ]
        
        # Write test file
        input_file = os.path.join(self.temp_dir, "test_input.txt")
        with open(input_file, 'w', encoding='utf-8') as f:
            for line in test_data:
                f.write(line + '\n')
        
        # Process with cleaner
        cleaner = EntityCleaner()
        cleaned_entities = cleaner.clean_entity_file(input_file)
        
        # Save results
        output_file = os.path.join(self.temp_dir, "test_output.txt")
        cleaner.save_cleaned_entities(cleaned_entities, output_file)
        
        # Verify results
        assert os.path.exists(output_file)
        
        with open(output_file, 'r', encoding='utf-8') as f:
            output_lines = [line.strip() for line in f.readlines()]
        
        # Check that duplicates and generic categories were removed
        # First line should be cleaned of duplicates and generics
        assert "甄士隱" in output_lines[0]
        assert "書房" in output_lines[0]
        assert output_lines[0].count("甄士隱") == 1  # No duplicates
        
        # Verify statistics
        stats = cleaner.get_cleaning_statistics()
        assert stats["duplicates_removed"] > 0
        assert stats["generic_categories_removed"] > 0
        assert stats["total_lines_processed"] == len(test_data)
    
    def test_performance_large_dataset(self):
        """Test performance with larger dataset."""
        # Create large test dataset
        large_dataset = []
        for i in range(100):
            # Create lines with varying complexity
            if i % 3 == 0:
                # Lines with duplicates
                large_dataset.append('["甄士隱", "甄士隱", "書房", "功名"]')
            elif i % 3 == 1:
                # Lines with generic categories
                large_dataset.append('["賈雨村", "朝代", "胡州"]')
            else:
                # Clean lines
                large_dataset.append('["林黛玉", "賈寶玉", "薛寶釵"]')
        
        # Write large test file
        input_file = os.path.join(self.temp_dir, "large_test.txt")
        with open(input_file, 'w', encoding='utf-8') as f:
            for line in large_dataset:
                f.write(line + '\n')
        
        # Process and time the operation
        import time
        start_time = time.time()
        
        cleaner = EntityCleaner()
        cleaned_entities = cleaner.clean_entity_file(input_file)
        
        end_time = time.time()
        processing_time = end_time - start_time
        
        # Verify results
        assert len(cleaned_entities) == 100
        assert processing_time < 10.0  # Should complete within 10 seconds
        
        # Check statistics
        stats = cleaner.get_cleaning_statistics()
        assert stats["total_lines_processed"] == 100
        assert stats["duplicates_removed"] > 0
        assert stats["generic_categories_removed"] > 0


if __name__ == "__main__":
    # Run tests when script is executed directly
    pytest.main([__file__, "-v", "--json-report", "--json-report-file=test_clean_entities_report.json"])
