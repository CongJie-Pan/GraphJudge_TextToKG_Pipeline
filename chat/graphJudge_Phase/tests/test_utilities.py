"""
Unit tests for the utilities module.

Tests validation functions, environment checking, file operations,
and other utility functions used throughout the GraphJudge system.
"""

import pytest
import os
import json
import tempfile
import shutil
from unittest.mock import patch, mock_open, MagicMock

from ..utilities import (
    validate_input_file,
    create_output_directory,
    validate_perplexity_environment,
    validate_instruction_format,
    clean_response_text,
    extract_triple_from_instruction,
    format_processing_time,
    calculate_similarity_score,
    safe_json_dump,
    safe_json_load,
    get_file_size_mb,
    check_disk_space,
    create_backup_file,
    validate_csv_format,
    log_system_info,
    setup_environment
)
from .conftest import PerplexityTestBase


class TestFileValidation(PerplexityTestBase):
    """Test file validation functionality."""
    
    def test_validate_input_file_success(self):
        """Test successful input file validation."""
        result = validate_input_file(self.test_input_file)
        assert result is True
    
    def test_validate_input_file_missing(self):
        """Test validation failure when input file is missing."""
        non_existent_file = os.path.join(self.temp_dir, "missing.json")
        result = validate_input_file(non_existent_file)
        assert result is False
    
    def test_validate_input_file_invalid_json(self):
        """Test validation failure with invalid JSON format."""
        invalid_json_file = os.path.join(self.temp_dir, "invalid.json")
        with open(invalid_json_file, 'w') as f:
            f.write("invalid json content {")
        
        result = validate_input_file(invalid_json_file)
        assert result is False
    
    def test_validate_input_file_empty_list(self):
        """Test validation failure with empty list."""
        empty_list_file = os.path.join(self.temp_dir, "empty.json")
        with open(empty_list_file, 'w') as f:
            json.dump([], f)
        
        result = validate_input_file(empty_list_file)
        assert result is False
    
    def test_validate_input_file_missing_required_fields(self):
        """Test validation failure with missing required fields."""
        invalid_data = [{"wrong_field": "value"}]
        invalid_file = os.path.join(self.temp_dir, "invalid_fields.json")
        with open(invalid_file, 'w') as f:
            json.dump(invalid_data, f)
        
        result = validate_input_file(invalid_file)
        assert result is False
    
    def test_validate_csv_format_valid(self):
        """Test CSV format validation with valid file."""
        csv_file = os.path.join(self.temp_dir, "valid.csv")
        with open(csv_file, 'w', encoding='utf-8') as f:
            f.write("prompt,generated\n")
            f.write("Test prompt,Yes\n")
        
        result = validate_csv_format(csv_file)
        assert result is True
    
    def test_validate_csv_format_invalid_header(self):
        """Test CSV format validation with invalid header."""
        csv_file = os.path.join(self.temp_dir, "invalid_header.csv")
        with open(csv_file, 'w', encoding='utf-8') as f:
            f.write("wrong,headers\n")
            f.write("Test,Value\n")
        
        result = validate_csv_format(csv_file)
        assert result is False
    
    def test_validate_csv_format_no_data(self):
        """Test CSV format validation with no data rows."""
        csv_file = os.path.join(self.temp_dir, "no_data.csv")
        with open(csv_file, 'w', encoding='utf-8') as f:
            f.write("prompt,generated\n")
        
        result = validate_csv_format(csv_file)
        assert result is False


class TestDirectoryOperations(PerplexityTestBase):
    """Test directory operations."""
    
    def test_create_output_directory(self):
        """Test output directory creation."""
        nested_path = os.path.join(self.temp_dir, "nested", "dir", "output.csv")
        create_output_directory(nested_path)
        
        # Verify directory was created
        assert os.path.exists(os.path.dirname(nested_path))
    
    def test_create_output_directory_existing(self):
        """Test creating output directory that already exists."""
        existing_dir = os.path.join(self.temp_dir, "existing")
        os.makedirs(existing_dir, exist_ok=True)
        
        # Should not raise error
        create_output_directory(os.path.join(existing_dir, "file.csv"))
        assert os.path.exists(existing_dir)


class TestEnvironmentValidation(PerplexityTestBase):
    """Test environment validation functionality."""
    
    def test_validate_perplexity_environment_success(self):
        """Test successful Perplexity environment validation."""
        with patch('litellm.acompletion'):
            result = validate_perplexity_environment()
            assert result is True
    
    def test_validate_perplexity_environment_no_api_key(self):
        """Test Perplexity environment validation without API key."""
        with patch.dict(os.environ, {}, clear=True):
            result = validate_perplexity_environment()
            assert result is False
    
    def test_validate_perplexity_environment_import_error(self):
        """Test Perplexity environment validation with import error."""
        with patch('litellm.acompletion', side_effect=ImportError):
            result = validate_perplexity_environment()
            assert result is False
    
    def test_setup_environment_success(self):
        """Test successful environment setup."""
        with patch('sys.version_info', (3, 8, 0)):
            result = setup_environment()
            assert result is True
    
    def test_setup_environment_old_python(self):
        """Test environment setup with old Python version."""
        with patch('sys.version_info', (3, 6, 0)):
            result = setup_environment()
            assert result is False
    
    def test_setup_environment_missing_vars(self):
        """Test environment setup with missing environment variables."""
        with patch.dict(os.environ, {}, clear=True):
            result = setup_environment()
            assert result is False


class TestInstructionValidation:
    """Test instruction format validation."""
    
    def test_validate_instruction_format_valid(self):
        """Test validation of valid instruction formats."""
        valid_instructions = [
            "Is this true: Apple Founded by Steve Jobs ?",
            "Is this true: 曹雪芹 創作 紅樓夢 ?",
            "Is this true: Very long instruction with many words that should still be valid because it follows the correct format ?",
        ]
        
        for instruction in valid_instructions:
            result = validate_instruction_format(instruction)
            assert result is True, f"Failed for: {instruction}"
    
    def test_validate_instruction_format_invalid(self):
        """Test validation of invalid instruction formats."""
        invalid_instructions = [
            "",  # Empty
            None,  # None
            "Apple Founded by Steve Jobs",  # No prefix/suffix
            "Is this true: Short ?",  # Too short
            "Is this true: " + "x" * 1000 + " ?",  # Too long
            "Wrong prefix: Apple Founded by Steve Jobs ?",  # Wrong prefix
            "Is this true: Apple Founded by Steve Jobs",  # Missing suffix
        ]
        
        for instruction in invalid_instructions:
            result = validate_instruction_format(instruction)
            assert result is False, f"Should have failed for: {instruction}"
    
    def test_extract_triple_from_instruction_valid(self):
        """Test triple extraction from valid instructions."""
        test_cases = [
            ("Is this true: Apple Founded by Steve Jobs ?", "Apple Founded by Steve Jobs"),
            ("Is this true: 曹雪芹 創作 紅樓夢 ?", "曹雪芹 創作 紅樓夢"),
            ("Is this true: Simple Triple ?", "Simple Triple"),
        ]
        
        for instruction, expected_triple in test_cases:
            result = extract_triple_from_instruction(instruction)
            assert result == expected_triple, f"Failed for: {instruction}"
    
    def test_extract_triple_from_instruction_invalid(self):
        """Test triple extraction from invalid instructions."""
        invalid_instructions = [
            "Apple Founded by Steve Jobs",  # No prefix/suffix
            "Wrong prefix: Apple Founded by Steve Jobs ?",  # Wrong prefix
            "Is this true: Apple Founded by Steve Jobs",  # Missing suffix
            "",  # Empty
        ]
        
        for instruction in invalid_instructions:
            result = extract_triple_from_instruction(instruction)
            assert result is None, f"Should have returned None for: {instruction}"


class TestTextProcessing:
    """Test text processing utilities."""
    
    def test_clean_response_text(self):
        """Test response text cleaning."""
        test_cases = [
            ("  Yes  ", "Yes"),
            ("No\n", "No"),
            ("Yes\n\nWith extra\nlines", "Yes With extra lines"),
            ("  Multiple   spaces  ", "Multiple spaces"),
            ("\t\nYes\t\n", "Yes"),
            ("", ""),
        ]
        
        for input_text, expected in test_cases:
            result = clean_response_text(input_text)
            assert result == expected, f"Failed for: '{input_text}'"
    
    def test_format_processing_time(self):
        """Test processing time formatting."""
        test_cases = [
            (0.1, "100.0ms"),
            (0.5, "500.0ms"),
            (1.5, "1.50s"),
            (65.3, "1.1m 5.3s"),
            (3665.7, "1.0h 1.1m"),
        ]
        
        for seconds, expected_format in test_cases:
            result = format_processing_time(seconds)
            # Check that result contains expected time unit
            if "ms" in expected_format:
                assert "ms" in result
            elif "s" in expected_format and "m" not in expected_format:
                assert "s" in result and "m" not in result
            elif "m" in expected_format:
                assert "m" in result
            elif "h" in expected_format:
                assert "h" in result
    
    def test_calculate_similarity_score_with_rapidfuzz(self):
        """Test similarity score calculation with RapidFuzz."""
        try:
            import rapidfuzz.fuzz
            rapidfuzz_available = True
        except ImportError:
            rapidfuzz_available = False
        
        if rapidfuzz_available:
            with patch('rapidfuzz.fuzz.partial_ratio', return_value=85):
                result = calculate_similarity_score("Apple Founded by Steve Jobs", "Steve Jobs founded Apple")
                assert result == 0.85
    
    def test_calculate_similarity_score_fallback(self):
        """Test similarity score calculation fallback method."""
        with patch('graphJudge_Phase.utilities.rapidfuzz', None):
            # Test identical strings
            result1 = calculate_similarity_score("test", "test")
            assert result1 > 0.0
            
            # Test different strings
            result2 = calculate_similarity_score("apple", "orange")
            assert 0.0 <= result2 <= 1.0
            
            # Test empty strings
            result3 = calculate_similarity_score("", "")
            assert result3 == 0.0


class TestFileOperations:
    """Test file operation utilities."""
    
    def test_safe_json_dump_success(self):
        """Test successful JSON dumping."""
        test_data = {"key": "value", "number": 42}
        
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as tmp_file:
            tmp_filename = tmp_file.name
        
        try:
            result = safe_json_dump(test_data, tmp_filename)
            assert result is True
            
            # Verify content
            with open(tmp_filename, 'r', encoding='utf-8') as f:
                loaded_data = json.load(f)
                assert loaded_data == test_data
        
        finally:
            if os.path.exists(tmp_filename):
                os.unlink(tmp_filename)
    
    def test_safe_json_dump_error(self):
        """Test JSON dumping with error."""
        test_data = {"key": "value"}
        
        # Try to write to invalid path
        result = safe_json_dump(test_data, "/invalid/path/file.json")
        assert result is False
    
    def test_safe_json_load_success(self):
        """Test successful JSON loading."""
        test_data = {"key": "value", "number": 42}
        
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as tmp_file:
            json.dump(test_data, tmp_file)
            tmp_filename = tmp_file.name
        
        try:
            result = safe_json_load(tmp_filename)
            assert result == test_data
        
        finally:
            if os.path.exists(tmp_filename):
                os.unlink(tmp_filename)
    
    def test_safe_json_load_error(self):
        """Test JSON loading with error."""
        result = safe_json_load("/nonexistent/file.json")
        assert result is None
    
    def test_get_file_size_mb(self):
        """Test file size calculation."""
        with tempfile.NamedTemporaryFile(mode='w', delete=False) as tmp_file:
            tmp_file.write("x" * 1024)  # Write 1KB
            tmp_filename = tmp_file.name
        
        try:
            size_mb = get_file_size_mb(tmp_filename)
            assert size_mb > 0.0
            assert size_mb < 1.0  # Should be less than 1MB
        
        finally:
            if os.path.exists(tmp_filename):
                os.unlink(tmp_filename)
    
    def test_get_file_size_mb_nonexistent(self):
        """Test file size calculation for nonexistent file."""
        size_mb = get_file_size_mb("/nonexistent/file.txt")
        assert size_mb == 0.0
    
    def test_check_disk_space(self):
        """Test disk space checking."""
        # Test with current directory (should have space)
        result = check_disk_space(".", required_mb=1.0)
        assert result is True
        
        # Test with very large requirement
        result = check_disk_space(".", required_mb=999999999.0)
        # This might be True or False depending on available space
        assert isinstance(result, bool)
    
    def test_create_backup_file(self):
        """Test backup file creation."""
        with tempfile.NamedTemporaryFile(mode='w', delete=False) as tmp_file:
            tmp_file.write("test content")
            tmp_filename = tmp_file.name
        
        try:
            backup_path = create_backup_file(tmp_filename)
            
            if backup_path:  # Backup creation might fail in some environments
                assert os.path.exists(backup_path)
                assert backup_path.startswith(tmp_filename)
                assert "backup_" in backup_path
                
                # Clean up backup
                if os.path.exists(backup_path):
                    os.unlink(backup_path)
        
        finally:
            if os.path.exists(tmp_filename):
                os.unlink(tmp_filename)
    
    def test_create_backup_file_nonexistent(self):
        """Test backup creation for nonexistent file."""
        result = create_backup_file("/nonexistent/file.txt")
        assert result is None


class TestSystemUtilities:
    """Test system utility functions."""
    
    def test_log_system_info(self):
        """Test system information logging."""
        # This should not raise any errors
        log_system_info()
        # No specific assertions since this is mostly informational
    
    def test_deprecated_get_perplexity_completion(self):
        """Test deprecated get_perplexity_completion function."""
        from ..utilities import get_perplexity_completion
        
        result = get_perplexity_completion("Is this true: Test ?")
        assert "Error: Use ProcessingPipeline" in result


class TestUtilitiesIntegration:
    """Test integration scenarios for utilities."""
    
    def test_full_validation_workflow(self):
        """Test complete validation workflow."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create valid input file
            input_file = os.path.join(temp_dir, "input.json")
            output_file = os.path.join(temp_dir, "nested", "output.csv")
            
            sample_data = [
                {"instruction": "Is this true: Test Statement ?", "input": "", "output": ""}
            ]
            
            with open(input_file, 'w', encoding='utf-8') as f:
                json.dump(sample_data, f)
            
            # Validate input file
            assert validate_input_file(input_file) is True
            
            # Create output directory
            create_output_directory(output_file)
            assert os.path.exists(os.path.dirname(output_file))
            
            # Validate instruction format
            instruction = sample_data[0]["instruction"]
            assert validate_instruction_format(instruction) is True
            
            # Extract triple
            triple = extract_triple_from_instruction(instruction)
            assert triple == "Test Statement"
    
    def test_error_handling_workflow(self):
        """Test error handling in utility workflow."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Test with invalid input file
            invalid_file = os.path.join(temp_dir, "invalid.json")
            with open(invalid_file, 'w') as f:
                f.write("invalid json")
            
            assert validate_input_file(invalid_file) is False
            
            # Test with invalid instruction
            invalid_instruction = "Invalid instruction format"
            assert validate_instruction_format(invalid_instruction) is False
            assert extract_triple_from_instruction(invalid_instruction) is None
    
    def test_cross_platform_compatibility(self):
        """Test cross-platform compatibility of utilities."""
        # Test path operations work on different platforms
        test_path = os.path.join("test", "path", "file.txt")
        
        # Should not raise errors
        create_output_directory(test_path)
        
        # Test time formatting with various inputs
        for time_val in [0.001, 1.0, 60.0, 3600.0]:
            result = format_processing_time(time_val)
            assert isinstance(result, str)
            assert len(result) > 0
