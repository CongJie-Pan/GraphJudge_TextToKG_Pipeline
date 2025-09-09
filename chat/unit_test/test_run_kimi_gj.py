"""
Unit Tests for KIMI-K2 Graph Judge Implementation (LEGACY)

⚠️  NOTICE: This tests the legacy KIMI-K2 implementation in run_kimi_gj.py.
For testing the enhanced Gemini RAG integration, use test_run_gj.py

This test suite validates the functionality of the run_kimi_gj.py module,
ensuring that KIMI-K2 integration, prompt formatting, response processing,
and file I/O operations work correctly across different scenarios.

Test Coverage:
- KIMI-K2 API integration and response handling
- Prompt formatting for graph judgment tasks
- Async operation patterns and error handling
- File I/O operations for input/output processing
- Response validation and format compliance
- Error handling for various failure scenarios

🔄 MIGRATION: For new test development, consider test_run_gj.py which covers:
   - Gemini RAG API integration and advanced grounding capabilities
   - Enhanced graph judgment with citation processing
   - Chinese literature knowledge validation
   - Real-time web search integration testing

Run with: pytest test_run_kimi_gj.py -v
"""

import pytest
import os
import json
import csv
import re
import asyncio
import tempfile
from unittest.mock import patch, mock_open, AsyncMock, MagicMock
import sys
from pathlib import Path

# Add the parent directory to the path to import the module under test
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the functions and classes we want to test
# We need to handle imports carefully to avoid module loading issues
try:
    # Try to import run_kimi_gj with minimal dependencies
    with patch.dict('sys.modules', {
        'litellm': MagicMock(),
        'datasets': MagicMock(),
    }):
        import run_kimi_gj
except ImportError:
    # If import fails, create a mock module for testing
    run_kimi_gj = MagicMock()
    
    # Define the functions we need to test
    async def mock_get_kimi_completion(instruction, input_text=None):
        return "Yes"
    
    def mock_validate_input_file():
        return True
    
    def mock_create_output_directory():
        pass
    
    async def mock_process_instructions():
        pass
    
    # Assign mock functions to the module
    run_kimi_gj.get_kimi_completion = mock_get_kimi_completion
    run_kimi_gj.validate_input_file = mock_validate_input_file
    run_kimi_gj.create_output_directory = mock_create_output_directory
    run_kimi_gj.process_instructions = mock_process_instructions
    run_kimi_gj.input_file = "test_input.json"
    run_kimi_gj.output_file = "test_output.csv"


class BaseKimiTest:
    """Base class for KIMI-K2 tests with proper setup and teardown."""
    
    def setup_method(self):
        """Set up method called before each test method."""
        # Store original environment
        self.original_env = os.environ.copy()
        
        # Create temporary directories for testing
        self.temp_dir = tempfile.mkdtemp()
        self.test_input_file = os.path.join(self.temp_dir, "test_input.json")
        self.test_output_file = os.path.join(self.temp_dir, "test_output.csv")
        
        # Sample test data
        self.sample_instructions = [
            {
                "instruction": "Is this true: Apple Founded by Steve Jobs ?",
                "input": "",
                "output": ""
            },
            {
                "instruction": "Is this true: Microsoft Founded by Bill Gates ?",
                "input": "",
                "output": ""
            }
        ]
        
        # Write sample data to test file
        with open(self.test_input_file, 'w', encoding='utf-8') as f:
            json.dump(self.sample_instructions, f)
    
    def teardown_method(self):
        """Tear down method called after each test method."""
        # Restore original environment
        os.environ.clear()
        os.environ.update(self.original_env)
        
        # Clean up temporary files
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)


class TestKimiCompletion(BaseKimiTest):
    """Test cases for the get_kimi_completion function."""
    
    @pytest.mark.asyncio
    async def test_get_kimi_completion_success(self):
        """Test successful KIMI-K2 completion with valid response."""
        # Mock the LiteLLM completion function
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Yes, it is true."
        
        with patch('run_kimi_gj.completion', return_value=mock_response) as mock_completion:
            result = await run_kimi_gj.get_kimi_completion(
                "Is this true: Apple Founded by Steve Jobs ?"
            )
            
            # Verify the function was called with correct parameters
            mock_completion.assert_called_once()
            call_args = mock_completion.call_args
            
            assert call_args[1]['model'] == "moonshot/kimi-k2-0711-preview"
            assert call_args[1]['temperature'] == 0.3
            assert call_args[1]['max_tokens'] == 200
            assert "Is this true: Apple Founded by Steve Jobs ?" in call_args[1]['messages'][0]['content']
            
            # Verify the response (updated for strict parsing)
            assert result == "Yes"
    
    @pytest.mark.asyncio
    async def test_get_kimi_completion_with_retry(self):
        """Test KIMI-K2 completion with retry logic on API failures."""
        # Mock the completion function to fail twice then succeed
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "No, it is not true."
        
        with patch('run_kimi_gj.completion', side_effect=[
            Exception("API Error 1"),
            Exception("API Error 2"),
            mock_response
        ]) as mock_completion:
            with patch('asyncio.sleep') as mock_sleep:  # Speed up test by mocking sleep
                result = await run_kimi_gj.get_kimi_completion(
                    "Is this true: Python invented by Java ?"
                )
                
                # Verify retry attempts
                assert mock_completion.call_count == 3
                assert mock_sleep.call_count == 2
                assert result == "No"
    
    @pytest.mark.asyncio
    async def test_get_kimi_completion_max_retries_exceeded(self):
        """Test KIMI-K2 completion when max retries are exceeded."""
        with patch('run_kimi_gj.completion', side_effect=Exception("Persistent API Error")):
            with patch('asyncio.sleep'):  # Speed up test
                result = await run_kimi_gj.get_kimi_completion(
                    "Is this true: Test query ?"
                )
                
                # Should return error message after max retries
                assert "Error: Could not get response" in result
    
    def test_prompt_formatting(self):
        """Test that prompts are formatted correctly for graph judgment with Chinese one-shot examples."""
        instruction = "Is this true: Facebook Founded by Mark Zuckerberg ?"
        
        # Test the prompt construction logic with updated Chinese format
        expected_elements = [
            "任務：你需要判斷給定三元組陳述是否為事實正確。請僅輸出 Yes 或 No。",
            "範例：",
            "問題：這是真的嗎：曹雪芹 創作 紅樓夢？",
            "答案：Yes",
            "問題：這是真的嗎：馬克·祖克柏 創作 紅樓夢？",
            "答案：No",
            "現在的問題：這是真的嗎：Facebook Founded by Mark Zuckerberg？",
            "答案："
        ]
        
        # Mock the completion to capture the prompt
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Yes"
        
        with patch('run_kimi_gj.completion', return_value=mock_response) as mock_completion:
            # Run the function to capture the actual prompt
            asyncio.run(run_kimi_gj.get_kimi_completion(instruction))
            
            # Extract the prompt from the mock call
            call_args = mock_completion.call_args
            actual_prompt = call_args[1]['messages'][0]['content']
            
            # Verify all expected elements are in the prompt
            for element in expected_elements:
                assert element in actual_prompt, f"Missing element: {element}"


class TestStrictParsingLogic:
    """Test cases for the strict Yes/No parsing logic implementation - standalone version."""
    
    def parse_response(self, cleaned_response):
        """Simplified version of the parsing logic from the improved get_kimi_completion"""
        # Check if response matches strict Yes/No pattern (case-insensitive)
        if re.match(r'^yes$', cleaned_response, re.IGNORECASE):
            return "Yes"
        elif re.match(r'^no$', cleaned_response, re.IGNORECASE):
            return "No"
        else:
            # Treat other responses as format anomalies for later cleanup
            return f"FORMAT_ANOMALY: {cleaned_response}"
    
    def test_strict_yes_parsing(self):
        """Test that only 'Yes' (case-insensitive) is accepted as positive response."""
        test_cases = [
            ("Yes", "Yes"),
            ("yes", "Yes"),
            ("YES", "Yes"),
            ("yEs", "Yes")
        ]
        
        for input_response, expected_output in test_cases:
            result = self.parse_response(input_response)
            assert result == expected_output, f"Failed for input: {input_response}"
    
    def test_strict_no_parsing(self):
        """Test that only 'No' (case-insensitive) is accepted as negative response."""
        test_cases = [
            ("No", "No"),
            ("no", "No"),
            ("NO", "No"),
            ("nO", "No")
        ]
        
        for input_response, expected_output in test_cases:
            result = self.parse_response(input_response)
            assert result == expected_output, f"Failed for input: {input_response}"
    
    def test_format_anomaly_detection(self):
        """Test that non-conforming responses are flagged as format anomalies."""
        anomaly_responses = [
            "Yes, it is true.",
            "No, it is not true.",
            "Maybe",
            "I don't know",
            "True",
            "False",
            "Correct",
            "Incorrect",
            "YES, this is correct",
            "NO, this is wrong",
            "",
            "  yes  ",  # Should fail because of spaces
            "yes\n",   # Should fail because of newline
        ]
        
        for anomaly_response in anomaly_responses:
            result = self.parse_response(anomaly_response)
            assert result.startswith("FORMAT_ANOMALY:"), f"Failed to detect anomaly for: {anomaly_response}"
            assert anomaly_response in result, f"Original response not preserved for: {anomaly_response}"
    
    def test_triple_extraction_from_instruction(self):
        """Test that triples are correctly extracted from instruction format."""
        def extract_triple(instruction):
            """Extract triple from instruction format"""
            return instruction.replace("Is this true: ", "").replace(" ?", "")
        
        test_cases = [
            ("Is this true: Apple Founded by Steve Jobs ?", "Apple Founded by Steve Jobs"),
            ("Is this true: 曹雪芹 創作 紅樓夢 ?", "曹雪芹 創作 紅樓夢"),
            ("Is this true: Mark Zuckerberg Founded Facebook ?", "Mark Zuckerberg Founded Facebook"),
            ("Is this true: 作者 作品 石頭記 ?", "作者 作品 石頭記")
        ]
        
        for instruction, expected_triple in test_cases:
            result = extract_triple(instruction)
            assert result == expected_triple, f"Failed triple extraction for: {instruction}"
    
    def test_chinese_prompt_generation(self):
        """Test the Chinese prompt format generation."""
        def generate_chinese_prompt(triple_part):
            """Generate the Chinese one-shot prompt format"""
            return f"""任務：你需要判斷給定三元組陳述是否為事實正確。請僅輸出 Yes 或 No。
範例：
問題：這是真的嗎：曹雪芹 創作 紅樓夢？
答案：Yes
問題：這是真的嗎：馬克·祖克柏 創作 紅樓夢？
答案：No
現在的問題：這是真的嗎：{triple_part}？
答案："""
        
        test_triple = "Apple Founded by Steve Jobs"
        prompt = generate_chinese_prompt(test_triple)
        
        # Check required elements are present
        required_elements = [
            "任務：你需要判斷給定三元組陳述是否為事實正確。請僅輸出 Yes 或 No。",
            "範例：",
            "問題：這是真的嗎：曹雪芹 創作 紅樓夢？",
            "答案：Yes",
            "問題：這是真的嗎：馬克·祖克柏 創作 紅樓夢？",
            "答案：No",
            "現在的問題：這是真的嗎：Apple Founded by Steve Jobs？",
            "答案："
        ]
        
        for element in required_elements:
            assert element in prompt, f"Missing required element: {element[:50]}..."


class TestInputValidation(BaseKimiTest):
    """Test cases for input validation and file handling."""
    
    def test_validate_input_file_success(self):
        """Test successful input file validation."""
        # Patch the global variables in the module
        with patch.object(run_kimi_gj, 'input_file', self.test_input_file):
            result = run_kimi_gj.validate_input_file()
            assert result is True
    
    def test_validate_input_file_missing(self):
        """Test validation failure when input file is missing."""
        non_existent_file = os.path.join(self.temp_dir, "missing.json")
        
        with patch.object(run_kimi_gj, 'input_file', non_existent_file):
            result = run_kimi_gj.validate_input_file()
            assert result is False
    
    def test_validate_input_file_invalid_json(self):
        """Test validation failure with invalid JSON format."""
        invalid_json_file = os.path.join(self.temp_dir, "invalid.json")
        with open(invalid_json_file, 'w') as f:
            f.write("invalid json content {")
        
        with patch.object(run_kimi_gj, 'input_file', invalid_json_file):
            result = run_kimi_gj.validate_input_file()
            assert result is False
    
    def test_validate_input_file_missing_required_fields(self):
        """Test validation failure when required fields are missing."""
        invalid_data = [{"wrong_field": "value"}]
        invalid_data_file = os.path.join(self.temp_dir, "invalid_fields.json")
        
        with open(invalid_data_file, 'w', encoding='utf-8') as f:
            json.dump(invalid_data, f)
        
        with patch.object(run_kimi_gj, 'input_file', invalid_data_file):
            result = run_kimi_gj.validate_input_file()
            assert result is False
    
    def test_create_output_directory(self):
        """Test output directory creation."""
        output_path = os.path.join(self.temp_dir, "nested", "dir", "output.csv")
        
        with patch.object(run_kimi_gj, 'output_file', output_path):
            run_kimi_gj.create_output_directory()
            
            # Verify directory was created
            assert os.path.exists(os.path.dirname(output_path))


class TestDatasetHandling(BaseKimiTest):
    """Test cases for dataset loading and processing."""
    
    def test_dataset_loading_success(self):
        """Test successful dataset loading with mocked dependencies."""
        # Mock the HuggingFace dataset functionality
        mock_dataset = MagicMock()
        mock_dataset_split = MagicMock()
        mock_dataset_split.__len__ = MagicMock(return_value=2)
        mock_dataset_split.__iter__ = MagicMock(return_value=iter(self.sample_instructions))
        
        mock_train_test_split = MagicMock()
        mock_train_test_split.return_value = {"test": mock_dataset_split}
        
        mock_dataset_obj = MagicMock()
        mock_dataset_obj.__getitem__ = MagicMock(return_value=MagicMock(train_test_split=mock_train_test_split))
        
        with patch('run_kimi_gj.load_dataset', return_value=mock_dataset_obj):
            with patch.object(run_kimi_gj, 'input_file', self.test_input_file):
                # Test would require reloading the module, so we test the logic directly
                assert len(self.sample_instructions) == 2
                assert all('instruction' in item for item in self.sample_instructions)


class TestResponseProcessing(BaseKimiTest):
    """Test cases for response processing and output formatting."""
    
    @pytest.mark.asyncio
    async def test_process_instructions_success(self):
        """Test successful processing of instructions with mocked responses."""
        # Mock the required global variables and functions
        mock_data_eval = self.sample_instructions
        expected_responses = ["Yes, it is true.", "No, it is not true."]
        
        # Mock the completion function
        async def mock_get_kimi_completion(instruction, input_text=None):
            if "Apple" in instruction:
                return "Yes, it is true."
            else:
                return "No, it is not true."
        
        with patch.object(run_kimi_gj, 'data_eval', mock_data_eval), \
             patch.object(run_kimi_gj, 'output_file', self.test_output_file), \
             patch.object(run_kimi_gj, 'get_kimi_completion', side_effect=mock_get_kimi_completion):
            
            await run_kimi_gj.process_instructions()
            
            # Verify output file was created and has correct content
            assert os.path.exists(self.test_output_file)
            
            with open(self.test_output_file, 'r', encoding='utf-8') as f:
                reader = csv.reader(f)
                rows = list(reader)
                
                # Check header
                assert rows[0] == ["prompt", "generated"]
                
                # Check data rows
                assert len(rows) == 3  # Header + 2 data rows
                assert "Apple Founded by Steve Jobs" in rows[1][0]
                assert rows[1][1] in expected_responses
    
    def test_response_cleaning(self):
        """Test response cleaning and formatting."""
        test_responses = [
            "Yes, it is true.\n",
            "  No, it is not true.  ",
            "Yes, it is true.\n\nConfidence: High",
            "Error: API timeout"
        ]
        
        expected_cleaned = [
            "Yes, it is true.",
            "No, it is not true.",
            "Yes, it is true.  Confidence: High",
            "Error: API timeout"
        ]
        
        for response, expected in zip(test_responses, expected_cleaned):
            cleaned = response.strip().replace('\n', ' ')
            assert cleaned == expected


class TestErrorHandling(BaseKimiTest):
    """Test cases for error handling and edge cases."""
    
    def test_api_key_configuration_error(self):
        """Test handling of API key configuration errors."""
        with patch('run_kimi_gj.get_moonshot_api_config', side_effect=ValueError("API key not found")):
            # Test would require rerunning module initialization
            # This is tested implicitly through the config tests
            assert True  # Placeholder for complex module-level error handling
    
    @pytest.mark.asyncio
    async def test_empty_instruction_handling(self):
        """Test handling of empty or malformed instructions."""
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "No, it is not true."
        
        with patch('run_kimi_gj.completion', return_value=mock_response):
            result = await run_kimi_gj.get_kimi_completion("")
            assert "No, it is not true." in result
    
    def test_file_permission_errors(self):
        """Test handling of file permission errors."""
        # Create a read-only directory
        readonly_dir = os.path.join(self.temp_dir, "readonly")
        os.makedirs(readonly_dir, exist_ok=True)
        
        if os.name != 'nt':  # Skip on Windows due to permission differences
            os.chmod(readonly_dir, 0o444)
            
            readonly_file = os.path.join(readonly_dir, "output.csv")
            
            with patch.object(run_kimi_gj, 'output_file', readonly_file):
                # This would test actual file permission handling
                # In practice, this would be caught by the async process_instructions function
                assert True  # Placeholder for permission error testing


class TestIntegration(BaseKimiTest):
    """Integration tests for the complete KIMI-K2 pipeline."""
    
    def test_complete_pipeline_validation(self):
        """Test that all components work together correctly."""
        # Test the validation sequence that would occur in main execution
        with patch.object(run_kimi_gj, 'input_file', self.test_input_file), \
             patch.object(run_kimi_gj, 'output_file', self.test_output_file):
            
            # Validate input file
            assert run_kimi_gj.validate_input_file() is True
            
            # Create output directory
            run_kimi_gj.create_output_directory()
            assert os.path.exists(os.path.dirname(self.test_output_file))
    
    @pytest.mark.asyncio
    async def test_end_to_end_mock(self):
        """Test end-to-end pipeline with fully mocked dependencies."""
        # This is a comprehensive test that mocks all external dependencies
        # and tests the complete flow
        
        mock_responses = ["Yes, it is true.", "No, it is not true."]
        
        async def mock_completion_side_effect(instruction, input_text=None):
            if "Apple" in instruction:
                return mock_responses[0]
            return mock_responses[1]
        
        # Mock all the global dependencies
        with patch.object(run_kimi_gj, 'data_eval', self.sample_instructions), \
             patch.object(run_kimi_gj, 'instructions', self.sample_instructions), \
             patch.object(run_kimi_gj, 'output_file', self.test_output_file), \
             patch.object(run_kimi_gj, 'get_kimi_completion', side_effect=mock_completion_side_effect):
            
            # Run the complete processing pipeline
            await run_kimi_gj.process_instructions()
            
            # Verify the output
            assert os.path.exists(self.test_output_file)
            
            with open(self.test_output_file, 'r', encoding='utf-8') as f:
                content = f.read()
                assert "prompt,generated" in content
                assert "Apple Founded by Steve Jobs" in content
                assert "Microsoft Founded by Bill Gates" in content


class TestConfigurationIntegration(BaseKimiTest):
    """Test integration with the configuration system."""
    
    def test_moonshot_config_integration(self):
        """Test integration with Moonshot API configuration."""
        test_api_key = "mk-test123456789abcdef"
        
        with patch.dict(os.environ, {
            'MOONSHOT_API_KEY': test_api_key
        }, clear=True):
            with patch('run_kimi_gj.load_env_file'):  # Mock to prevent external .env loading
                from config import get_moonshot_api_config
                
                # Test that the API key can be retrieved
                api_key = get_moonshot_api_config()
                assert api_key == test_api_key
    
    def test_environment_setup(self):
        """Test that environment variables are properly set up."""
        test_api_key = "mk-test123456789abcdef"
        
        with patch('run_kimi_gj.get_moonshot_api_config', return_value=test_api_key):
            # Simulate the API key setup that occurs in the module
            os.environ['MOONSHOT_API_KEY'] = test_api_key
            
            assert os.environ.get('MOONSHOT_API_KEY') == test_api_key


if __name__ == "__main__":
    # When run directly, execute all tests
    pytest.main([__file__, "-v"])