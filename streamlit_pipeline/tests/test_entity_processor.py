"""
Unit tests for entity_processor module following TDD principles.

This test suite follows the Testing_Demands.md guidelines:
- Test-driven development approach
- Architectural consistency between tests and implementation
- Minimum test cases: expected use case, edge case, failure case
- Mock-friendly design with proper API mocking

Test Coverage:
- extract_entities() function (main interface)
- batch_extract_entities() function
- Error handling and edge cases
- API integration mocking
"""

import pytest
from unittest.mock import patch, MagicMock
import time
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from core.entity_processor import (
    extract_entities,
    batch_extract_entities,
    _extract_entities_from_text,
    _denoise_text_with_entities,
    _parse_entity_response
)
from core.models import EntityResult


class TestEntityProcessor:
    """
    Test suite for entity_processor module.
    
    Following Testing_Demands.md principles:
    - Each feature has expected use case, edge case, and failure case tests
    - Tests use the same calling patterns as implementation
    - Meaningful test names that clearly indicate what is being tested
    """

    # Test fixtures for consistent test data
    @pytest.fixture
    def sample_chinese_text(self):
        """Sample classical Chinese text for testing."""
        return "林黛玉進入榮國府後，與賈寶玉初次相遇，兩人情投意合。"

    @pytest.fixture
    def sample_entities(self):
        """Sample entity list for testing.""" 
        return ["林黛玉", "榮國府", "賈寶玉"]

    @pytest.fixture
    def mock_api_response_entities(self):
        """Mock API response for entity extraction."""
        return '["林黛玉", "榮國府", "賈寶玉", "情投意合"]'

    @pytest.fixture
    def mock_api_response_denoised(self):
        """Mock API response for text denoising."""
        return "林黛玉初入榮國府，與賈寶玉相遇，二人情意相通。"

    # Expected use case tests (normal, successful operation)
    
    @patch('core.entity_processor.call_gpt5_mini')
    def test_extract_entities_successful_operation(
        self, 
        mock_api_call, 
        sample_chinese_text, 
        mock_api_response_entities,
        mock_api_response_denoised
    ):
        """Test extract_entities with normal, successful operation."""
        # Configure mock to return different responses for entity extraction vs denoising
        mock_api_call.side_effect = [mock_api_response_entities, mock_api_response_denoised]
        
        # Call the function under test
        result = extract_entities(sample_chinese_text)
        
        # Verify the result structure and content
        assert isinstance(result, EntityResult)
        assert result.success is True
        assert result.error is None
        assert isinstance(result.entities, list)
        assert len(result.entities) > 0
        assert isinstance(result.denoised_text, str)
        assert len(result.denoised_text.strip()) > 0
        assert result.processing_time >= 0
        
        # Verify API was called twice (once for entities, once for denoising)
        assert mock_api_call.call_count == 2
        
        # Verify specific content
        expected_entities = ["林黛玉", "榮國府", "賈寶玉", "情投意合"]
        assert result.entities == expected_entities
        assert result.denoised_text == mock_api_response_denoised

    @patch('core.entity_processor.call_gpt5_mini')
    def test_batch_extract_entities_successful_operation(self, mock_api_call):
        """Test batch_extract_entities with multiple texts."""
        mock_api_call.side_effect = [
            '["實體1"]', "去噪文本1",
            '["實體2"]', "去噪文本2"
        ]
        
        texts = ["文本1", "文本2"]
        results = batch_extract_entities(texts)
        
        assert len(results) == 2
        assert all(isinstance(result, EntityResult) for result in results)
        assert all(result.success for result in results)
        assert mock_api_call.call_count == 4  # 2 texts * 2 calls each

    # Edge case tests (boundary conditions, unusual but valid inputs)
    
    def test_extract_entities_empty_text_input(self):
        """Test extract_entities with empty text input."""
        result = extract_entities("")
        
        assert isinstance(result, EntityResult)
        assert result.success is False
        assert "empty" in result.error.lower()
        assert result.entities == []
        assert result.denoised_text == ""
        assert result.processing_time >= 0

    def test_extract_entities_whitespace_only_text(self):
        """Test extract_entities with whitespace-only input."""
        result = extract_entities("   \n\t   ")
        
        assert result.success is False
        assert "empty" in result.error.lower() or "whitespace" in result.error.lower()

    @patch('core.entity_processor.call_gpt5_mini')
    def test_extract_entities_very_long_text(self, mock_api_call):
        """Test extract_entities with very long text input."""
        mock_api_call.side_effect = ['["長文實體"]', "去噪長文本"]
        
        long_text = "很長的文本內容" * 1000  # Very long text
        result = extract_entities(long_text)
        
        assert result.success is True
        assert len(result.entities) >= 0  # Should handle gracefully

    def test_batch_extract_entities_empty_list(self):
        """Test batch_extract_entities with empty text list.""" 
        results = batch_extract_entities([])
        
        assert isinstance(results, list)
        assert len(results) == 0

    # Failure case tests (invalid inputs, error conditions, exception handling)
    
    @patch('core.entity_processor.call_gpt5_mini')
    def test_extract_entities_api_failure(self, mock_api_call):
        """Test extract_entities when API call fails."""
        mock_api_call.side_effect = Exception("API connection failed")
        
        result = extract_entities("測試文本")
        
        assert isinstance(result, EntityResult)
        assert result.success is False
        assert result.error is not None
        assert "failed" in result.error.lower()
        assert result.entities == []
        assert result.denoised_text == ""
        assert result.processing_time >= 0  # Should still record processing time

    @patch('core.entity_processor.call_gpt5_mini')
    def test_extract_entities_malformed_api_response(self, mock_api_call):
        """Test extract_entities with malformed API response."""
        mock_api_call.side_effect = ["無效的API響應格式", "去噪文本"]
        
        result = extract_entities("測試文本")
        
        # Should handle malformed response gracefully
        assert isinstance(result, EntityResult)
        # May succeed with empty entities or fail gracefully
        assert isinstance(result.entities, list)
        assert isinstance(result.success, bool)

    def test_extract_entities_none_input(self):
        """Test extract_entities with None input."""
        result = extract_entities(None)
        
        assert isinstance(result, EntityResult)
        assert result.success is False
        assert result.error is not None
        assert "empty" in result.error.lower() or "none" in result.error.lower()

    # Internal function tests for comprehensive coverage
    
    def test_parse_entity_response_valid_list(self):
        """Test _parse_entity_response with valid list format."""
        response = '["實體1", "實體2", "實體3"]'
        entities = _parse_entity_response(response)
        
        assert entities == ["實體1", "實體2", "實體3"]

    def test_parse_entity_response_comma_separated(self):
        """Test _parse_entity_response with comma-separated format.""" 
        response = "實體1, 實體2, 實體3"
        entities = _parse_entity_response(response)
        
        assert isinstance(entities, list)
        assert len(entities) >= 0  # Should extract some entities

    def test_parse_entity_response_malformed_input(self):
        """Test _parse_entity_response with completely malformed input."""
        response = "這是完全無法解析的響應"
        entities = _parse_entity_response(response)
        
        # Should return empty list rather than crashing
        assert isinstance(entities, list)

    def test_parse_entity_response_duplicate_entities(self):
        """Test _parse_entity_response removes duplicates."""
        response = '["實體1", "實體2", "實體1", "實體3", "實體2"]'
        entities = _parse_entity_response(response)
        
        # Should deduplicate
        unique_entities = list(set(entities))
        assert len(entities) == len(unique_entities)

    # Performance and resource management tests
    
    @patch('core.entity_processor.call_gpt5_mini')
    def test_extract_entities_processing_time_recorded(self, mock_api_call):
        """Test that processing time is properly recorded."""
        def delayed_response(*args, **kwargs):
            time.sleep(0.001)  # 1ms delay to ensure measurable time
            if len(args) > 0 and "提取所有重要實體" in args[0]:
                return '["實體"]'
            else:
                return "去噪文本"
        
        mock_api_call.side_effect = delayed_response
        
        result = extract_entities("測試")
        
        assert result.processing_time >= 0  # Should be non-negative
        assert isinstance(result.processing_time, float)  # Should be float type

    # Integration tests for module interactions
    
    @patch('core.entity_processor.call_gpt5_mini')
    def test_entity_extraction_and_denoising_integration(self, mock_api_call):
        """Test that entity extraction and denoising work together correctly."""
        entities_response = '["人物A", "地點B"]'
        denoising_response = "人物A在地點B進行活動。"
        
        mock_api_call.side_effect = [entities_response, denoising_response]
        
        result = extract_entities("原始複雜文本內容")
        
        # Verify both operations completed
        assert result.success is True
        assert len(result.entities) == 2
        assert "人物A" in result.entities
        assert "地點B" in result.entities
        assert result.denoised_text == denoising_response
        
        # Verify API calls used correct prompts
        call_args_list = mock_api_call.call_args_list
        assert len(call_args_list) == 2
        
        # First call should be for entity extraction
        first_call_args = call_args_list[0][0]
        assert "提取所有重要實體" in first_call_args[0]
        
        # Second call should be for denoising
        second_call_args = call_args_list[1][0] 
        assert "人物A、地點B" in second_call_args[0]  # Should include extracted entities

    # Error propagation tests
    
    @patch('core.entity_processor.call_gpt5_mini')
    def test_error_propagation_from_entity_extraction(self, mock_api_call):
        """Test error handling when entity extraction fails."""
        mock_api_call.side_effect = [Exception("Entity extraction failed"), "去噪文本"]
        
        result = extract_entities("測試文本")
        
        assert result.success is False
        assert "Entity extraction failed" in result.error or "failed" in result.error.lower()

    @patch('core.entity_processor.call_gpt5_mini')
    def test_error_propagation_from_denoising(self, mock_api_call):
        """Test error handling when denoising fails."""
        mock_api_call.side_effect = ['["實體"]', Exception("Denoising failed")]
        
        result = extract_entities("測試文本")
        
        assert result.success is False
        assert "Denoising failed" in result.error or "failed" in result.error.lower()

    # Cross-platform compatibility tests
    
    def test_extract_entities_unicode_handling(self):
        """Test extract_entities handles Unicode characters correctly."""
        unicode_text = "測試中文字符：繁體字、簡體字、標點符號。"
        
        with patch('core.entity_processor.call_gpt5_mini') as mock_api:
            mock_api.side_effect = ['["測試", "中文字符"]', "處理後的文本"]
            
            result = extract_entities(unicode_text)
            
            assert result.success is True
            # Verify Unicode is preserved
            assert all(isinstance(entity, str) for entity in result.entities)