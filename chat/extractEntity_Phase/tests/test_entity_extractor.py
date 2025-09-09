"""
Unit tests for Entity Extractor Module

This module tests the core entity extraction functionality including:
- Entity extraction from Chinese text
- Prompt generation and API request handling
- Response parsing and entity creation
- Deduplication and validation logic
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch
from typing import List

from extractEntity_Phase.core.entity_extractor import EntityExtractor, ExtractionConfig
from extractEntity_Phase.models.entities import Entity, EntityType, EntityList
from extractEntity_Phase.api.gpt5mini_client import APIRequest, APIResponse


class TestExtractionConfig:
    """Test ExtractionConfig dataclass."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = ExtractionConfig()
        
        assert config.temperature == 1.0
        assert config.max_tokens == 4000
        assert config.enable_deduplication is True
        assert config.min_confidence == 0.7
        assert config.max_entities_per_text == 50
        assert config.use_system_prompt is True
        assert config.include_examples is True
        assert config.language == "zh-TW"
        assert config.batch_size == 10
        assert config.max_concurrent == 3
    
    def test_custom_config(self):
        """Test custom configuration values."""
        config = ExtractionConfig(
            temperature=0.8,
            max_tokens=2000,
            enable_deduplication=False,
            min_confidence=0.5,
            max_entities_per_text=25,
            batch_size=5,
            max_concurrent=2
        )
        
        assert config.temperature == 0.8
        assert config.max_tokens == 2000
        assert config.enable_deduplication is False
        assert config.min_confidence == 0.5
        assert config.max_entities_per_text == 25
        assert config.batch_size == 5
        assert config.max_concurrent == 2


class TestEntityExtractor:
    """Test EntityExtractor class."""
    
    @pytest.fixture
    def mock_client(self):
        """Create mock GPT-5-mini client."""
        client = Mock()
        client.complete = AsyncMock()
        return client
    
    @pytest.fixture
    def mock_logger(self):
        """Create mock logger."""
        logger = Mock()
        logger.log = Mock()
        return logger
    
    @pytest.fixture
    def mock_text_processor(self):
        """Create mock Chinese text processor."""
        processor = Mock()
        processor.classify_entity_type = Mock(return_value=EntityType.PERSON)
        processor.normalize_text = Mock(side_effect=lambda x: x)
        processor.is_valid_chinese_text = Mock(return_value=True)
        return processor
    
    @pytest.fixture
    def entity_extractor(self, mock_client, mock_logger, mock_text_processor):
        """Create EntityExtractor instance with mocked dependencies."""
        with patch('extractEntity_Phase.core.entity_extractor.get_logger', return_value=mock_logger):
            with patch('extractEntity_Phase.core.entity_extractor.ChineseTextProcessor', return_value=mock_text_processor):
                extractor = EntityExtractor(mock_client)
                return extractor
    
    @pytest.fixture
    def sample_texts(self):
        """Sample Chinese texts for testing."""
        return [
            "甄士隱於書房閒坐，至手倦拋書，伏几少憩，不覺朦朧睡去。",
            "這閶門外有個十里街，街內有個仁清巷，巷內有個古廟，因地方窄狹，人皆呼作葫蘆廟。",
            "廟旁住著一家鄉宦，姓甄，名費，字士隱。嫡妻封氏，情性賢淑，深明禮義。"
        ]
    
    @pytest.fixture
    def sample_api_responses(self):
        """Sample API responses for testing."""
        return [
            APIResponse(
                content='["甄士隱", "書房"]',
                model="gpt-5-mini",
                usage={"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150},
                finish_reason="stop",
                response_time=1.5,
                cached=False
            ),
            APIResponse(
                content='["閶門", "十里街", "仁清巷", "古廟", "葫蘆廟"]',
                model="gpt-5-mini",
                usage={"prompt_tokens": 120, "completion_tokens": 60, "total_tokens": 180},
                finish_reason="stop",
                response_time=1.8,
                cached=False
            ),
            APIResponse(
                content='["甄士隱", "封氏", "鄉宦"]',
                model="gpt-5-mini",
                usage={"prompt_tokens": 110, "completion_tokens": 55, "total_tokens": 165},
                finish_reason="stop",
                response_time=1.6,
                cached=False
            )
        ]
    
    def test_initialization(self, entity_extractor):
        """Test EntityExtractor initialization."""
        assert entity_extractor.client is not None
        assert entity_extractor.config is not None
        assert entity_extractor.logger is not None
        assert entity_extractor.text_processor is not None
        assert entity_extractor.stats["total_texts_processed"] == 0
    
    def test_validate_input_texts_valid(self, entity_extractor, sample_texts):
        """Test input text validation with valid texts."""
        validated = entity_extractor._validate_input_texts(sample_texts)
        assert len(validated) == 3
        assert validated == sample_texts
    
    def test_validate_input_texts_empty(self, entity_extractor):
        """Test input text validation with empty texts."""
        texts = ["", "   ", None, "short"]
        validated = entity_extractor._validate_input_texts(texts)
        assert len(validated) == 0
    
    def test_validate_input_texts_too_short(self, entity_extractor):
        """Test input text validation with texts that are too short."""
        texts = ["短", "very short text", "a"]
        validated = entity_extractor._validate_input_texts(texts)
        assert len(validated) == 0
    
    def test_build_extraction_prompt_with_examples(self, entity_extractor):
        """Test prompt building with examples."""
        text = "測試文本"
        prompt = entity_extractor._build_prompt_with_examples(text)
        
        assert "目標：" in prompt
        assert "範例#1:" in prompt
        assert "範例#5:" in prompt
        assert text in prompt
        assert "實體列表：" in prompt
    
    def test_build_extraction_prompt_simple(self, entity_extractor):
        """Test simple prompt building without examples."""
        entity_extractor.config.include_examples = False
        text = "測試文本"
        prompt = entity_extractor._build_simple_prompt(text)
        
        assert "從以下古典中文文本中提取實體" in prompt
        assert text in prompt
        assert "實體列表：" in prompt
    
    def test_build_system_prompt(self, entity_extractor):
        """Test system prompt building."""
        prompt = entity_extractor._build_system_prompt()
        
        assert "你是一個專門處理古典中文文本的實體提取專家" in prompt
        assert "提取人物、地點、物品、概念等重要實體" in prompt
        assert "必須去除重複的實體" in prompt
        assert "返回格式必須是Python列表格式" in prompt
    
    def test_extract_entities_list_python_format(self, entity_extractor):
        """Test entity list extraction from Python list format."""
        content = '["實體1", "實體2", "實體3"]'
        entities = entity_extractor._extract_entities_list(content)
        
        assert entities == ["實體1", "實體2", "實體3"]
    
    def test_extract_entities_list_quoted_strings(self, entity_extractor):
        """Test entity list extraction from quoted strings."""
        content = '"實體1" "實體2" "實體3"'
        entities = entity_extractor._extract_entities_list(content)
        
        assert entities == ["實體1", "實體2", "實體3"]
    
    def test_extract_entities_list_delimiters(self, entity_extractor):
        """Test entity list extraction using delimiters."""
        content = "實體1，實體2；實體3"
        entities = entity_extractor._extract_entities_list(content)
        
        assert entities == ["實體1", "實體2", "實體3"]
    
    def test_extract_entities_list_fallback(self, entity_extractor):
        """Test entity list extraction fallback behavior."""
        content = "No entities found"
        entities = entity_extractor._extract_entities_list(content)
        
        assert entities == []
    
    def test_create_entity(self, entity_extractor):
        """Test entity creation from text."""
        entity_text = "甄士隱"
        source_text = "甄士隱於書房閒坐"
        
        entity = entity_extractor._create_entity(entity_text, source_text)
        
        assert entity is not None
        assert entity.text == "甄士隱"
        assert entity.type == EntityType.PERSON
        assert entity.confidence == 0.9
        assert entity.start_pos == 0
        assert entity.end_pos == 3
        assert entity.source_text == source_text
        assert entity.metadata["extraction_method"] == "gpt5mini"
    
    def test_create_entity_invalid(self, entity_extractor):
        """Test entity creation with invalid text."""
        entity = entity_extractor._create_entity("", "source")
        assert entity is None
        
        entity = entity_extractor._create_entity("a", "source")
        assert entity is None
    
    def test_deduplicate_entities(self, entity_extractor):
        """Test entity deduplication."""
        entities = [
            Entity(text="實體1", type=EntityType.PERSON, confidence=0.9),
            Entity(text="實體2", type=EntityType.LOCATION, confidence=0.8),
            Entity(text="實體1", type=EntityType.PERSON, confidence=0.95),  # Higher confidence
            Entity(text="實體3", type=EntityType.OBJECT, confidence=0.7)
        ]
        
        deduplicated = entity_extractor._deduplicate_entities(entities)
        
        assert len(deduplicated) == 3
        # Should keep the higher confidence version of "實體1"
        entity1 = next(e for e in deduplicated if e.text == "實體1")
        assert entity1.confidence == 0.95
    
    @pytest.mark.asyncio
    async def test_extract_entities_from_texts_success(self, entity_extractor, sample_texts, sample_api_responses):
        """Test successful entity extraction from texts."""
        # Mock the client responses
        entity_extractor.client.complete.side_effect = sample_api_responses
        
        # Mock asyncio.get_event_loop().time()
        with patch('asyncio.get_event_loop') as mock_loop:
            mock_loop.return_value.time.return_value = 1234567890.0
            
            results = await entity_extractor.extract_entities_from_texts(sample_texts)
        
        assert len(results) == 3
        assert all(isinstance(result, EntityList) for result in results)
        
        # Check first result
        first_result = results[0]
        assert len(first_result.entities) == 2
        assert first_result.entities[0].text == "甄士隱"
        assert first_result.entities[1].text == "書房"
        
        # Check statistics
        assert entity_extractor.stats["total_texts_processed"] == 3
        assert entity_extractor.stats["total_entities_extracted"] == 10
        assert entity_extractor.stats["successful_extractions"] == 3
    
    @pytest.mark.asyncio
    async def test_extract_entities_from_texts_empty_input(self, entity_extractor):
        """Test entity extraction with empty input."""
        results = await entity_extractor.extract_entities_from_texts([])
        assert results == []
    
    @pytest.mark.asyncio
    async def test_extract_entities_from_texts_validation_failure(self, entity_extractor):
        """Test entity extraction with validation failure."""
        # Mock text processor to reject all texts
        entity_extractor.text_processor.is_valid_chinese_text.return_value = False
        
        results = await entity_extractor.extract_entities_from_texts(["test"])
        assert results == []
    
    @pytest.mark.asyncio
    async def test_extract_entities_batch(self, entity_extractor, sample_texts, sample_api_responses):
        """Test entity extraction batch processing."""
        # Mock the client responses
        entity_extractor.client.complete.side_effect = sample_api_responses
        
        # Mock asyncio.get_event_loop().time()
        with patch('asyncio.get_event_loop') as mock_loop:
            mock_loop.return_value.time.return_value = 1234567890.0
            
            results = await entity_extractor._extract_entities_batch(sample_texts)
        
        assert len(results) == 3
        assert all(isinstance(result, EntityList) for result in results)
    
    @pytest.mark.asyncio
    async def test_execute_extraction_requests_success(self, entity_extractor, sample_api_responses):
        """Test successful API request execution."""
        requests = [
            APIRequest(prompt="test1"),
            APIRequest(prompt="test2"),
            APIRequest(prompt="test3")
        ]
        
        # Mock the client responses
        entity_extractor.client.complete.side_effect = sample_api_responses
        
        responses = await entity_extractor._execute_extraction_requests(requests)
        
        assert len(responses) == 3
        assert all(isinstance(response, APIResponse) for response in responses)
    
    @pytest.mark.asyncio
    async def test_execute_extraction_requests_failure(self, entity_extractor):
        """Test API request execution with failures."""
        requests = [APIRequest(prompt="test")]
        
        # Mock the client to raise an exception
        entity_extractor.client.complete.side_effect = Exception("API Error")
        
        responses = await entity_extractor._execute_extraction_requests(requests)
        
        assert len(responses) == 1
        assert responses[0].error == "API Error"
    
    def test_parse_extraction_response_success(self, entity_extractor, sample_api_responses):
        """Test successful response parsing."""
        response = sample_api_responses[0]
        source_text = "甄士隱於書房閒坐"
        
        entities = entity_extractor._parse_extraction_response(response, source_text)
        
        assert len(entities) == 2
        assert entities[0].text == "甄士隱"
        assert entities[1].text == "書房"
    
    def test_parse_extraction_response_error(self, entity_extractor):
        """Test response parsing with error."""
        response = APIResponse(
            content="",
            model="gpt-5-mini",
            usage={"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
            finish_reason="error",
            response_time=0.0,
            error="API Error"
        )
        source_text = "test"
        
        with pytest.raises(ValueError, match="API response error: API Error"):
            entity_extractor._parse_extraction_response(response, source_text)
    
    def test_parse_extraction_response_empty_content(self, entity_extractor):
        """Test response parsing with empty content."""
        response = APIResponse(
            content="",
            model="gpt-5-mini",
            usage={"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
            finish_reason="stop",
            response_time=0.0
        )
        source_text = "test"
        
        with pytest.raises(ValueError, match="Empty response content"):
            entity_extractor._parse_extraction_response(response, source_text)
    
    def test_classify_entity_type(self, entity_extractor):
        """Test entity type classification."""
        entity_text = "甄士隱"
        source_text = "甄士隱於書房閒坐"
        
        entity_type = entity_extractor._classify_entity_type(entity_text, source_text)
        
        assert entity_type == EntityType.PERSON
        entity_extractor.text_processor.classify_entity_type.assert_called_once_with(entity_text, source_text)
    
    def test_get_statistics(self, entity_extractor):
        """Test statistics retrieval."""
        # Set some statistics
        entity_extractor.stats["total_texts_processed"] = 10
        entity_extractor.stats["total_entities_extracted"] = 25
        
        stats = entity_extractor.get_statistics()
        
        assert stats["total_texts_processed"] == 10
        assert stats["total_entities_extracted"] == 25
        assert stats is not entity_extractor.stats  # Should return a copy
    
    def test_reset_statistics(self, entity_extractor):
        """Test statistics reset."""
        # Set some statistics
        entity_extractor.stats["total_texts_processed"] = 10
        entity_extractor.stats["total_entities_extracted"] = 25
        
        entity_extractor.reset_statistics()
        
        assert entity_extractor.stats["total_texts_processed"] == 0
        assert entity_extractor.stats["total_entities_extracted"] == 0
        assert entity_extractor.stats["successful_extractions"] == 0
        assert entity_extractor.stats["failed_extractions"] == 0
        assert entity_extractor.stats["cache_hits"] == 0
        assert entity_extractor.stats["cache_misses"] == 0


class TestEntityExtractorIntegration:
    """Integration tests for EntityExtractor."""
    
    @pytest.mark.asyncio
    async def test_full_extraction_workflow(self):
        """Test the complete entity extraction workflow."""
        # This test would require more complex mocking and setup
        # For now, we'll test the main components work together
        pass
    
    @pytest.mark.asyncio
    async def test_error_handling_and_recovery(self):
        """Test error handling and recovery mechanisms."""
        # This test would verify that the extractor can handle
        # various error conditions gracefully
        pass
