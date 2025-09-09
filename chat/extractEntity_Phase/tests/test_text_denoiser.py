"""
Unit tests for Text Denoiser Module

This module tests the core text denoising functionality including:
- Text denoising based on extracted entities
- Prompt generation and API request handling
- Response parsing and text validation
- Quality assessment and fallback mechanisms
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch
from typing import List

from extractEntity_Phase.core.text_denoiser import TextDenoiser, DenoisingConfig
from extractEntity_Phase.models.entities import Entity, EntityType, EntityList
from extractEntity_Phase.api.gpt5mini_client import APIRequest, APIResponse


class TestDenoisingConfig:
    """Test DenoisingConfig dataclass."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = DenoisingConfig()
        
        assert config.temperature == 1.0
        assert config.max_tokens == 4000
        assert config.preserve_classical_style is True
        assert config.maintain_factual_accuracy is True
        assert config.enable_entity_relationships is True
        assert config.min_output_length == 20
        assert config.max_output_length == 1000
        assert config.use_system_prompt is True
        assert config.include_examples is True
        assert config.language == "zh-TW"
        assert config.batch_size == 10
        assert config.max_concurrent == 3
    
    def test_custom_config(self):
        """Test custom configuration values."""
        config = DenoisingConfig(
            temperature=0.8,
            max_tokens=2000,
            preserve_classical_style=False,
            min_output_length=10,
            max_output_length=500,
            batch_size=5,
            max_concurrent=2
        )
        
        assert config.temperature == 0.8
        assert config.max_tokens == 2000
        assert config.preserve_classical_style is False
        assert config.min_output_length == 10
        assert config.max_output_length == 500
        assert config.batch_size == 5
        assert config.max_concurrent == 2


class TestTextDenoiser:
    """Test TextDenoiser class."""
    
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
        processor.is_valid_chinese_text = Mock(return_value=True)
        return processor
    
    @pytest.fixture
    def text_denoiser(self, mock_client, mock_logger, mock_text_processor):
        """Create TextDenoiser instance with mocked dependencies."""
        with patch('extractEntity_Phase.core.text_denoiser.get_logger', return_value=mock_logger):
            with patch('extractEntity_Phase.core.text_denoiser.ChineseTextProcessor', return_value=mock_text_processor):
                denoiser = TextDenoiser(mock_client)
                return denoiser
    
    @pytest.fixture
    def sample_texts(self):
        """Sample Chinese texts for testing."""
        return [
            "廟旁住著一家鄉宦，姓甄，名費，字士隱。嫡妻封氏，情性賢淑，深明禮義。",
            "賈雨村原系胡州人氏，也是詩書仕宦之族，因他生於末世，暫寄廟中安身。",
            "賈寶玉因夢遊太虛幻境，頓生疑懼，醒來後對林黛玉說起此事。"
        ]
    
    @pytest.fixture
    def sample_entities(self):
        """Sample entity collections for testing."""
        return [
            EntityList(
                entities=[
                    Entity(text="甄士隱", type=EntityType.PERSON),
                    Entity(text="封氏", type=EntityType.PERSON),
                    Entity(text="鄉宦", type=EntityType.ORGANIZATION)
                ]
            ),
            EntityList(
                entities=[
                    Entity(text="賈雨村", type=EntityType.PERSON),
                    Entity(text="胡州", type=EntityType.LOCATION),
                    Entity(text="詩書仕宦之族", type=EntityType.ORGANIZATION)
                ]
            ),
            EntityList(
                entities=[
                    Entity(text="賈寶玉", type=EntityType.PERSON),
                    Entity(text="太虛幻境", type=EntityType.CONCEPT),
                    Entity(text="林黛玉", type=EntityType.PERSON)
                ]
            )
        ]
    
    @pytest.fixture
    def sample_api_responses(self):
        """Sample API responses for testing."""
        return [
            APIResponse(
                content="甄士隱是一家鄉宦。甄士隱姓甄名費字士隱。甄士隱的妻子是封氏。封氏情性賢淑深明禮義。甄家是本地望族。",
                model="gpt-5-mini",
                usage={"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150},
                finish_reason="stop",
                response_time=1.5,
                cached=False
            ),
            APIResponse(
                content="賈雨村是胡州人氏。賈雨村是詩書仕宦之族。賈雨村生於末世。賈雨村進京求取功名。賈雨村想要重整基業。",
                model="gpt-5-mini",
                usage={"prompt_tokens": 120, "completion_tokens": 60, "total_tokens": 180},
                finish_reason="stop",
                response_time=1.8,
                cached=False
            ),
            APIResponse(
                content="賈寶玉夢遊太虛幻境。賈寶玉夢醒後頓生疑懼。賈寶玉將此事告知林黛玉。林黛玉聽後感到驚異。",
                model="gpt-5-mini",
                usage={"prompt_tokens": 110, "completion_tokens": 55, "total_tokens": 165},
                finish_reason="stop",
                response_time=1.6,
                cached=False
            )
        ]
    
    def test_initialization(self, text_denoiser):
        """Test TextDenoiser initialization."""
        assert text_denoiser.client is not None
        assert text_denoiser.config is not None
        assert text_denoiser.logger is not None
        assert text_denoiser.text_processor is not None
        assert text_denoiser.stats["total_texts_processed"] == 0
    
    def test_validate_input_pairs_valid(self, text_denoiser, sample_texts, sample_entities):
        """Test input validation with valid text-entity pairs."""
        validated = text_denoiser._validate_input_pairs(sample_texts, sample_entities)
        assert len(validated) == 3
        assert all(isinstance(pair, tuple) for pair in validated)
        assert all(len(pair) == 2 for pair in validated)
    
    def test_validate_input_pairs_mismatch(self, text_denoiser):
        """Test input validation with mismatched text and entity counts."""
        texts = ["text1", "text2"]
        entities = [EntityList(entities=[Entity(text="entity1")])]
        
        validated = text_denoiser._validate_input_pairs(texts, entities)
        assert len(validated) == 0
    
    def test_validate_input_pairs_empty_text(self, text_denoiser, sample_entities):
        """Test input validation with empty texts."""
        texts = ["", "   ", None]
        validated = text_denoiser._validate_input_pairs(texts, sample_entities[:3])
        assert len(validated) == 0
    
    def test_validate_input_pairs_too_short(self, text_denoiser, sample_entities):
        """Test input validation with texts that are too short."""
        texts = ["短", "very short", "a"]
        validated = text_denoiser._validate_input_pairs(texts, sample_entities[:3])
        assert len(validated) == 0
    
    def test_validate_input_pairs_no_entities(self, text_denoiser, sample_texts):
        """Test input validation with empty entity collections."""
        entities = [
            EntityList(entities=[]),
            EntityList(entities=[]),
            EntityList(entities=[])
        ]
        
        validated = text_denoiser._validate_input_pairs(sample_texts, entities)
        assert len(validated) == 0
    
    def test_extract_entity_strings_from_entity_collection(self, text_denoiser, sample_entities):
        """Test entity string extraction from EntityList."""
        entity_strings = text_denoiser._extract_entity_strings(sample_entities[0])
        
        assert entity_strings == ["甄士隱", "封氏", "鄉宦"]
    
    def test_extract_entity_strings_from_list_of_strings(self, text_denoiser):
        """Test entity string extraction from list of strings."""
        entities = ["實體1", "實體2", "實體3"]
        entity_strings = text_denoiser._extract_entity_strings(entities)
        
        assert entity_strings == ["實體1", "實體2", "實體3"]
    
    def test_extract_entity_strings_from_list_of_entities(self, text_denoiser):
        """Test entity string extraction from list of Entity objects."""
        entities = [
            Entity(text="實體1", type=EntityType.PERSON),
            Entity(text="實體2", type=EntityType.LOCATION),
            Entity(text="實體3", type=EntityType.OBJECT)
        ]
        entity_strings = text_denoiser._extract_entity_strings(entities)
        
        assert entity_strings == ["實體1", "實體2", "實體3"]
    
    def test_extract_entity_strings_fallback(self, text_denoiser):
        """Test entity string extraction fallback behavior."""
        entities = 123  # Non-standard type
        entity_strings = text_denoiser._extract_entity_strings(entities)
        
        assert entity_strings == ["123"]
    
    def test_build_denoising_prompt_with_examples(self, text_denoiser):
        """Test prompt building with examples."""
        text = "測試文本"
        entities = ["實體1", "實體2"]
        prompt = text_denoiser._build_prompt_with_examples(text, entities)
        
        assert "目標：" in prompt
        assert "範例#1:" in prompt
        assert "範例#3:" in prompt
        assert text in prompt
        assert str(entities) in prompt
        assert "去噪文本：" in prompt
    
    def test_build_denoising_prompt_simple(self, text_denoiser):
        """Test simple prompt building without examples."""
        text_denoiser.config.include_examples = False
        text = "測試文本"
        entities = ["實體1", "實體2"]
        prompt = text_denoiser._build_simple_prompt(text, entities)
        
        assert "基於給定的實體，對以下古典中文文本進行去噪處理" in prompt
        assert text in prompt
        assert str(entities) in prompt
        assert "去噪文本：" in prompt
    
    def test_build_system_prompt(self, text_denoiser):
        """Test system prompt building."""
        prompt = text_denoiser._build_system_prompt()
        
        assert "你是一個專門處理古典中文文本的去噪專家" in prompt
        assert "基於給定的實體，對文本進行去噪處理" in prompt
        assert "移除無關的描述性文字和修飾語" in prompt
        assert "重組為清晰、簡潔的事實陳述" in prompt
        assert "保持古典中文的語言風格和韻味" in prompt
    
    def test_clean_denoised_text_remove_prefixes(self, text_denoiser):
        """Test cleaning denoised text by removing prefixes."""
        content = "去噪文本：這是清理後的文本"
        cleaned = text_denoiser._clean_denoised_text(content)
        
        assert cleaned == "這是清理後的文本"
    
    def test_clean_denoised_text_remove_suffixes(self, text_denoiser):
        """Test cleaning denoised text by removing suffixes."""
        content = "這是清理後的文本。"
        cleaned = text_denoiser._clean_denoised_text(content)
        
        assert cleaned == "這是清理後的文本"
    
    def test_clean_denoised_text_normalize_whitespace(self, text_denoiser):
        """Test cleaning denoised text by normalizing whitespace."""
        content = "這是  清理後的   文本"
        cleaned = text_denoiser._clean_denoised_text(content)
        
        assert cleaned == "這是 清理後的 文本"
    
    def test_validate_denoised_text_valid(self, text_denoiser, sample_entities):
        """Test denoised text validation with valid text."""
        denoised_text = "甄士隱是一家鄉宦。甄士隱的妻子是封氏。"
        original_text = "廟旁住著一家鄉宦，姓甄，名費，字士隱。嫡妻封氏，情性賢淑，深明禮義。"
        entities = sample_entities[0]
        
        is_valid = text_denoiser._validate_denoised_text(denoised_text, original_text, entities)
        assert is_valid is True
    
    def test_validate_denoised_text_too_short(self, text_denoiser, sample_entities):
        """Test denoised text validation with text that is too short."""
        denoised_text = "短"
        original_text = "很長的原始文本"
        entities = sample_entities[0]
        
        is_valid = text_denoiser._validate_denoised_text(denoised_text, original_text, entities)
        assert is_valid is False
    
    def test_validate_denoised_text_too_long(self, text_denoiser, sample_entities):
        """Test denoised text validation with text that is too long."""
        # Create a very long text
        denoised_text = "很長的文本。" * 200  # Exceeds max_output_length
        original_text = "原始文本"
        entities = sample_entities[0]
        
        is_valid = text_denoiser._validate_denoised_text(denoised_text, original_text, entities)
        assert is_valid is False
    
    def test_validate_denoised_text_low_entity_coverage(self, text_denoiser):
        """Test denoised text validation with low entity coverage."""
        denoised_text = "這是一個不包含任何實體的文本"
        original_text = "原始文本"
        entities = EntityList(entities=[
            Entity(text="實體1", type=EntityType.PERSON),
            Entity(text="實體2", type=EntityType.LOCATION)
        ])
        
        is_valid = text_denoiser._validate_denoised_text(denoised_text, original_text, entities)
        assert is_valid is False
    
    def test_validate_denoised_text_too_similar(self, text_denoiser, sample_entities):
        """Test denoised text validation with text too similar to original."""
        denoised_text = "廟旁住著一家鄉宦，姓甄，名費，字士隱。嫡妻封氏，情性賢淑，深明禮義。"
        original_text = "廟旁住著一家鄉宦，姓甄，名費，字士隱。嫡妻封氏，情性賢淑，深明禮義。"
        entities = sample_entities[0]
        
        is_valid = text_denoiser._validate_denoised_text(denoised_text, original_text, entities)
        assert is_valid is False
    
    def test_calculate_similarity_identical(self, text_denoiser):
        """Test similarity calculation with identical texts."""
        text1 = "相同的文本"
        text2 = "相同的文本"
        
        similarity = text_denoiser._calculate_similarity(text1, text2)
        assert similarity == 1.0
    
    def test_calculate_similarity_different(self, text_denoiser):
        """Test similarity calculation with different texts."""
        text1 = "文本一"
        text2 = "文本二"
        
        similarity = text_denoiser._calculate_similarity(text1, text2)
        assert similarity < 1.0
        assert similarity > 0.0
    
    def test_calculate_similarity_empty(self, text_denoiser):
        """Test similarity calculation with empty texts."""
        similarity = text_denoiser._calculate_similarity("", "test")
        assert similarity == 0.0
        
        similarity = text_denoiser._calculate_similarity("test", "")
        assert similarity == 0.0
    
    @pytest.mark.asyncio
    async def test_denoise_texts_success(self, text_denoiser, sample_texts, sample_entities, sample_api_responses):
        """Test successful text denoising."""
        # Mock the client responses
        text_denoiser.client.complete.side_effect = sample_api_responses
        
        results = await text_denoiser.denoise_texts(sample_texts, sample_entities)
        
        assert len(results) == 3
        assert all(isinstance(result, str) for result in results)
        assert all(len(result) > 0 for result in results)
        
        # Check statistics
        assert text_denoiser.stats["total_texts_processed"] == 3
        assert text_denoiser.stats["total_texts_denoised"] == 3
        assert text_denoiser.stats["successful_denoising"] == 3
        assert text_denoiser.stats["failed_denoising"] == 0
    
    @pytest.mark.asyncio
    async def test_denoise_texts_empty_input(self, text_denoiser):
        """Test text denoising with empty input."""
        results = await text_denoiser.denoise_texts([], [])
        assert results == []
    
    @pytest.mark.asyncio
    async def test_denoise_texts_mismatch(self, text_denoiser):
        """Test text denoising with mismatched text and entity counts."""
        results = await text_denoiser.denoise_texts(["text1"], [])
        assert results == []
    
    @pytest.mark.asyncio
    async def test_denoise_texts_validation_failure(self, text_denoiser, sample_entities):
        """Test text denoising with validation failure."""
        # Mock text processor to reject all texts
        text_denoiser.text_processor.is_valid_chinese_text.return_value = False
        
        results = await text_denoiser.denoise_texts(["test"], sample_entities[:1])
        assert results == []
    
    @pytest.mark.asyncio
    async def test_denoise_texts_batch(self, text_denoiser, sample_texts, sample_entities, sample_api_responses):
        """Test text denoising batch processing."""
        # Mock the client responses
        text_denoiser.client.complete.side_effect = sample_api_responses
        
        results = await text_denoiser._denoise_texts_batch(list(zip(sample_texts, sample_entities)))
        
        assert len(results) == 3
        assert all(isinstance(result, str) for result in results)
    
    @pytest.mark.asyncio
    async def test_execute_denoising_requests_success(self, text_denoiser, sample_api_responses):
        """Test successful API request execution."""
        requests = [
            APIRequest(prompt="test1"),
            APIRequest(prompt="test2"),
            APIRequest(prompt="test3")
        ]
        
        # Mock the client responses
        text_denoiser.client.complete.side_effect = sample_api_responses
        
        responses = await text_denoiser._execute_denoising_requests(requests)
        
        assert len(responses) == 3
        assert all(isinstance(response, APIResponse) for response in responses)
    
    @pytest.mark.asyncio
    async def test_execute_denoising_requests_failure(self, text_denoiser):
        """Test API request execution with failures."""
        requests = [APIRequest(prompt="test")]
        
        # Mock the client to raise an exception
        text_denoiser.client.complete.side_effect = Exception("API Error")
        
        responses = await text_denoiser._execute_denoising_requests(requests)
        
        assert len(responses) == 1
        assert responses[0].error == "API Error"
    
    def test_parse_denoising_response_success(self, text_denoiser, sample_api_responses, sample_entities):
        """Test successful response parsing."""
        response = sample_api_responses[0]
        original_text = "廟旁住著一家鄉宦，姓甄，名費，字士隱。嫡妻封氏，情性賢淑，深明禮義。"
        entities = sample_entities[0]
        
        denoised_text = text_denoiser._parse_denoising_response(response, original_text, entities)
        
        assert denoised_text is not None
        assert "甄士隱是一家鄉宦" in denoised_text
        assert "封氏" in denoised_text
    
    def test_parse_denoising_response_error(self, text_denoiser, sample_entities):
        """Test response parsing with error."""
        response = APIResponse(
            content="",
            model="gpt-5-mini",
            usage={"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
            finish_reason="error",
            response_time=0.0,
            error="API Error"
        )
        original_text = "test"
        entities = sample_entities[0]
        
        with pytest.raises(ValueError, match="API response error: API Error"):
            text_denoiser._parse_denoising_response(response, original_text, entities)
    
    def test_parse_denoising_response_empty_content(self, text_denoiser, sample_entities):
        """Test response parsing with empty content."""
        response = APIResponse(
            content="",
            model="gpt-5-mini",
            usage={"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
            finish_reason="stop",
            response_time=0.0
        )
        original_text = "test"
        entities = sample_entities[0]
        
        with pytest.raises(ValueError, match="Empty response content"):
            text_denoiser._parse_denoising_response(response, original_text, entities)
    
    def test_parse_denoising_response_validation_failure(self, text_denoiser, sample_entities):
        """Test response parsing with validation failure."""
        response = APIResponse(
            content="很短的文本",
            model="gpt-5-mini",
            usage={"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
            finish_reason="stop",
            response_time=0.0
        )
        original_text = "很長的原始文本"
        entities = sample_entities[0]
        
        # Mock validation to fail
        with patch.object(text_denoiser, '_validate_denoised_text', return_value=False):
            denoised_text = text_denoiser._parse_denoising_response(response, original_text, entities)
            assert denoised_text is None
    
    def test_update_statistics(self, text_denoiser):
        """Test statistics update."""
        original_texts = ["文本1", "文本2", "文本3"]
        denoised_texts = ["去噪文本1", "去噪文本2", ""]  # One failed
        
        text_denoiser._update_statistics(original_texts, denoised_texts)
        
        assert text_denoiser.stats["total_texts_processed"] == 3
        assert text_denoiser.stats["total_texts_denoised"] == 3
        assert text_denoiser.stats["successful_denoising"] == 2
        assert text_denoiser.stats["failed_denoising"] == 1
        assert text_denoiser.stats["average_compression_ratio"] > 0
    
    def test_get_statistics(self, text_denoiser):
        """Test statistics retrieval."""
        # Set some statistics
        text_denoiser.stats["total_texts_processed"] = 10
        text_denoiser.stats["total_texts_denoised"] = 8
        
        stats = text_denoiser.get_statistics()
        
        assert stats["total_texts_processed"] == 10
        assert stats["total_texts_denoised"] == 8
        assert stats is not text_denoiser.stats  # Should return a copy
    
    def test_reset_statistics(self, text_denoiser):
        """Test statistics reset."""
        # Set some statistics
        text_denoiser.stats["total_texts_processed"] = 10
        text_denoiser.stats["total_texts_denoised"] = 8
        
        text_denoiser.reset_statistics()
        
        assert text_denoiser.stats["total_texts_processed"] == 0
        assert text_denoiser.stats["total_texts_denoised"] == 0
        assert text_denoiser.stats["successful_denoising"] == 0
        assert text_denoiser.stats["failed_denoising"] == 0
        assert text_denoiser.stats["cache_hits"] == 0
        assert text_denoiser.stats["cache_misses"] == 0
        assert text_denoiser.stats["average_compression_ratio"] == 0.0


class TestTextDenoiserIntegration:
    """Integration tests for TextDenoiser."""
    
    @pytest.mark.asyncio
    async def test_full_denoising_workflow(self):
        """Test the complete text denoising workflow."""
        # This test would require more complex mocking and setup
        # For now, we'll test the main components work together
        pass
    
    @pytest.mark.asyncio
    async def test_error_handling_and_recovery(self):
        """Test error handling and recovery mechanisms."""
        # This test would verify that the denoiser can handle
        # various error conditions gracefully
        pass
