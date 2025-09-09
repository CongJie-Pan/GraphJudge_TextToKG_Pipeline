"""
Integration Tests for ECTD Pipeline

This module tests the complete integration of the ECTD pipeline components:
- Entity extraction and text denoising workflow
- End-to-end pipeline execution
- Component interaction and data flow
- Error handling across module boundaries
"""

import pytest
import asyncio
import os
import tempfile
import json
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from typing import List

from extractEntity_Phase.core.pipeline_orchestrator import PipelineOrchestrator, PipelineConfig
from extractEntity_Phase.core.entity_extractor import EntityExtractor, ExtractionConfig
from extractEntity_Phase.core.text_denoiser import TextDenoiser, DenoisingConfig
from extractEntity_Phase.api.gpt5mini_client import GPT5MiniClient, GPT5MiniConfig, APIRequest, APIResponse
from extractEntity_Phase.models.entities import Entity, EntityType, EntityList
from extractEntity_Phase.models.pipeline_state import PipelineStage, ProcessingStatus


class TestECTDPipelineIntegration:
    """Test complete ECTD pipeline integration."""
    
    @pytest.fixture
    def sample_texts(self):
        """Sample Chinese texts for testing."""
        return [
            "甄士隱於書房閒坐，至手倦拋書，伏几少憩，不覺朦朧睡去。",
            "這閶門外有個十里街，街內有個仁清巷，巷內有個古廟，因地方窄狹，人皆呼作葫蘆廟。",
            "廟旁住著一家鄉宦，姓甄，名費，字士隱。嫡妻封氏，情性賢淑，深明禮義。"
        ]
    
    @pytest.fixture
    def sample_entity_responses(self):
        """Sample API responses for entity extraction."""
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
    
    @pytest.fixture
    def sample_denoising_responses(self):
        """Sample API responses for text denoising."""
        return [
            APIResponse(
                content="甄士隱於書房閒坐。",
                model="gpt-5-mini",
                usage={"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150},
                finish_reason="stop",
                response_time=1.5,
                cached=False
            ),
            APIResponse(
                content="閶門外有十里街，街內有仁清巷，巷內有古廟，人皆呼作葫蘆廟。",
                model="gpt-5-mini",
                usage={"prompt_tokens": 120, "completion_tokens": 60, "total_tokens": 180},
                finish_reason="stop",
                response_time=1.8,
                cached=False
            ),
            APIResponse(
                content="甄士隱是一家鄉宦。甄士隱的妻子是封氏。",
                model="gpt-5-mini",
                usage={"prompt_tokens": 110, "completion_tokens": 55, "total_tokens": 165},
                finish_reason="stop",
                response_time=1.6,
                cached=False
            )
        ]
    
    @pytest.fixture
    def mock_gpt5mini_client(self):
        """Create mock GPT-5-mini client."""
        client = Mock(spec=GPT5MiniClient)
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
    def pipeline_config(self):
        """Create pipeline configuration for testing."""
        return PipelineConfig(
            batch_size=2,
            max_concurrent=2,
            enable_caching=True,
            enable_rate_limiting=True
        )
    
    @pytest.fixture
    def entity_extractor(self, mock_gpt5mini_client, mock_logger, mock_text_processor):
        """Create EntityExtractor instance with mocked dependencies."""
        with patch('extractEntity_Phase.core.entity_extractor.get_logger', return_value=mock_logger):
            with patch('extractEntity_Phase.core.entity_extractor.ChineseTextProcessor', return_value=mock_text_processor):
                extractor = EntityExtractor(mock_gpt5mini_client)
                return extractor
    
    @pytest.fixture
    def text_denoiser(self, mock_gpt5mini_client, mock_logger, mock_text_processor):
        """Create TextDenoiser instance with mocked dependencies."""
        with patch('extractEntity_Phase.core.text_denoiser.get_logger', return_value=mock_logger):
            with patch('extractEntity_Phase.core.text_denoiser.ChineseTextProcessor', return_value=mock_text_processor):
                denoiser = TextDenoiser(mock_gpt5mini_client)
                return denoiser
    
    @pytest.fixture
    def pipeline_orchestrator(self, pipeline_config, mock_logger, mock_text_processor):
        """Create PipelineOrchestrator instance with mocked dependencies."""
        with patch('extractEntity_Phase.core.pipeline_orchestrator.get_logger', return_value=mock_logger):
            with patch('extractEntity_Phase.core.pipeline_orchestrator.ChineseTextProcessor', return_value=mock_text_processor):
                orchestrator = PipelineOrchestrator(pipeline_config)
                return orchestrator
    
    @pytest.mark.asyncio
    async def test_entity_extraction_integration(self, entity_extractor, sample_texts, sample_entity_responses):
        """Test entity extraction workflow integration."""
        # Mock API responses
        entity_extractor.client.complete.side_effect = sample_entity_responses
        
        # Mock asyncio.get_event_loop().time()
        with patch('asyncio.get_event_loop') as mock_loop:
            mock_loop.return_value.time.return_value = 1234567890.0
            
            # Extract entities
            results = await entity_extractor.extract_entities_from_texts(sample_texts)
        
        # Verify results
        assert len(results) == 3
        assert all(isinstance(result, EntityList) for result in results)
        
        # Check first result
        first_result = results[0]
        assert len(first_result.entities) == 2
        assert first_result.entities[0].text == "甄士隱"
        assert first_result.entities[0].type == EntityType.PERSON
        assert first_result.entities[1].text == "書房"
        assert first_result.entities[1].type == EntityType.LOCATION
        
        # Check statistics
        stats = entity_extractor.get_statistics()
        assert stats["total_texts_processed"] == 3
        assert stats["total_entities_extracted"] == 10
        assert stats["successful_extractions"] == 3
    
    @pytest.mark.asyncio
    async def test_text_denoising_integration(self, text_denoiser, sample_texts, sample_entity_responses, sample_denoising_responses):
        """Test text denoising workflow integration."""
        # First extract entities
        entity_extractor = EntityExtractor(text_denoiser.client)
        entity_extractor.client.complete.side_effect = sample_entity_responses
        
        with patch('asyncio.get_event_loop') as mock_loop:
            mock_loop.return_value.time.return_value = 1234567890.0
            entity_collections = await entity_extractor.extract_entities_from_texts(sample_texts)
        
        # Mock denoising API responses
        text_denoiser.client.complete.side_effect = sample_denoising_responses
        
        # Denoise texts
        denoised_texts = await text_denoiser.denoise_texts(sample_texts, entity_collections)
        
        # Verify results
        assert len(denoised_texts) == 3
        assert all(isinstance(text, str) for text in denoised_texts)
        assert all(len(text) > 0 for text in denoised_texts)
        
        # Check specific denoised texts
        assert "甄士隱於書房閒坐" in denoised_texts[0]
        assert "閶門外有十里街" in denoised_texts[1]
        assert "甄士隱是一家鄉宦" in denoised_texts[2]
        
        # Check statistics
        stats = text_denoiser.get_statistics()
        assert stats["total_texts_processed"] == 3
        assert stats["total_texts_denoised"] == 3
        assert stats["successful_denoising"] == 3
    
    @pytest.mark.asyncio
    async def test_pipeline_orchestrator_integration(self, pipeline_orchestrator, sample_texts, sample_entity_responses, sample_denoising_responses):
        """Test complete pipeline orchestrator integration."""
        # Mock all dependencies
        with patch.object(pipeline_orchestrator, '_initialize_components') as mock_init:
            with patch.object(pipeline_orchestrator, '_load_input_texts', return_value=sample_texts) as mock_load:
                # Mock entity extraction
                with patch.object(pipeline_orchestrator, '_run_entity_extraction_stage') as mock_extract:
                    # Mock text denoising
                    with patch.object(pipeline_orchestrator, '_run_text_denoising_stage') as mock_denoise:
                        # Mock output generation
                        with patch.object(pipeline_orchestrator, '_run_output_generation_stage') as mock_output:
                            # Mock finalization
                            with patch.object(pipeline_orchestrator, '_run_finalization_stage') as mock_finalize:
                                # Run pipeline
                                result = await pipeline_orchestrator.run_pipeline()
                                
                                # Verify pipeline execution
                                assert result is True
                                mock_init.assert_called_once()
                                mock_load.assert_called_once()
                                mock_extract.assert_called_once_with(sample_texts)
                                mock_denoise.assert_called_once_with(sample_texts)
                                mock_output.assert_called_once()
                                mock_finalize.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_end_to_end_pipeline_execution(self, pipeline_orchestrator, sample_texts, sample_entity_responses, sample_denoising_responses):
        """Test end-to-end pipeline execution with real component interaction."""
        # Create temporary output directory
        with tempfile.TemporaryDirectory() as temp_dir:
            pipeline_orchestrator.config.output_dir = temp_dir
            
            # Mock GPT-5-mini client
            mock_client = Mock(spec=GPT5MiniClient)
            mock_client.complete = AsyncMock()
            
            # Set up response sequence for both entity extraction and denoising
            mock_client.complete.side_effect = sample_entity_responses + sample_denoising_responses
            
            # Mock component initialization
            with patch.object(pipeline_orchestrator, '_initialize_components') as mock_init:
                # Mock the actual components
                mock_entity_extractor = Mock(spec=EntityExtractor)
                mock_entity_extractor.extract_entities_from_texts = AsyncMock()
                mock_entity_extractor.stats = {"cache_hits": 2, "cache_misses": 1}
                mock_entity_extractor.reset_statistics = Mock()
                
                mock_text_denoiser = Mock(spec=TextDenoiser)
                mock_text_denoiser.denoise_texts = AsyncMock()
                mock_text_denoiser.stats = {"cache_hits": 1, "cache_misses": 2}
                mock_text_denoiser.reset_statistics = Mock()
                
                # Set up entity extraction results
                entity_collections = [
                    EntityList(
                        entities=[
                            Entity(text="甄士隱", type=EntityType.PERSON),
                            Entity(text="書房", type=EntityType.LOCATION)
                        ]
                    ),
                    EntityList(
                        entities=[
                            Entity(text="閶門", type=EntityType.LOCATION),
                            Entity(text="十里街", type=EntityType.LOCATION),
                            Entity(text="仁清巷", type=EntityType.LOCATION),
                            Entity(text="古廟", type=EntityType.LOCATION),
                            Entity(text="葫蘆廟", type=EntityType.LOCATION)
                        ]
                    ),
                    EntityList(
                        entities=[
                            Entity(text="甄士隱", type=EntityType.PERSON),
                            Entity(text="封氏", type=EntityType.PERSON),
                            Entity(text="鄉宦", type=EntityType.ORGANIZATION)
                        ]
                    )
                ]
                
                mock_entity_extractor.extract_entities_from_texts.return_value = entity_collections
                
                # Set up denoising results
                denoised_texts = [
                    "甄士隱於書房閒坐。",
                    "閶門外有十里街，街內有仁清巷，巷內有古廟，人皆呼作葫蘆廟。",
                    "甄士隱是一家鄉宦。甄士隱的妻子是封氏。"
                ]
                
                mock_text_denoiser.denoise_texts.return_value = denoised_texts
                
                # Mock component creation
                with patch('extractEntity_Phase.core.pipeline_orchestrator.GPT5MiniClient', return_value=mock_client):
                    with patch('extractEntity_Phase.core.pipeline_orchestrator.EntityExtractor', return_value=mock_entity_extractor):
                        with patch('extractEntity_Phase.core.pipeline_orchestrator.TextDenoiser', return_value=mock_text_denoiser):
                            # Run pipeline
                            result = await pipeline_orchestrator.run_pipeline(sample_texts)
                            
                            # Verify pipeline execution
                            assert result is True
                            
                            # Check that components were used
                            mock_entity_extractor.extract_entities_from_texts.assert_called_once()
                            mock_text_denoiser.denoise_texts.assert_called_once()
                            
                            # Check pipeline state
                            pipeline_state = pipeline_orchestrator.get_pipeline_state()
                            assert pipeline_state.status == ProcessingStatus.COMPLETED
                            
                            # Check that output files were created
                            entity_file = os.path.join(temp_dir, "test_entity.txt")
                            denoised_file = os.path.join(temp_dir, "test_denoised.target")
                            stats_file = os.path.join(temp_dir, "pipeline_statistics.json")
                            state_file = os.path.join(temp_dir, "pipeline_state.json")
                            
                            assert os.path.exists(entity_file)
                            assert os.path.exists(denoised_file)
                            assert os.path.exists(stats_file)
                            assert os.path.exists(state_file)
    
    @pytest.mark.asyncio
    async def test_pipeline_error_handling_and_recovery(self, pipeline_orchestrator, sample_texts):
        """Test pipeline error handling and recovery mechanisms."""
        # Create temporary output directory
        with tempfile.TemporaryDirectory() as temp_dir:
            pipeline_orchestrator.config.output_dir = temp_dir
            
            # Mock component initialization to fail
            with patch.object(pipeline_orchestrator, '_initialize_components', side_effect=Exception("Initialization failed")):
                # Run pipeline
                result = await pipeline_orchestrator.run_pipeline(sample_texts)
                
                # Verify pipeline failed
                assert result is False
                
                # Check pipeline state
                pipeline_state = pipeline_orchestrator.get_pipeline_state()
                assert pipeline_state.status == ProcessingStatus.FAILED
                
                # Check that error information was saved
                error_file = os.path.join(temp_dir, "pipeline_error.txt")
                if os.path.exists(error_file):
                    with open(error_file, 'r', encoding='utf-8') as f:
                        error_content = f.read()
                    
                    assert "Initialization failed" in error_content
    
    @pytest.mark.asyncio
    async def test_pipeline_with_partial_failures(self, pipeline_orchestrator, sample_texts):
        """Test pipeline execution with partial stage failures."""
        # Create temporary output directory
        with tempfile.TemporaryDirectory() as temp_dir:
            pipeline_orchestrator.config.output_dir = temp_dir
            
            # Mock dependencies
            with patch.object(pipeline_orchestrator, '_initialize_components'):
                with patch.object(pipeline_orchestrator, '_load_input_texts', return_value=sample_texts):
                    # Mock entity extraction to succeed
                    with patch.object(pipeline_orchestrator, '_run_entity_extraction_stage'):
                        # Mock text denoising to fail
                        with patch.object(pipeline_orchestrator, '_run_text_denoising_stage', side_effect=Exception("Denoising failed")):
                            with patch.object(pipeline_orchestrator, '_handle_pipeline_failure') as mock_handle:
                                # Run pipeline
                                result = await pipeline_orchestrator.run_pipeline()
                                
                                # Verify pipeline failed
                                assert result is False
                                
                                # Check that failure was handled
                                mock_handle.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_pipeline_statistics_and_monitoring(self, pipeline_orchestrator, sample_texts):
        """Test pipeline statistics collection and monitoring."""
        # Mock dependencies
        with patch.object(pipeline_orchestrator, '_initialize_components'):
            with patch.object(pipeline_orchestrator, '_load_input_texts', return_value=sample_texts):
                with patch.object(pipeline_orchestrator, '_run_entity_extraction_stage'):
                    with patch.object(pipeline_orchestrator, '_run_text_denoising_stage'):
                        with patch.object(pipeline_orchestrator, '_run_output_generation_stage'):
                            with patch.object(pipeline_orchestrator, '_run_finalization_stage'):
                                # Run pipeline
                                result = await pipeline_orchestrator.run_pipeline()
                                
                                # Verify pipeline succeeded
                                assert result is True
                                
                                # Check pipeline statistics
                                pipeline_stats = pipeline_orchestrator.get_pipeline_statistics()
                                assert pipeline_stats["total_texts_processed"] == 3
                                assert pipeline_stats["start_time"] is not None
                                assert pipeline_stats["end_time"] is not None
                                assert pipeline_stats["total_duration"] > 0
                                
                                # Check component statistics
                                extraction_stats = pipeline_orchestrator.get_extraction_statistics()
                                denoising_stats = pipeline_orchestrator.get_denoising_statistics()
                                
                                assert extraction_stats is not None
                                assert denoising_stats is not None
    
    @pytest.mark.asyncio
    async def test_pipeline_configuration_variations(self):
        """Test pipeline execution with different configuration variations."""
        # Test with different batch sizes
        configs = [
            PipelineConfig(batch_size=1, max_concurrent=1),
            PipelineConfig(batch_size=5, max_concurrent=2),
            PipelineConfig(batch_size=10, max_concurrent=3)
        ]
        
        for config in configs:
            orchestrator = PipelineOrchestrator(config)
            
            # Verify configuration was applied
            assert orchestrator.config.batch_size == config.batch_size
            assert orchestrator.config.max_concurrent == config.max_concurrent
            
            # Verify default configs were created
            assert orchestrator.config.extraction_config is not None
            assert orchestrator.config.denoising_config is not None
            assert orchestrator.config.gpt5mini_config is not None
    
    @pytest.mark.asyncio
    async def test_pipeline_reset_functionality(self, pipeline_orchestrator, sample_texts):
        """Test pipeline reset functionality."""
        # Set some initial state
        pipeline_orchestrator.extracted_entities = ["entity1", "entity2"]
        pipeline_orchestrator.denoised_texts = ["text1", "text2"]
        pipeline_orchestrator.original_texts = ["original1", "original2"]
        pipeline_orchestrator.pipeline_stats["total_texts_processed"] = 10
        
        # Mock dependencies
        pipeline_orchestrator.entity_extractor = Mock()
        pipeline_orchestrator.text_denoiser = Mock()
        
        # Reset pipeline
        pipeline_orchestrator.reset_pipeline()
        
        # Verify reset
        assert pipeline_orchestrator.extracted_entities == []
        assert pipeline_orchestrator.denoised_texts == []
        assert pipeline_orchestrator.original_texts == []
        assert pipeline_orchestrator.pipeline_stats["total_texts_processed"] == 0
        
        # Verify dependencies were reset
        pipeline_orchestrator.entity_extractor.reset_statistics.assert_called_once()
        pipeline_orchestrator.text_denoiser.reset_statistics.assert_called_once()


class TestECTDPipelinePerformance:
    """Test ECTD pipeline performance characteristics."""
    
    @pytest.mark.asyncio
    async def test_pipeline_batch_processing(self):
        """Test pipeline batch processing performance."""
        # This test would measure and verify batch processing performance
        pass
    
    @pytest.mark.asyncio
    async def test_pipeline_concurrent_processing(self):
        """Test pipeline concurrent processing performance."""
        # This test would measure and verify concurrent processing performance
        pass
    
    @pytest.mark.asyncio
    async def test_pipeline_memory_usage(self):
        """Test pipeline memory usage characteristics."""
        # This test would measure and verify memory usage patterns
        pass


class TestECTDPipelineRobustness:
    """Test ECTD pipeline robustness and error handling."""
    
    @pytest.mark.asyncio
    async def test_pipeline_network_failures(self):
        """Test pipeline behavior under network failures."""
        # This test would verify pipeline behavior when network calls fail
        pass
    
    @pytest.mark.asyncio
    async def test_pipeline_invalid_inputs(self):
        """Test pipeline behavior with invalid inputs."""
        # This test would verify pipeline behavior with malformed or invalid inputs
        pass
    
    @pytest.mark.asyncio
    async def test_pipeline_resource_constraints(self):
        """Test pipeline behavior under resource constraints."""
        # This test would verify pipeline behavior under memory or CPU constraints
        pass
