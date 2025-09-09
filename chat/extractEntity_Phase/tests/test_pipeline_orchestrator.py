"""
Unit tests for Pipeline Orchestrator Module

This module tests the main pipeline coordination functionality including:
- Pipeline initialization and component setup
- Stage execution and progress tracking
- Error handling and recovery
- Output generation and file management
"""

import pytest
import asyncio
import os
import tempfile
import json
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from typing import List
from pathlib import Path

from extractEntity_Phase.core.pipeline_orchestrator import PipelineOrchestrator, PipelineConfig
from extractEntity_Phase.core.entity_extractor import ExtractionConfig
from extractEntity_Phase.core.text_denoiser import DenoisingConfig
from extractEntity_Phase.api.gpt5mini_client import GPT5MiniConfig
from extractEntity_Phase.models.entities import Entity, EntityType, EntityList
from extractEntity_Phase.models.pipeline_state import PipelineStage, ProcessingStatus


class TestPipelineConfig:
    """Test PipelineConfig dataclass."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = PipelineConfig()
        
        assert config.input_file is None
        assert config.output_dir is None
        assert config.iteration == 1
        assert config.extraction_config is not None
        assert config.denoising_config is not None
        assert config.gpt5mini_config is not None
        assert config.batch_size == 10
        assert config.max_concurrent == 3
        assert config.enable_caching is True
        assert config.enable_rate_limiting is True
        assert config.validate_inputs is True
        assert config.validate_outputs is True
        assert config.create_backup is True
    
    def test_custom_config(self):
        """Test custom configuration values."""
        config = PipelineConfig(
            input_file="test.txt",
            output_dir="./test_output",
            iteration=2,
            batch_size=5,
            max_concurrent=2,
            enable_caching=False
        )
        
        assert config.input_file == "test.txt"
        assert config.output_dir == "./test_output"
        assert config.iteration == 2
        assert config.batch_size == 5
        assert config.max_concurrent == 2
        assert config.enable_caching is False
    
    def test_post_init_defaults(self):
        """Test post-init default configuration setting."""
        config = PipelineConfig()
        
        # Check that default configs are created
        assert isinstance(config.extraction_config, ExtractionConfig)
        assert isinstance(config.denoising_config, DenoisingConfig)
        assert isinstance(config.gpt5mini_config, GPT5MiniConfig)


class TestPipelineOrchestrator:
    """Test PipelineOrchestrator class."""
    
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
    def mock_gpt5mini_client(self):
        """Create mock GPT-5-mini client."""
        client = Mock()
        client.complete = AsyncMock()
        return client
    
    @pytest.fixture
    def mock_entity_extractor(self):
        """Create mock entity extractor."""
        extractor = Mock()
        extractor.extract_entities_from_texts = AsyncMock()
        extractor.stats = {"cache_hits": 5, "cache_misses": 3}
        extractor.reset_statistics = Mock()
        return extractor
    
    @pytest.fixture
    def mock_text_denoiser(self):
        """Create mock text denoiser."""
        denoiser = Mock()
        denoiser.denoise_texts = AsyncMock()
        denoiser.stats = {"cache_hits": 4, "cache_misses": 2}
        denoiser.reset_statistics = Mock()
        return denoiser
    
    @pytest.fixture
    def pipeline_orchestrator(self, mock_logger, mock_text_processor):
        """Create PipelineOrchestrator instance with mocked dependencies."""
        with patch('extractEntity_Phase.core.pipeline_orchestrator.get_logger', return_value=mock_logger):
            with patch('extractEntity_Phase.core.pipeline_orchestrator.ChineseTextProcessor', return_value=mock_text_processor):
                orchestrator = PipelineOrchestrator()
                return orchestrator
    
    @pytest.fixture
    def sample_texts(self):
        """Sample Chinese texts for testing."""
        return [
            "甄士隱於書房閒坐，至手倦拋書，伏几少憩，不覺朦朧睡去。",
            "這閶門外有個十里街，街內有個仁清巷，巷內有個古廟，因地方窄狹，人皆呼作葫蘆廟。",
            "廟旁住著一家鄉宦，姓甄，名費，字士隱。嫡妻封氏，情性賢淑，深明禮義。"
        ]
    
    @pytest.fixture
    def sample_entities(self):
        """Sample entity collections for testing."""
        return [
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
    
    @pytest.fixture
    def sample_denoised_texts(self):
        """Sample denoised texts for testing."""
        return [
            "甄士隱於書房閒坐。",
            "閶門外有十里街，街內有仁清巷，巷內有古廟，人皆呼作葫蘆廟。",
            "甄士隱是一家鄉宦。甄士隱的妻子是封氏。"
        ]
    
    def test_initialization(self, pipeline_orchestrator):
        """Test PipelineOrchestrator initialization."""
        assert pipeline_orchestrator.config is not None
        assert pipeline_orchestrator.logger is not None
        assert pipeline_orchestrator.text_processor is not None
        assert pipeline_orchestrator.gpt5mini_client is None
        assert pipeline_orchestrator.entity_extractor is None
        assert pipeline_orchestrator.text_denoiser is None
        assert pipeline_orchestrator.pipeline_state is not None
        assert pipeline_orchestrator.current_stage is None
        assert pipeline_orchestrator.extracted_entities == []
        assert pipeline_orchestrator.denoised_texts == []
        assert pipeline_orchestrator.original_texts == []
        assert pipeline_orchestrator.pipeline_stats["total_texts_processed"] == 0
    
    @pytest.mark.asyncio
    async def test_initialize_components(self, pipeline_orchestrator):
        """Test component initialization."""
        with patch('extractEntity_Phase.core.pipeline_orchestrator.GPT5MiniClient') as mock_client_class:
            with patch('extractEntity_Phase.core.pipeline_orchestrator.EntityExtractor') as mock_extractor_class:
                with patch('extractEntity_Phase.core.pipeline_orchestrator.TextDenoiser') as mock_denoiser_class:
                    mock_client_class.return_value = Mock()
                    mock_extractor_class.return_value = Mock()
                    mock_denoiser_class.return_value = Mock()
                    
                    await pipeline_orchestrator._initialize_components()
                    
                    assert pipeline_orchestrator.gpt5mini_client is not None
                    assert pipeline_orchestrator.entity_extractor is not None
                    assert pipeline_orchestrator.text_denoiser is not None
    
    @pytest.mark.asyncio
    async def test_load_input_texts_from_list(self, pipeline_orchestrator, sample_texts):
        """Test loading input texts from provided list."""
        texts = await pipeline_orchestrator._load_input_texts(sample_texts)
        
        assert texts == sample_texts
        assert len(texts) == 3
    
    @pytest.mark.asyncio
    async def test_load_input_texts_from_file(self, pipeline_orchestrator):
        """Test loading input texts from file."""
        # Create temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False, encoding='utf-8') as f:
            f.write("文本1\n文本2\n文本3\n")
            temp_file = f.name
        
        try:
            pipeline_orchestrator.config.input_file = temp_file
            texts = await pipeline_orchestrator._load_input_texts()
            
            assert texts == ["文本1", "文本2", "文本3"]
            assert len(texts) == 3
        finally:
            os.unlink(temp_file)
    
    @pytest.mark.asyncio
    async def test_load_input_texts_file_not_found(self, pipeline_orchestrator):
        """Test loading input texts from non-existent file."""
        pipeline_orchestrator.config.input_file = "nonexistent.txt"
        
        with pytest.raises(ValueError, match="Input file not found: nonexistent.txt"):
            await pipeline_orchestrator._load_input_texts()
    
    @pytest.mark.asyncio
    async def test_load_input_texts_no_input(self, pipeline_orchestrator):
        """Test loading input texts with no input specified."""
        with pytest.raises(ValueError, match="No input file specified and no texts provided"):
            await pipeline_orchestrator._load_input_texts()
    
    @pytest.mark.asyncio
    async def test_run_entity_extraction_stage_success(self, pipeline_orchestrator, sample_texts, sample_entities):
        """Test successful entity extraction stage."""
        # Mock dependencies
        pipeline_orchestrator.entity_extractor = Mock()
        pipeline_orchestrator.entity_extractor.extract_entities_from_texts = AsyncMock(return_value=sample_entities)
        
        await pipeline_orchestrator._run_entity_extraction_stage(sample_texts)
        
        # Check that entities were extracted
        assert pipeline_orchestrator.extracted_entities == sample_entities
        
        # Check pipeline statistics
        assert pipeline_orchestrator.pipeline_stats["total_entities_extracted"] == 10
        
        # Check stage progress
        stage_progress = pipeline_orchestrator.pipeline_state.get_stage_progress(PipelineStage.ENTITY_EXTRACTION)
        assert stage_progress is not None
        assert stage_progress.status == ProcessingStatus.COMPLETED
        assert stage_progress.progress_percentage == 100.0
    
    @pytest.mark.asyncio
    async def test_run_entity_extraction_stage_failure(self, pipeline_orchestrator, sample_texts):
        """Test entity extraction stage failure."""
        # Mock dependencies
        pipeline_orchestrator.entity_extractor = Mock()
        pipeline_orchestrator.entity_extractor.extract_entities_from_texts = AsyncMock(side_effect=Exception("Extraction failed"))
        
        with pytest.raises(Exception, match="Extraction failed"):
            await pipeline_orchestrator._run_entity_extraction_stage(sample_texts)
        
        # Check stage progress
        stage_progress = pipeline_orchestrator.pipeline_state.get_stage_progress(PipelineStage.ENTITY_EXTRACTION)
        assert stage_progress is not None
        assert stage_progress.status == ProcessingStatus.FAILED
        assert len(stage_progress.errors) == 1
    
    @pytest.mark.asyncio
    async def test_run_text_denoising_stage_success(self, pipeline_orchestrator, sample_texts, sample_entities, sample_denoised_texts):
        """Test successful text denoising stage."""
        # Set up extracted entities
        pipeline_orchestrator.extracted_entities = sample_entities
        
        # Mock dependencies
        pipeline_orchestrator.text_denoiser = Mock()
        pipeline_orchestrator.text_denoiser.denoise_texts = AsyncMock(return_value=sample_denoised_texts)
        
        await pipeline_orchestrator._run_text_denoising_stage(sample_texts)
        
        # Check that texts were denoised
        assert pipeline_orchestrator.denoised_texts == sample_denoised_texts
        
        # Check pipeline statistics
        assert pipeline_orchestrator.pipeline_stats["total_texts_denoised"] == 3
        
        # Check stage progress
        stage_progress = pipeline_orchestrator.pipeline_state.get_stage_progress(PipelineStage.TEXT_DENOISING)
        assert stage_progress is not None
        assert stage_progress.status == ProcessingStatus.COMPLETED
        assert stage_progress.progress_percentage == 100.0
    
    @pytest.mark.asyncio
    async def test_run_text_denoising_stage_failure(self, pipeline_orchestrator, sample_texts, sample_entities):
        """Test text denoising stage failure."""
        # Set up extracted entities
        pipeline_orchestrator.extracted_entities = sample_entities
        
        # Mock dependencies
        pipeline_orchestrator.text_denoiser = Mock()
        pipeline_orchestrator.text_denoiser.denoise_texts = AsyncMock(side_effect=Exception("Denoising failed"))
        
        with pytest.raises(Exception, match="Denoising failed"):
            await pipeline_orchestrator._run_text_denoising_stage(sample_texts)
        
        # Check stage progress
        stage_progress = pipeline_orchestrator.pipeline_state.get_stage_progress(PipelineStage.TEXT_DENOISING)
        assert stage_progress is not None
        assert stage_progress.status == ProcessingStatus.FAILED
        assert len(stage_progress.errors) == 1
    
    @pytest.mark.asyncio
    async def test_run_output_generation_stage_success(self, pipeline_orchestrator, sample_entities, sample_denoised_texts):
        """Test successful output generation stage."""
        # Set up data
        pipeline_orchestrator.extracted_entities = sample_entities
        pipeline_orchestrator.denoised_texts = sample_denoised_texts
        
        with tempfile.TemporaryDirectory() as temp_dir:
            pipeline_orchestrator.config.output_dir = temp_dir
            
            await pipeline_orchestrator._run_output_generation_stage()
            
            # Check that files were created
            entity_file = os.path.join(temp_dir, "test_entity.txt")
            denoised_file = os.path.join(temp_dir, "test_denoised.target")
            stats_file = os.path.join(temp_dir, "pipeline_statistics.json")
            state_file = os.path.join(temp_dir, "pipeline_state.json")
            
            assert os.path.exists(entity_file)
            assert os.path.exists(denoised_file)
            assert os.path.exists(stats_file)
            assert os.path.exists(state_file)
            
            # Check stage progress
            stage_progress = pipeline_orchestrator.pipeline_state.get_stage_progress(PipelineStage.OUTPUT_GENERATION)
            assert stage_progress is not None
            assert stage_progress.status == ProcessingStatus.COMPLETED
    
    @pytest.mark.asyncio
    async def test_run_output_generation_stage_failure(self, pipeline_orchestrator):
        """Test output generation stage failure."""
        # Mock file operations to fail
        with patch.object(pipeline_orchestrator, '_save_entities', side_effect=Exception("Save failed")):
            with pytest.raises(Exception, match="Save failed"):
                await pipeline_orchestrator._run_output_generation_stage()
            
            # Check stage progress
            stage_progress = pipeline_orchestrator.pipeline_state.get_stage_progress(PipelineStage.OUTPUT_GENERATION)
            assert stage_progress is not None
            assert stage_progress.status == ProcessingStatus.FAILED
    
    @pytest.mark.asyncio
    async def test_run_finalization_stage_success(self, pipeline_orchestrator):
        """Test successful pipeline finalization stage."""
        # Mock dependencies
        pipeline_orchestrator.entity_extractor = Mock()
        pipeline_orchestrator.entity_extractor.stats = {"cache_hits": 5, "cache_misses": 3}
        pipeline_orchestrator.text_denoiser = Mock()
        pipeline_orchestrator.text_denoiser.stats = {"cache_hits": 4, "cache_misses": 2}
        
        await pipeline_orchestrator._run_finalization_stage()
        
        # Check that pipeline was marked as completed
        assert pipeline_orchestrator.pipeline_state.status == ProcessingStatus.COMPLETED
        
        # Check stage progress
        stage_progress = pipeline_orchestrator.pipeline_state.get_stage_progress(PipelineStage.CLEANUP)
        assert stage_progress is not None
        assert stage_progress.status == ProcessingStatus.COMPLETED
        
        # Check statistics
        assert pipeline_orchestrator.pipeline_stats["success_rate"] > 0
        assert pipeline_orchestrator.pipeline_stats["cache_hit_rate"] > 0
    
    def test_create_output_directory_custom(self, pipeline_orchestrator):
        """Test creating custom output directory."""
        pipeline_orchestrator.config.output_dir = "./custom_output"
        
        output_dir = pipeline_orchestrator._create_output_directory()
        
        assert output_dir == "./custom_output"
        assert os.path.exists(output_dir)
        
        # Cleanup
        os.rmdir(output_dir)
    
    def test_create_output_directory_default(self, pipeline_orchestrator):
        """Test creating default output directory."""
        # Set environment variable
        os.environ['PIPELINE_OUTPUT_DIR'] = './default_output'
        
        output_dir = pipeline_orchestrator._create_output_directory()
        
        assert output_dir == "./default_output/Graph_Iteration1"
        assert os.path.exists(output_dir)
        
        # Cleanup
        os.rmdir(output_dir)
        os.rmdir("./default_output")
        del os.environ['PIPELINE_OUTPUT_DIR']
    
    @pytest.mark.asyncio
    async def test_save_entities(self, pipeline_orchestrator, sample_entities):
        """Test saving entities to file."""
        pipeline_orchestrator.extracted_entities = sample_entities
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False, encoding='utf-8') as f:
            temp_file = f.name
        
        try:
            await pipeline_orchestrator._save_entities(temp_file)
            
            # Check file contents
            with open(temp_file, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            assert len(lines) == 3
            assert '["甄士隱", "書房"]' in lines[0]
            assert '["閶門", "十里街", "仁清巷", "古廟", "葫蘆廟"]' in lines[1]
            assert '["甄士隱", "封氏", "鄉宦"]' in lines[2]
        finally:
            os.unlink(temp_file)
    
    @pytest.mark.asyncio
    async def test_save_denoised_texts(self, pipeline_orchestrator, sample_denoised_texts):
        """Test saving denoised texts to file."""
        pipeline_orchestrator.denoised_texts = sample_denoised_texts
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False, encoding='utf-8') as f:
            temp_file = f.name
        
        try:
            await pipeline_orchestrator._save_denoised_texts(temp_file)
            
            # Check file contents
            with open(temp_file, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            assert len(lines) == 3
            assert "甄士隱於書房閒坐。" in lines[0]
            assert "閶門外有十里街，街內有仁清巷，巷內有古廟，人皆呼作葫蘆廟。" in lines[1]
            assert "甄士隱是一家鄉宦。甄士隱的妻子是封氏。" in lines[2]
        finally:
            os.unlink(temp_file)
    
    @pytest.mark.asyncio
    async def test_save_pipeline_statistics(self, pipeline_orchestrator):
        """Test saving pipeline statistics to file."""
        # Set some statistics
        pipeline_orchestrator.pipeline_stats["total_texts_processed"] = 10
        pipeline_orchestrator.pipeline_stats["total_entities_extracted"] = 25
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False, encoding='utf-8') as f:
            temp_file = f.name
        
        try:
            await pipeline_orchestrator._save_pipeline_statistics(temp_file)
            
            # Check file contents
            with open(temp_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            assert data["total_texts_processed"] == 10
            assert data["total_entities_extracted"] == 25
        finally:
            os.unlink(temp_file)
    
    @pytest.mark.asyncio
    async def test_save_pipeline_state(self, pipeline_orchestrator):
        """Test saving pipeline state to file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False, encoding='utf-8') as f:
            temp_file = f.name
        
        try:
            await pipeline_orchestrator._save_pipeline_state(temp_file)
            
            # Check file contents
            with open(temp_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            assert "status" in data
            assert "stage_progress" in data
        finally:
            os.unlink(temp_file)
    
    def test_calculate_final_statistics(self, pipeline_orchestrator):
        """Test final statistics calculation."""
        # Mock dependencies
        pipeline_orchestrator.entity_extractor = Mock()
        pipeline_orchestrator.entity_extractor.stats = {"cache_hits": 5, "cache_misses": 3}
        pipeline_orchestrator.text_denoiser = Mock()
        pipeline_orchestrator.text_denoiser.stats = {"cache_hits": 4, "cache_misses": 2}
        
        pipeline_orchestrator._calculate_final_statistics()
        
        # Check that statistics were calculated
        assert pipeline_orchestrator.pipeline_stats["success_rate"] >= 0
        assert pipeline_orchestrator.pipeline_stats["cache_hit_rate"] > 0
    
    def test_log_pipeline_summary(self, pipeline_orchestrator):
        """Test pipeline summary logging."""
        # Set some statistics
        pipeline_orchestrator.pipeline_stats["total_texts_processed"] = 10
        pipeline_orchestrator.pipeline_stats["total_entities_extracted"] = 25
        pipeline_orchestrator.pipeline_stats["total_texts_denoised"] = 8
        pipeline_orchestrator.pipeline_stats["success_rate"] = 0.8
        pipeline_orchestrator.pipeline_stats["cache_hit_rate"] = 0.75
        pipeline_orchestrator.pipeline_stats["total_duration"] = 120.5
        
        # Mock logger
        mock_logger = Mock()
        pipeline_orchestrator.logger = mock_logger
        
        pipeline_orchestrator._log_pipeline_summary()
        
        # Check that summary was logged
        assert mock_logger.log.call_count >= 6  # At least 6 log calls for summary
    
    @pytest.mark.asyncio
    async def test_handle_pipeline_failure(self, pipeline_orchestrator, sample_entities, sample_denoised_texts):
        """Test pipeline failure handling."""
        # Set up data
        pipeline_orchestrator.extracted_entities = sample_entities
        pipeline_orchestrator.denoised_texts = sample_denoised_texts
        
        error = Exception("Pipeline failed")
        
        with tempfile.TemporaryDirectory() as temp_dir:
            pipeline_orchestrator.config.output_dir = temp_dir
            
            await pipeline_orchestrator._handle_pipeline_failure(error)
            
            # Check that partial results were saved
            partial_entities_file = os.path.join(temp_dir, "partial_entities.txt")
            partial_denoised_file = os.path.join(temp_dir, "partial_denoised.txt")
            error_file = os.path.join(temp_dir, "pipeline_error.txt")
            
            assert os.path.exists(partial_entities_file)
            assert os.path.exists(partial_denoised_file)
            assert os.path.exists(error_file)
            
            # Check error file contents
            with open(error_file, 'r', encoding='utf-8') as f:
                error_content = f.read()
            
            assert "Pipeline failed at:" in error_content
            assert "Pipeline failed" in error_content
    
    @pytest.mark.asyncio
    async def test_run_pipeline_success(self, pipeline_orchestrator, sample_texts):
        """Test successful pipeline execution."""
        # Mock all dependencies
        with patch.object(pipeline_orchestrator, '_initialize_components') as mock_init:
            with patch.object(pipeline_orchestrator, '_load_input_texts', return_value=sample_texts) as mock_load:
                with patch.object(pipeline_orchestrator, '_run_entity_extraction_stage') as mock_extract:
                    with patch.object(pipeline_orchestrator, '_run_text_denoising_stage') as mock_denoise:
                        with patch.object(pipeline_orchestrator, '_run_output_generation_stage') as mock_output:
                            with patch.object(pipeline_orchestrator, '_run_finalization_stage') as mock_finalize:
                                result = await pipeline_orchestrator.run_pipeline()
                                
                                assert result is True
                                mock_init.assert_called_once()
                                mock_load.assert_called_once()
                                mock_extract.assert_called_once_with(sample_texts)
                                mock_denoise.assert_called_once_with(sample_texts)
                                mock_output.assert_called_once()
                                mock_finalize.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_run_pipeline_failure(self, pipeline_orchestrator, sample_texts):
        """Test pipeline execution failure."""
        # Mock dependencies to fail
        with patch.object(pipeline_orchestrator, '_initialize_components') as mock_init:
            with patch.object(pipeline_orchestrator, '_load_input_texts', return_value=sample_texts) as mock_load:
                with patch.object(pipeline_orchestrator, '_run_entity_extraction_stage', side_effect=Exception("Stage failed")) as mock_extract:
                    with patch.object(pipeline_orchestrator, '_handle_pipeline_failure') as mock_handle:
                        result = await pipeline_orchestrator.run_pipeline()
                        
                        assert result is False
                        mock_init.assert_called_once()
                        mock_load.assert_called_once()
                        mock_extract.assert_called_once_with(sample_texts)
                        mock_handle.assert_called_once()
    
    def test_get_pipeline_state(self, pipeline_orchestrator):
        """Test getting pipeline state."""
        state = pipeline_orchestrator.get_pipeline_state()
        assert state is pipeline_orchestrator.pipeline_state
    
    def test_get_pipeline_statistics(self, pipeline_orchestrator):
        """Test getting pipeline statistics."""
        # Set some statistics
        pipeline_orchestrator.pipeline_stats["total_texts_processed"] = 10
        
        stats = pipeline_orchestrator.get_pipeline_statistics()
        
        assert stats["total_texts_processed"] == 10
        assert stats is not pipeline_orchestrator.pipeline_stats  # Should return a copy
    
    def test_get_extraction_statistics(self, pipeline_orchestrator):
        """Test getting extraction statistics."""
        # Mock entity extractor
        pipeline_orchestrator.entity_extractor = Mock()
        pipeline_orchestrator.entity_extractor.get_statistics.return_value = {"test": "value"}
        
        stats = pipeline_orchestrator.get_extraction_statistics()
        
        assert stats == {"test": "value"}
        pipeline_orchestrator.entity_extractor.get_statistics.assert_called_once()
    
    def test_get_denoising_statistics(self, pipeline_orchestrator):
        """Test getting denoising statistics."""
        # Mock text denoiser
        pipeline_orchestrator.text_denoiser = Mock()
        pipeline_orchestrator.text_denoiser.get_statistics.return_value = {"test": "value"}
        
        stats = pipeline_orchestrator.get_denoising_statistics()
        
        assert stats == {"test": "value"}
        pipeline_orchestrator.text_denoiser.get_statistics.assert_called_once()
    
    def test_reset_pipeline(self, pipeline_orchestrator):
        """Test pipeline reset."""
        # Set some data
        pipeline_orchestrator.extracted_entities = ["entity1", "entity2"]
        pipeline_orchestrator.denoised_texts = ["text1", "text2"]
        pipeline_orchestrator.original_texts = ["original1", "original2"]
        pipeline_orchestrator.pipeline_stats["total_texts_processed"] = 10
        
        # Mock dependencies
        pipeline_orchestrator.entity_extractor = Mock()
        pipeline_orchestrator.text_denoiser = Mock()
        
        pipeline_orchestrator.reset_pipeline()
        
        # Check that everything was reset
        assert pipeline_orchestrator.extracted_entities == []
        assert pipeline_orchestrator.denoised_texts == []
        assert pipeline_orchestrator.original_texts == []
        assert pipeline_orchestrator.pipeline_stats["total_texts_processed"] == 0
        
        # Check that dependencies were reset
        pipeline_orchestrator.entity_extractor.reset_statistics.assert_called_once()
        pipeline_orchestrator.text_denoiser.reset_statistics.assert_called_once()


class TestPipelineOrchestratorIntegration:
    """Integration tests for PipelineOrchestrator."""
    
    @pytest.mark.asyncio
    async def test_full_pipeline_workflow(self):
        """Test the complete pipeline workflow."""
        # This test would require more complex mocking and setup
        # For now, we'll test the main components work together
        pass
    
    @pytest.mark.asyncio
    async def test_error_recovery_and_continuation(self):
        """Test error recovery and pipeline continuation."""
        # This test would verify that the orchestrator can handle
        # various error conditions and continue processing
        pass
