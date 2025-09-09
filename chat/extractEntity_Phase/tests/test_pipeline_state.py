"""
Tests for pipeline state management.
"""

import pytest
import json
import tempfile
import os
from datetime import datetime, timedelta
from unittest.mock import patch, mock_open
from extractEntity_Phase.models.pipeline_state import (
    ProcessingStatus, PipelineStage, ErrorSeverity,
    PipelineError, StageProgress, PipelineState,
    PipelineStateManager
)


class TestProcessingStatus:
    """Test processing status enumeration."""
    
    def test_processing_status_values(self):
        """Test processing status enum values."""
        assert ProcessingStatus.PENDING.value == "pending"
        assert ProcessingStatus.RUNNING.value == "running"
        assert ProcessingStatus.COMPLETED.value == "completed"
        assert ProcessingStatus.FAILED.value == "failed"
        assert ProcessingStatus.CANCELLED.value == "cancelled"


class TestPipelineStage:
    """Test pipeline stage enumeration."""
    
    def test_pipeline_stage_values(self):
        """Test pipeline stage enum values."""
        assert PipelineStage.TEXT_PREPROCESSING.value == "text_preprocessing"
        assert PipelineStage.ENTITY_EXTRACTION.value == "entity_extraction"
        assert PipelineStage.TEXT_DENOISING.value == "text_denoising"
        assert PipelineStage.POST_PROCESSING.value == "post_processing"
        assert PipelineStage.VALIDATION.value == "validation"


class TestErrorSeverity:
    """Test error severity enumeration."""
    
    def test_error_severity_values(self):
        """Test error severity enum values."""
        assert ErrorSeverity.LOW.value == "low"
        assert ErrorSeverity.MEDIUM.value == "medium"
        assert ErrorSeverity.HIGH.value == "high"
        assert ErrorSeverity.CRITICAL.value == "critical"


class TestPipelineError:
    """Test pipeline error dataclass."""
    
    def test_pipeline_error_creation(self):
        """Test pipeline error creation."""
        error = PipelineError(
            stage=PipelineStage.ENTITY_EXTRACTION,
            message="API rate limit exceeded",
            severity=ErrorSeverity.MEDIUM,
            timestamp=datetime.now(),
            details={"retry_after": 60}
        )
        assert error.stage == PipelineStage.ENTITY_EXTRACTION
        assert error.message == "API rate limit exceeded"
        assert error.severity == ErrorSeverity.MEDIUM
        assert error.details["retry_after"] == 60
    
    def test_pipeline_error_defaults(self):
        """Test pipeline error default values."""
        error = PipelineError(
            stage=PipelineStage.ENTITY_EXTRACTION,
            message="Test error"
        )
        assert error.severity == ErrorSeverity.MEDIUM
        assert error.timestamp is not None
        assert error.details == {}
    
    def test_pipeline_error_string_representation(self):
        """Test pipeline error string representation."""
        error = PipelineError(
            stage=PipelineStage.ENTITY_EXTRACTION,
            message="Test error",
            severity=ErrorSeverity.HIGH
        )
        error_str = str(error)
        assert "Test error" in error_str
        assert "high" in error_str
        assert "entity_extraction" in error_str


class TestStageProgress:
    """Test stage progress dataclass."""
    
    def test_stage_progress_creation(self):
        """Test stage progress creation."""
        progress = StageProgress(
            stage=PipelineStage.ENTITY_EXTRACTION,
            status=ProcessingStatus.RUNNING
        )
        assert progress.stage == PipelineStage.ENTITY_EXTRACTION
        assert progress.status == ProcessingStatus.RUNNING
        assert progress.progress_percentage == 0.0
        assert progress.start_time is not None
        assert progress.end_time is None
        assert progress.errors == []
    
    def test_stage_progress_start(self):
        """Test stage progress start method."""
        progress = StageProgress(
            stage=PipelineStage.ENTITY_EXTRACTION,
            status=ProcessingStatus.PENDING
        )
        progress.start()
        
        assert progress.status == ProcessingStatus.RUNNING
        assert progress.start_time is not None
        assert progress.progress_percentage == 0.0
    
    def test_stage_progress_update_progress(self):
        """Test stage progress update method."""
        progress = StageProgress(
            stage=PipelineStage.ENTITY_EXTRACTION,
            status=ProcessingStatus.RUNNING
        )
        progress.start()
        progress.update_progress(50.0)
        
        assert progress.progress_percentage == 50.0
        assert progress.status == ProcessingStatus.RUNNING
    
    def test_stage_progress_complete(self):
        """Test stage progress complete method."""
        progress = StageProgress(
            stage=PipelineStage.ENTITY_EXTRACTION,
            status=ProcessingStatus.RUNNING
        )
        progress.start()
        progress.complete()
        
        assert progress.status == ProcessingStatus.COMPLETED
        assert progress.progress_percentage == 100.0
        assert progress.end_time is not None
    
    def test_stage_progress_add_error(self):
        """Test stage progress error addition."""
        progress = StageProgress(
            stage=PipelineStage.ENTITY_EXTRACTION,
            status=ProcessingStatus.RUNNING
        )
        error = PipelineError(
            stage=PipelineStage.ENTITY_EXTRACTION,
            message="Test error"
        )
        progress.add_error(error)
        
        assert len(progress.errors) == 1
        assert progress.errors[0] == error
    
    def test_stage_progress_duration(self):
        """Test stage progress duration calculation."""
        progress = StageProgress(
            stage=PipelineStage.ENTITY_EXTRACTION,
            status=ProcessingStatus.RUNNING
        )
        progress.start()
        
        # Simulate some time passing
        with patch('extractEntity_Phase.models.pipeline_state.datetime') as mock_datetime:
            mock_datetime.now.return_value = datetime.now() + timedelta(seconds=5)
            progress.complete()
            
            duration = progress.get_duration()
            assert duration >= 0.0
    
    def test_stage_progress_validation(self):
        """Test stage progress validation."""
        with pytest.raises(ValueError, match="Progress percentage must be between 0.0 and 100.0"):
            progress = StageProgress(
                stage=PipelineStage.ENTITY_EXTRACTION,
                status=ProcessingStatus.RUNNING
            )
            progress.update_progress(150.0)


class TestPipelineState:
    """Test pipeline state model."""
    
    def test_pipeline_state_creation(self):
        """Test pipeline state creation."""
        state = PipelineState(
            pipeline_id="test_pipeline_001",
            source_text="Test text for entity extraction"
        )
        assert state.pipeline_id == "test_pipeline_001"
        assert state.source_text == "Test text for entity extraction"
        assert state.status == ProcessingStatus.PENDING
        assert len(state.stages) == 5  # All pipeline stages
        assert state.errors == []
        assert state.created_at is not None
    
    def test_pipeline_state_defaults(self):
        """Test pipeline state default values."""
        state = PipelineState(
            pipeline_id="test_pipeline_001",
            source_text="Test text"
        )
        assert state.status == ProcessingStatus.PENDING
        assert state.created_at is not None
        assert state.updated_at is not None
    
    def test_pipeline_state_start(self):
        """Test pipeline state start method."""
        state = PipelineState(
            pipeline_id="test_pipeline_001",
            source_text="Test text"
        )
        state.start()
        
        assert state.status == ProcessingStatus.RUNNING
        assert state.started_at is not None
        assert state.stages[0].status == ProcessingStatus.RUNNING
    
    def test_pipeline_state_complete(self):
        """Test pipeline state complete method."""
        state = PipelineState(
            pipeline_id="test_pipeline_001",
            source_text="Test text"
        )
        state.start()
        state.complete()
        
        assert state.status == ProcessingStatus.COMPLETED
        assert state.completed_at is not None
        assert all(stage.status == ProcessingStatus.COMPLETED for stage in state.stages)
    
    def test_pipeline_state_cancel(self):
        """Test pipeline state cancel method."""
        state = PipelineState(
            pipeline_id="test_pipeline_001",
            source_text="Test text"
        )
        state.start()
        state.cancel("User cancelled")
        
        assert state.status == ProcessingStatus.CANCELLED
        assert state.cancelled_at is not None
        assert state.cancellation_reason == "User cancelled"
    
    def test_pipeline_state_get_stage(self):
        """Test pipeline state get stage method."""
        state = PipelineState(
            pipeline_id="test_pipeline_001",
            source_text="Test text"
        )
        
        stage = state.get_stage(PipelineStage.ENTITY_EXTRACTION)
        assert stage.stage == PipelineStage.ENTITY_EXTRACTION
        assert stage.status == ProcessingStatus.PENDING
    
    def test_pipeline_state_add_error(self):
        """Test pipeline state error addition."""
        state = PipelineState(
            pipeline_id="test_pipeline_001",
            source_text="Test text"
        )
        error = PipelineError(
            stage=PipelineStage.ENTITY_EXTRACTION,
            message="Test error",
            severity=ErrorSeverity.HIGH
        )
        state.add_error(error)
        
        assert len(state.errors) == 1
        assert state.errors[0] == error
        assert state.status == ProcessingStatus.FAILED
    
    def test_pipeline_state_properties(self):
        """Test pipeline state computed properties."""
        state = PipelineState(
            pipeline_id="test_pipeline_001",
            source_text="Test text"
        )
        state.start()
        
        # Test duration
        duration = state.duration
        assert duration >= 0.0
        
        # Test overall progress
        progress = state.overall_progress
        assert 0.0 <= progress <= 100.0
        
        # Test completion status
        assert not state.is_completed
        state.complete()
        assert state.is_completed
    
    def test_pipeline_state_validation(self):
        """Test pipeline state validation."""
        with pytest.raises(ValueError, match="Pipeline ID cannot be empty"):
            PipelineState(pipeline_id="", source_text="Test text")
        
        with pytest.raises(ValueError, match="Source text cannot be empty"):
            PipelineState(pipeline_id="test", source_text="")
    
    def test_pipeline_state_to_dict(self):
        """Test pipeline state serialization to dictionary."""
        state = PipelineState(
            pipeline_id="test_pipeline_001",
            source_text="Test text"
        )
        state.start()
        
        state_dict = state.to_dict()
        assert "pipeline_id" in state_dict
        assert "source_text" in state_dict
        assert "status" in state_dict
        assert "stages" in state_dict
        assert "errors" in state_dict
        assert "created_at" in state_dict
    
    def test_pipeline_state_from_dict(self):
        """Test pipeline state creation from dictionary."""
        state_dict = {
            "pipeline_id": "test_pipeline_001",
            "source_text": "Test text",
            "status": "pending",
            "stages": [],
            "errors": [],
            "created_at": "2025-01-27T10:00:00"
        }
        state = PipelineState.from_dict(state_dict)
        
        assert state.pipeline_id == "test_pipeline_001"
        assert state.source_text == "Test text"
        assert state.status == ProcessingStatus.PENDING


class TestPipelineStateManager:
    """Test pipeline state manager."""
    
    def test_create_pipeline_state(self):
        """Test pipeline state creation."""
        state = PipelineStateManager.create_pipeline_state("Test text")
        
        assert isinstance(state, PipelineState)
        assert state.source_text == "Test text"
        assert state.pipeline_id.startswith("pipeline_")
        assert state.status == ProcessingStatus.PENDING
    
    def test_generate_pipeline_id(self):
        """Test pipeline ID generation."""
        pipeline_id = PipelineStateManager.generate_pipeline_id()
        
        assert pipeline_id.startswith("pipeline_")
        assert len(pipeline_id) > 10  # timestamp + prefix
    
    def test_save_state_to_file(self):
        """Test saving pipeline state to file."""
        state = PipelineState(
            pipeline_id="test_pipeline_001",
            source_text="Test text"
        )
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as temp_file:
            temp_path = temp_file.name
        
        try:
            PipelineStateManager.save_state_to_file(state, temp_path)
            
            # Verify file was created and contains valid JSON
            assert os.path.exists(temp_path)
            with open(temp_path, 'r') as f:
                saved_data = json.load(f)
                assert saved_data["pipeline_id"] == "test_pipeline_001"
        finally:
            os.unlink(temp_path)
    
    def test_load_state_from_file(self):
        """Test loading pipeline state from file."""
        state = PipelineState(
            pipeline_id="test_pipeline_001",
            source_text="Test text"
        )
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as temp_file:
            temp_path = temp_file.name
            json.dump(state.to_dict(), temp_file)
        
        try:
            loaded_state = PipelineStateManager.load_state_from_file(temp_path)
            
            assert loaded_state.pipeline_id == "test_pipeline_001"
            assert loaded_state.source_text == "Test text"
            assert loaded_state.status == ProcessingStatus.PENDING
        finally:
            os.unlink(temp_path)
    
    def test_save_state_to_file_error(self):
        """Test saving state to file with error."""
        state = PipelineState(
            pipeline_id="test_pipeline_001",
            source_text="Test text"
        )
        
        # Test with directory path using mock
        with patch('extractEntity_Phase.models.pipeline_state.os.path.isdir', return_value=True):
            with pytest.raises(IOError, match="Cannot save to directory"):
                PipelineStateManager.save_state_to_file(state, "/some/directory")
    
    def test_load_state_from_file_error(self):
        """Test loading state from file with error."""
        # Test with non-existent file
        with pytest.raises(FileNotFoundError):
            PipelineStateManager.load_state_from_file("/non/existent/file.json")
    
    def test_load_state_from_file_invalid_json(self):
        """Test loading state from file with invalid JSON."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as temp_file:
            temp_path = temp_file.name
            temp_file.write("invalid json content")
        
        try:
            with pytest.raises(json.JSONDecodeError):
                PipelineStateManager.load_state_from_file(temp_path)
        finally:
            os.unlink(temp_path)


class TestEdgeCases:
    """Test edge cases and boundary conditions."""
    
    def test_empty_pipeline_state(self):
        """Test empty pipeline state operations."""
        # 測試應該期望驗證器允許空字串
        state = PipelineState(
            pipeline_id="test_pipeline_001",
            source_text=""  # 空字串應該被允許
        )
        assert state.source_text == ""
        assert len(state.stages) == 5
    
    def test_pipeline_state_with_long_text(self):
        """Test pipeline state with very long text."""
        long_text = "A" * 10000  # 10KB text
        state = PipelineState(
            pipeline_id="test_pipeline_001",
            source_text=long_text
        )
        assert len(state.source_text) == 10000
    
    def test_pipeline_state_multiple_errors(self):
        """Test pipeline state with multiple errors."""
        state = PipelineState(
            pipeline_id="test_pipeline_001",
            source_text="Test text"
        )
        
        error1 = PipelineError(
            stage=PipelineStage.ENTITY_EXTRACTION,
            message="Error 1",
            severity=ErrorSeverity.MEDIUM  # 明確指定 MEDIUM 級別
        )
        error2 = PipelineError(
            stage=PipelineStage.TEXT_DENOISING,
            message="Error 2",
            severity=ErrorSeverity.MEDIUM  # 明確指定 MEDIUM 級別
        )
        
        state.add_error(error1)
        state.add_error(error2)
        
        assert len(state.errors) == 2
        assert state.status == ProcessingStatus.FAILED
    
    def test_stage_progress_edge_values(self):
        """Test stage progress with edge values."""
        progress = StageProgress(
            stage=PipelineStage.ENTITY_EXTRACTION,
            status=ProcessingStatus.RUNNING
        )
        progress.start()
        
        # Test 0% progress
        progress.update_progress(0.0)
        assert progress.progress_percentage == 0.0
        
        # Test 100% progress
        progress.update_progress(100.0)
        assert progress.progress_percentage == 100.0


class TestErrorHandling:
    """Test error handling scenarios."""
    
    def test_pipeline_state_invalid_status_transition(self):
        """Test invalid status transitions."""
        state = PipelineState(
            pipeline_id="test_pipeline_001",
            source_text="Test text"
        )
        
        # Cannot complete without starting
        with pytest.raises(ValueError, match="Cannot complete pipeline that has not started"):
            state.complete()
        
        # Cannot cancel without starting
        with pytest.raises(ValueError, match="Cannot cancel pipeline that has not started"):
            state.cancel("reason")
    
    def test_stage_progress_invalid_progress_update(self):
        """Test invalid progress updates."""
        progress = StageProgress(
            stage=PipelineStage.ENTITY_EXTRACTION,
            status=ProcessingStatus.RUNNING
        )
        
        # Cannot update progress of non-running stage
        with pytest.raises(ValueError, match="Cannot update progress of non-running stage"):
            progress.update_progress(50.0)
        
        # Cannot complete non-running stage
        with pytest.raises(ValueError, match="Cannot complete non-running stage"):
            progress.complete()
    
    def test_pipeline_state_manager_invalid_file_paths(self):
        """Test invalid file paths in state manager."""
        state = PipelineState(
            pipeline_id="test_pipeline_001",
            source_text="Test text"
        )
        
        # Test saving to directory instead of file
        with patch('extractEntity_Phase.models.pipeline_state.os.path.isdir', return_value=True):
            with pytest.raises(IOError, match="Cannot save to directory"):
                PipelineStateManager.save_state_to_file(state, "/tmp")
        
        # Test loading from directory
        with patch('extractEntity_Phase.models.pipeline_state.os.path.isdir', return_value=True):
            with pytest.raises(IOError, match="Cannot load from directory"):
                PipelineStateManager.load_state_from_file("/tmp")
