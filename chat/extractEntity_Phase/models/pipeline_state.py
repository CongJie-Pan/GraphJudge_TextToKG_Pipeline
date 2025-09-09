"""
Pipeline State Management Module

This module defines the data structures for tracking pipeline execution state,
including processing status, progress tracking, and error handling.

The module provides comprehensive state management for the ECTD pipeline,
enabling monitoring, debugging, and recovery of pipeline operations.
"""

from typing import List, Optional, Dict, Any, Union
from enum import Enum, StrEnum
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from pydantic import BaseModel, Field, validator
import json
import os


class ProcessingStatus(str, Enum):
    """Enumeration of pipeline processing statuses."""
    
    PENDING = "pending"           # Pipeline is waiting to start
    INITIALIZING = "initializing"  # Pipeline is initializing
    RUNNING = "running"           # Pipeline is actively processing
    PAUSED = "paused"            # Pipeline is temporarily paused
    COMPLETED = "completed"       # Pipeline completed successfully
    FAILED = "failed"             # Pipeline failed with errors
    CANCELLED = "cancelled"       # Pipeline was cancelled by user
    ROLLING_BACK = "rolling_back" # Pipeline is rolling back changes


class PipelineStage(StrEnum):
    """Enumeration of pipeline processing stages."""
    
    TEXT_PREPROCESSING = "text_preprocessing"    # Text preprocessing stage
    ENTITY_EXTRACTION = "entity_extraction"      # Entity extraction from text
    TEXT_DENOISING = "text_denoising"           # Text denoising based on entities
    POST_PROCESSING = "post_processing"         # Post-processing stage
    VALIDATION = "validation"                   # Validation stage

class ErrorSeverity(StrEnum):
    """Enumeration of error severity levels."""
    
    LOW = "low"           # Low severity error
    MEDIUM = "medium"     # Medium severity error  
    HIGH = "high"         # High severity error
    CRITICAL = "critical" # Critical error that stops processing

@dataclass
class PipelineError:
    """Pipeline error information."""
    
    stage: PipelineStage
    message: str
    severity: ErrorSeverity = ErrorSeverity.MEDIUM  # set to medium by default
    error_code: Optional[str] = None
    exception: Optional[Exception] = None
    timestamp: datetime = field(default_factory=datetime.now)
    details: Dict[str, Any] = field(default_factory=dict)  # set to empty dictionary by default
    
    def __str__(self) -> str:
        """String representation of pipeline error."""
        return f"[{self.severity}] {self.stage}: {self.message}"

    def to_dict(self) -> Dict[str, Any]:
        """Convert error to dictionary for serialization."""
        return {
            "stage": self.stage.value,
            "severity": self.severity.value,
            "message": self.message,
            "error_code": self.error_code,
            "exception_type": type(self.exception).__name__ if self.exception else None,
            "exception_message": str(self.exception) if self.exception else None,
            "timestamp": self.timestamp.isoformat(),
            "details": self.details  # set to empty dictionary by default
        }

@dataclass
class StageProgress:
    """Progress information for a specific pipeline stage."""
    
    stage: PipelineStage
    status: ProcessingStatus
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    progress_percentage: float = 0.0
    items_processed: int = 0
    total_items: int = 0
    current_item: Optional[str] = None
    errors: List[PipelineError] = field(default_factory=list)
    
    def __post_init__(self):
        """Initialize start_time if status is RUNNING."""
        if self.status == ProcessingStatus.RUNNING and self.start_time is None:
            self.start_time = datetime.now()
    
    @property
    def duration(self) -> Optional[timedelta]:
        """Get stage duration if completed."""
        if self.start_time and self.end_time:
            return self.end_time - self.start_time
        return None
    
    @property
    def is_completed(self) -> bool:
        """Check if stage is completed."""
        return self.status in [ProcessingStatus.COMPLETED, ProcessingStatus.FAILED]
    
    @property
    def is_running(self) -> bool:
        """Check if stage is currently running."""
        return self.status == ProcessingStatus.RUNNING
    
    def start(self) -> None:
        """Mark stage as started."""
        self.start_time = datetime.now()
        self.status = ProcessingStatus.RUNNING
        self.progress_percentage = 0.0
    
    def update_progress(self, progress_percentage: float) -> None:
        """
        Update stage progress.
        
        Args:
            progress_percentage: Progress percentage (0.0 to 100.0)
        """
        if self.status != ProcessingStatus.RUNNING:
            raise ValueError("Cannot update progress of non-running stage")
        
        if not (0.0 <= progress_percentage <= 100.0):
            raise ValueError("Progress percentage must be between 0.0 and 100.0")
        
        self.progress_percentage = progress_percentage
    
    def complete(self, success: bool = True) -> None:
        """
        Mark stage as completed.
        
        Args:
            success: Whether stage completed successfully
        """
        if self.start_time is None:
            raise ValueError("Cannot complete non-running stage")
        
        self.end_time = datetime.now()
        self.status = ProcessingStatus.COMPLETED if success else ProcessingStatus.FAILED
        self.progress_percentage = 100.0 if success else self.progress_percentage
    
    def add_error(self, error: PipelineError) -> None:
        """
        Add error to stage.
        
        Args:
            error: Pipeline error to add
        """
        self.errors.append(error)
        
        # Update status based on error severity
        if error.severity in [ErrorSeverity.CRITICAL, ErrorSeverity.HIGH]:
            self.status = ProcessingStatus.FAILED
    
    def get_duration(self) -> Optional[float]:
        """Get stage duration in seconds if completed."""
        if self.start_time and self.end_time:
            return (self.end_time - self.start_time).total_seconds()
        return None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert stage progress to dictionary for serialization."""
        return {
            "stage": self.stage.value,
            "status": self.status.value,
            "start_time": self.start_time.isoformat() if self.start_time else None,
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "duration": str(self.duration) if self.duration else None,
            "progress_percentage": round(self.progress_percentage, 2),
            "items_processed": self.items_processed,
            "total_items": self.total_items,
            "current_item": self.current_item,
            "errors": [error.to_dict() for error in self.errors],
            "is_completed": self.is_completed,
            "is_running": self.is_running
        }

class PipelineState(BaseModel):
    """
    Complete pipeline execution state.
    
    This class tracks the complete state of the ECTD pipeline execution,
    including all stages, progress, errors, and metadata.
    """
    
    model_config = {"arbitrary_types_allowed": True}
    
    pipeline_id: str = Field(..., description="Unique pipeline identifier")
    source_text: str = Field(..., description="Source text for processing")
    status: ProcessingStatus = Field(default=ProcessingStatus.PENDING, description="Overall pipeline status")
    created_at: datetime = Field(default_factory=datetime.now, description="Pipeline creation time")
    updated_at: datetime = Field(default_factory=datetime.now, description="Pipeline last update time")
    started_at: Optional[datetime] = Field(None, description="Pipeline start time")
    completed_at: Optional[datetime] = Field(None, description="Pipeline completion time")
    cancelled_at: Optional[datetime] = Field(None, description="Pipeline cancellation time")
    cancellation_reason: Optional[str] = Field(None, description="Reason for cancellation")
    start_time: Optional[datetime] = Field(None, description="Pipeline start time")
    end_time: Optional[datetime] = Field(None, description="Pipeline end time")
    stages: List[StageProgress] = Field(default_factory=list, description="Stage progress tracking")
    errors: List[PipelineError] = Field(default_factory=list, description="Pipeline-level errors")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional pipeline metadata")
    configuration: Dict[str, Any] = Field(default_factory=dict, description="Pipeline configuration snapshot")
    
    @validator('pipeline_id')
    def validate_pipeline_id(cls, v):
        """Validate pipeline ID."""
        if not v or not v.strip():
            raise ValueError("Pipeline ID cannot be empty")
        return v.strip()
    
    @validator('source_text')
    def validate_source_text(cls, v):
        """Validate source text."""
        if v is None:
            raise ValueError("Source text cannot be None")
        # 允許空字串，只拒絕 None
        return v
    
    def model_post_init(self, __context):
        """Initialize default stages after object creation."""
        if not self.stages:
            self._initialize_stages()
    
    def _initialize_stages(self) -> None:
        """Initialize all pipeline stages with default progress."""
        self.stages = [
            StageProgress(stage=stage, status=ProcessingStatus.PENDING)
            for stage in PipelineStage
        ]
    
    @property
    def duration(self) -> float:
        """Get pipeline duration in seconds if completed."""
        if self.start_time and self.end_time:
            return (self.end_time - self.start_time).total_seconds()
        return 0.0
    
    @property
    def overall_progress(self) -> float:
        """Get overall pipeline progress percentage."""
        if not self.stages:
            return 0.0
        
        total_progress = sum(stage.progress_percentage for stage in self.stages)
        return round(total_progress / len(self.stages), 2)
    
    @property
    def is_completed(self) -> bool:
        """Check if pipeline is completed."""
        return self.status in [ProcessingStatus.COMPLETED, ProcessingStatus.FAILED, ProcessingStatus.CANCELLED]
    
    @property
    def is_running(self) -> bool:
        """Check if pipeline is currently running."""
        return self.status == ProcessingStatus.RUNNING
    
    @property
    def has_errors(self) -> bool:
        """Check if pipeline has any errors."""
        return len(self.errors) > 0 or any(len(stage.errors) > 0 for stage in self.stages)
    
    @property
    def critical_errors(self) -> List[PipelineError]:
        """Get list of critical errors."""
        critical_errors = [e for e in self.errors if e.severity == ErrorSeverity.CRITICAL]
        
        for stage in self.stages:
            critical_errors.extend([e for e in stage.errors if e.severity == ErrorSeverity.CRITICAL])
        
        return critical_errors
    
    def start(self) -> None:
        """Start the pipeline execution."""
        self.start_time = datetime.now()
        self.started_at = datetime.now()
        self.status = ProcessingStatus.RUNNING
        
        # Start first stage
        if self.stages:
            self.stages[0].start()
    
    def complete(self, success: bool = True) -> None:
        """
        Complete the pipeline execution.
        
        Args:
            success: Whether pipeline completed successfully
        """
        if self.status == ProcessingStatus.PENDING:
            raise ValueError("Cannot complete pipeline that has not started")
        
        self.end_time = datetime.now()
        self.completed_at = datetime.now()
        self.status = ProcessingStatus.COMPLETED if success else ProcessingStatus.FAILED
        
        # Complete all stages
        for stage in self.stages:
            if stage.is_running:
                stage.complete(success)
            elif stage.status == ProcessingStatus.PENDING:
                stage.status = ProcessingStatus.COMPLETED if success else ProcessingStatus.FAILED

    def cancel(self, reason: str = None) -> None:
        """Cancel the pipeline execution."""
        if self.status == ProcessingStatus.PENDING:
            raise ValueError("Cannot cancel pipeline that has not started")
        
        self.end_time = datetime.now()
        self.cancelled_at = datetime.now()
        self.cancellation_reason = reason
        self.status = ProcessingStatus.CANCELLED
        
        # Cancel all running stages
        for stage in self.stages:
            if stage.is_running:
                stage.status = ProcessingStatus.CANCELLED
    
    def get_stage(self, stage_name: PipelineStage) -> Optional[StageProgress]:
        """
        Get stage progress by name.
        
        Args:
            stage_name: Name of the stage to retrieve
            
        Returns:
            Stage progress object or None if not found
        """
        for stage in self.stages:
            if stage.stage == stage_name:
                return stage
        return None
    
    def start_stage(self, stage_name: PipelineStage) -> None:
        """
        Start a specific pipeline stage.
        
        Args:
            stage_name: Name of the stage to start
        """
        stage = self.get_stage(stage_name)
        if stage:
            stage.start()
    
    def complete_stage(self, stage_name: PipelineStage, success: bool = True) -> None:
        """
        Complete a specific pipeline stage.
        
        Args:
            stage_name: Name of the stage to complete
            success: Whether stage completed successfully
        """
        stage = self.get_stage(stage_name)
        if stage:
            stage.complete(success)
    
    def add_error(self, error: PipelineError) -> None:
        """
        Add error to pipeline.
        
        Args:
            error: Pipeline error to add
        """
        self.errors.append(error)
        
        # Update overall status based on error severity
        if error.severity in [ErrorSeverity.CRITICAL, ErrorSeverity.HIGH, ErrorSeverity.MEDIUM]:
            self.status = ProcessingStatus.FAILED
            self.end_time = datetime.now()
    
    def add_stage_error(self, stage_name: PipelineStage, error: PipelineError) -> None:
        """
        Add error to specific stage.
        
        Args:
            stage_name: Name of the stage to add error to
            error: Pipeline error to add
        """
        stage = self.get_stage(stage_name)
        if stage:
            stage.add_error(error)
            
            # Update overall status if critical error
            if error.severity == ErrorSeverity.CRITICAL:
                self.status = ProcessingStatus.FAILED
                self.end_time = datetime.now()
    
    def get_summary(self) -> Dict[str, Any]:
        """Get pipeline execution summary."""
        return {
            "pipeline_id": self.pipeline_id,
            "status": self.status.value,
            "start_time": self.start_time.isoformat() if self.start_time else None,
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "duration": str(self.duration) if self.duration else None,
            "overall_progress": self.overall_progress,
            "stages": {
                stage.stage.value: stage.to_dict() 
                for stage in self.stages
            },
            "total_errors": len(self.errors) + sum(len(stage.errors) for stage in self.stages),
            "critical_errors": len(self.critical_errors),
            "is_completed": self.is_completed,
            "is_running": self.is_running,
            "has_errors": self.has_errors
        }
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert pipeline state to dictionary for serialization."""
        return {
            "pipeline_id": self.pipeline_id,
            "source_text": self.source_text,
            "status": self.status.value,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "cancelled_at": self.cancelled_at.isoformat() if self.cancelled_at else None,
            "cancellation_reason": self.cancellation_reason,
            "start_time": self.start_time.isoformat() if self.start_time else None,
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "duration": str(self.duration) if self.duration else None,
            "overall_progress": self.overall_progress,
            "stages": [stage.to_dict() for stage in self.stages],
            "errors": [error.to_dict() for error in self.errors],
            "metadata": self.metadata,
            "configuration": self.configuration,
            "summary": self.get_summary()
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'PipelineState':
        """Create pipeline state from dictionary."""
        # Handle datetime conversions
        for time_field in ['created_at', 'updated_at', 'started_at', 'completed_at', 'cancelled_at', 'start_time', 'end_time']:
            if time_field in data and isinstance(data[time_field], str):
                data[time_field] = datetime.fromisoformat(data[time_field])
        
        # Convert stage dictionaries to StageProgress objects
        if 'stages' in data:
            converted_stages = []
            for stage_data in data['stages']:
                stage_enum = PipelineStage(stage_data['stage'])
                stage_progress = StageProgress(
                    stage=stage_enum,
                    status=ProcessingStatus(stage_data['status']),
                    start_time=datetime.fromisoformat(stage_data['start_time']) if stage_data.get('start_time') else None,
                    end_time=datetime.fromisoformat(stage_data['end_time']) if stage_data.get('end_time') else None,
                    progress_percentage=stage_data.get('progress_percentage', 0.0),
                    items_processed=stage_data.get('items_processed', 0),
                    total_items=stage_data.get('total_items', 0),
                    current_item=stage_data.get('current_item'),
                    errors=[PipelineError(**e) for e in stage_data.get('errors', [])]
                )
                converted_stages.append(stage_progress)
            data['stages'] = converted_stages
        
        # Convert error dictionaries to PipelineError objects
        if 'errors' in data:
            data['errors'] = [PipelineError(**e) for e in data['errors']]
        
        return cls(**data)
    
    def __str__(self) -> str:
        """String representation of pipeline state."""
        return f"PipelineState({self.pipeline_id}, status={self.status.value}, progress={self.overall_progress}%)"
    
    def __repr__(self) -> str:
        """Detailed string representation of pipeline state."""
        return f"PipelineState(pipeline_id='{self.pipeline_id}', status={self.status.value}, stages={len(self.stages)}, errors={len(self.errors)})"

class PipelineStateManager:
    """
    Manager for pipeline state operations.
    
    This class provides utility methods for creating, updating, and managing
    pipeline state objects throughout the execution lifecycle.
    """
    
    @staticmethod
    def create_pipeline_state(source_text: str, configuration: Dict[str, Any] = None) -> PipelineState:
        """
        Create a new pipeline state instance.
        
        Args:
            source_text: Source text for processing
            configuration: Pipeline configuration snapshot
            
        Returns:
            New pipeline state instance
        """
        pipeline_id = PipelineStateManager.generate_pipeline_id()
        return PipelineState(
            pipeline_id=pipeline_id,
            source_text=source_text,
            configuration=configuration or {}
        )
    
    @staticmethod
    def save_state_to_file(state: PipelineState, file_path: str) -> None:
        """
        Save pipeline state to file.
        
        Args:
            state: Pipeline state to save
            file_path: Path to save state file
        """
        try:
            # 檢查路徑是否為目錄
            if os.path.isdir(file_path):
                raise IOError(f"Cannot save to directory: {file_path}")
            
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(state.to_dict(), f, ensure_ascii=False, indent=2)
        except (OSError, IOError, PermissionError, FileNotFoundError) as e:
            raise IOError(f"Failed to save state to file {file_path}: {e}")
    
    @staticmethod
    def load_state_from_file(file_path: str) -> PipelineState:
        """
        Load pipeline state from file.
        
        Args:
            file_path: Path to state file
            
        Returns:
            Loaded pipeline state instance
        """
        try:
            # 檢查路徑是否為目錄
            if os.path.isdir(file_path):
                raise IOError(f"Cannot load from directory: {file_path}")
                
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except FileNotFoundError:
            raise FileNotFoundError(f"File not found: {file_path}")
        except (OSError, IOError, PermissionError) as e:
            raise IOError(f"Failed to load state from file {file_path}: {e}")
        
        return PipelineState.from_dict(data)
    
    @staticmethod
    def generate_pipeline_id() -> str:
        """
        Generate a unique pipeline identifier.
        
        Returns:
            Unique pipeline ID string
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        import uuid
        unique_id = str(uuid.uuid4())[:8]
        return f"pipeline_{timestamp}_{unique_id}"
