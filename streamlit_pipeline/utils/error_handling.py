"""
Simplified Error Handling and Logging System for GraphJudge Streamlit Pipeline.

This module provides a unified approach to error handling and logging that is
optimized for Streamlit integration. It replaces the complex file-based logging
systems from the original CLI scripts with user-friendly error reporting.

Key principles:
- Errors returned as data, never raised as exceptions
- User-friendly error messages for non-technical users  
- Streamlit-compatible progress indication
- Session-specific logging with unique run IDs
- Graceful degradation for partial results

Following spec.md Sections 8, 10, 12 requirements.
"""

import logging
import streamlit as st
import uuid
import time
from datetime import datetime
from enum import Enum, auto
from typing import Optional, Dict, Any, List, Callable, Union
from dataclasses import dataclass, field
from contextlib import contextmanager
import sys
import io
from pathlib import Path


class ErrorType(Enum):
    """
    Classification of error types as specified in spec.md Section 10.
    
    This helps provide appropriate error handling strategies and user messages.
    """
    CONFIGURATION = "configuration"      # Invalid/missing API keys
    API_AUTH = "api_auth"               # 401 Unauthorized
    API_RATE_LIMIT = "api_rate_limit"   # 429 Rate Limit
    API_SERVER = "api_server"           # 5xx Server Error
    API_TIMEOUT = "api_timeout"         # Request timeout
    VALIDATION = "validation"           # Malformed data/responses
    FILE_SYSTEM = "file_system"         # File read/write errors
    INPUT_VALIDATION = "input_validation"  # Invalid user input
    PROCESSING = "processing"           # General processing errors


class ErrorSeverity(Enum):
    """Error severity levels for appropriate user messaging."""
    LOW = "low"          # Minor issues, processing can continue
    MEDIUM = "medium"    # Significant issues, partial results possible
    HIGH = "high"        # Critical issues, processing must stop
    CRITICAL = "critical"  # System-level issues, application may be unstable


@dataclass
class ErrorInfo:
    """
    Structured error information following spec.md Section 8 data model patterns.
    
    Contains all information needed to handle errors gracefully and provide
    meaningful feedback to users without exposing technical details.
    
    Attributes:
        error_type: Classification of the error type
        severity: How serious the error is
        message: User-friendly error message
        technical_details: Detailed technical information for debugging
        suggestions: List of suggested actions for the user
        stage: Which pipeline stage the error occurred in
        timestamp: When the error occurred
        retry_possible: Whether the operation can be retried
        partial_results: Whether partial results are available despite the error
    """
    error_type: ErrorType
    severity: ErrorSeverity  
    message: str
    technical_details: Optional[str] = None
    suggestions: List[str] = field(default_factory=list)
    stage: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.now)
    retry_possible: bool = False
    partial_results: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert error info to dictionary for logging and debugging."""
        return {
            'error_type': self.error_type.value,
            'severity': self.severity.value,
            'message': self.message,
            'technical_details': self.technical_details,
            'suggestions': self.suggestions,
            'stage': self.stage,
            'timestamp': self.timestamp.isoformat(),
            'retry_possible': self.retry_possible,
            'partial_results': self.partial_results
        }
    
    def display_in_streamlit(self):
        """Display the error in Streamlit with appropriate styling."""
        if self.severity == ErrorSeverity.CRITICAL:
            st.error(f"🚨 Critical Error: {self.message}")
        elif self.severity == ErrorSeverity.HIGH:
            st.error(f"❌ Error: {self.message}")
        elif self.severity == ErrorSeverity.MEDIUM:
            st.warning(f"⚠️ Warning: {self.message}")
        else:
            st.info(f"ℹ️ Notice: {self.message}")
        
        if self.suggestions:
            st.markdown("**Suggested actions:**")
            for suggestion in self.suggestions:
                st.markdown(f"• {suggestion}")
        
        if self.technical_details and st.checkbox("Show technical details"):
            st.code(self.technical_details, language="text")


class StreamlitLogger:
    """
    Streamlit-compatible logging system with session-specific functionality.
    
    This class provides logging that integrates well with Streamlit's session-based
    architecture while maintaining the ability to track operations across the
    three-stage pipeline.
    
    Features:
    - Session-specific logging with unique run IDs
    - Real-time progress indication through Streamlit UI
    - Structured logging with pipeline stage tracking
    - Optional file output for debugging
    """
    
    def __init__(self, run_id: Optional[str] = None, log_to_file: bool = False):
        """
        Initialize the Streamlit logger.
        
        Args:
            run_id: Unique identifier for this pipeline run
            log_to_file: Whether to also log to file for debugging
        """
        self.run_id = run_id or str(uuid.uuid4())[:8]
        self.log_to_file = log_to_file
        self.start_time = datetime.now()
        self.log_entries: List[Dict[str, Any]] = []
        
        # Initialize Python logger if file logging is enabled
        if log_to_file:
            self._setup_file_logger()
    
    def _setup_file_logger(self):
        """Set up file-based logging for debugging purposes."""
        log_dir = Path("streamlit_pipeline/logs")
        log_dir.mkdir(parents=True, exist_ok=True)
        
        log_file = log_dir / f"pipeline_{self.run_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        
        self.file_logger = logging.getLogger(f"pipeline_{self.run_id}")
        self.file_logger.setLevel(logging.DEBUG)
        
        handler = logging.FileHandler(log_file, encoding='utf-8')
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        handler.setFormatter(formatter)
        self.file_logger.addHandler(handler)
    
    def info(self, message: str, stage: Optional[str] = None, extra: Optional[Dict] = None):
        """Log an info message."""
        self._log("INFO", message, stage, extra)
    
    def warning(self, message: str, stage: Optional[str] = None, extra: Optional[Dict] = None):
        """Log a warning message."""
        self._log("WARNING", message, stage, extra)
    
    def error(self, message: str, stage: Optional[str] = None, extra: Optional[Dict] = None):
        """Log an error message."""
        self._log("ERROR", message, stage, extra)
    
    def debug(self, message: str, stage: Optional[str] = None, extra: Optional[Dict] = None):
        """Log a debug message."""
        self._log("DEBUG", message, stage, extra)

    # Convenience methods for backward compatibility
    def log_info(self, message: str, extra: Optional[Dict] = None):
        """Log an info message (backward compatibility)."""
        self.info(message, extra=extra)

    def log_error(self, message: str, extra: Optional[Dict] = None):
        """Log an error message (backward compatibility)."""
        self.error(message, extra=extra)

    def log_warning(self, message: str, extra: Optional[Dict] = None):
        """Log a warning message (backward compatibility)."""
        self.warning(message, extra=extra)
    
    def _log(self, level: str, message: str, stage: Optional[str], extra: Optional[Dict]):
        """Internal logging method."""
        timestamp = datetime.now()
        elapsed = (timestamp - self.start_time).total_seconds()
        
        log_entry = {
            'timestamp': timestamp.isoformat(),
            'run_id': self.run_id,
            'level': level,
            'message': message,
            'stage': stage,
            'elapsed_seconds': elapsed,
            'extra': extra or {}
        }
        
        self.log_entries.append(log_entry)
        
        # Log to file if enabled
        if self.log_to_file and hasattr(self, 'file_logger'):
            log_msg = f"[{stage or 'GENERAL'}] {message}"
            if extra:
                log_msg += f" | Extra: {extra}"
            
            if level == "ERROR":
                self.file_logger.error(log_msg)
            elif level == "WARNING":
                self.file_logger.warning(log_msg)
            elif level == "DEBUG":
                self.file_logger.debug(log_msg)
            else:
                self.file_logger.info(log_msg)
    
    def get_logs(self, level: Optional[str] = None) -> List[Dict[str, Any]]:
        """Retrieve log entries, optionally filtered by level."""
        if level:
            return [entry for entry in self.log_entries if entry['level'] == level]
        return self.log_entries.copy()
    
    def clear_logs(self):
        """Clear all log entries."""
        self.log_entries.clear()
    
    @contextmanager
    def progress_context(self, description: str, stage: Optional[str] = None):
        """
        Context manager for showing progress in Streamlit.
        
        Usage:
            with logger.progress_context("Processing entities..."):
                # Do work here
        """
        self.info(f"Starting: {description}", stage)
        start_time = time.time()
        
        # Show progress in Streamlit
        with st.spinner(description):
            try:
                yield
                elapsed = time.time() - start_time
                self.info(f"Completed: {description} (took {elapsed:.2f}s)", stage)
            except Exception as e:
                elapsed = time.time() - start_time
                self.error(f"Failed: {description} after {elapsed:.2f}s - {str(e)}", stage)
                raise


class ErrorHandler:
    """
    Unified error handling system for the GraphJudge pipeline.
    
    This class provides consistent error handling across all pipeline stages,
    following the spec.md Section 10 error taxonomy and handling strategies.
    """
    
    # Error message templates for user-friendly display
    ERROR_MESSAGES = {
        ErrorType.CONFIGURATION: {
            'message': "Configuration issue detected",
            'suggestions': ["Check your API keys in environment variables", "Verify configuration settings"]
        },
        ErrorType.API_AUTH: {
            'message': "Authentication failed with the AI service",
            'suggestions': ["Check your API key is valid", "Verify API permissions"]
        },
        ErrorType.API_RATE_LIMIT: {
            'message': "Rate limit exceeded - too many requests", 
            'suggestions': ["Wait a moment and try again", "Consider upgrading your API plan"]
        },
        ErrorType.API_SERVER: {
            'message': "The AI service is experiencing issues",
            'suggestions': ["Wait a moment and try again", "Check service status page"]
        },
        ErrorType.API_TIMEOUT: {
            'message': "Request took too long to complete",
            'suggestions': ["Try with shorter input text", "Check your internet connection"]
        },
        ErrorType.VALIDATION: {
            'message': "The AI service returned invalid data",
            'suggestions': ["Try again with different input", "Report this issue if it persists"]
        },
        ErrorType.FILE_SYSTEM: {
            'message': "File system error occurred", 
            'suggestions': ["Check available disk space", "Verify file permissions"]
        },
        ErrorType.INPUT_VALIDATION: {
            'message': "Input text has issues",
            'suggestions': ["Check your input text", "Try with shorter or cleaner text"]
        },
        ErrorType.PROCESSING: {
            'message': "Processing error occurred",
            'suggestions': ["Try again", "Check input format"]
        }
    }
    
    @classmethod 
    def create_error(cls, 
                    error_type: ErrorType,
                    message: Optional[str] = None,
                    technical_details: Optional[str] = None,
                    stage: Optional[str] = None,
                    severity: Optional[ErrorSeverity] = None,
                    retry_possible: bool = False,
                    partial_results: bool = False) -> ErrorInfo:
        """
        Create a structured error with appropriate defaults.
        
        Args:
            error_type: The type of error that occurred
            message: Custom error message (falls back to template)
            technical_details: Technical details for debugging
            stage: Which pipeline stage the error occurred in
            severity: How serious the error is (auto-determined if None)
            retry_possible: Whether the operation can be retried
            partial_results: Whether partial results are available
            
        Returns:
            ErrorInfo: Structured error information
        """
        template = cls.ERROR_MESSAGES.get(error_type, {})
        final_message = message or template.get('message', str(error_type.value))
        suggestions = template.get('suggestions', [])
        
        if severity is None:
            severity = cls._determine_severity(error_type)
        
        return ErrorInfo(
            error_type=error_type,
            severity=severity,
            message=final_message,
            technical_details=technical_details,
            suggestions=suggestions,
            stage=stage,
            retry_possible=retry_possible,
            partial_results=partial_results
        )
    
    @staticmethod
    def _determine_severity(error_type: ErrorType) -> ErrorSeverity:
        """Determine error severity based on error type."""
        severity_map = {
            ErrorType.CONFIGURATION: ErrorSeverity.CRITICAL,
            ErrorType.API_AUTH: ErrorSeverity.HIGH, 
            ErrorType.API_RATE_LIMIT: ErrorSeverity.MEDIUM,
            ErrorType.API_SERVER: ErrorSeverity.HIGH,
            ErrorType.API_TIMEOUT: ErrorSeverity.MEDIUM,
            ErrorType.VALIDATION: ErrorSeverity.MEDIUM,
            ErrorType.FILE_SYSTEM: ErrorSeverity.HIGH,
            ErrorType.INPUT_VALIDATION: ErrorSeverity.LOW,
            ErrorType.PROCESSING: ErrorSeverity.MEDIUM
        }
        return severity_map.get(error_type, ErrorSeverity.MEDIUM)
    
    @classmethod
    def handle_api_error(cls, exception: Exception, stage: str) -> ErrorInfo:
        """
        Handle API-related errors with appropriate classification.
        
        Args:
            exception: The exception that occurred
            stage: Which pipeline stage was running
            
        Returns:
            ErrorInfo: Classified error information
        """
        error_str = str(exception).lower()
        
        if "401" in error_str or "unauthorized" in error_str:
            return cls.create_error(
                ErrorType.API_AUTH,
                stage=stage,
                technical_details=str(exception),
                retry_possible=False
            )
        elif "429" in error_str or "rate limit" in error_str:
            return cls.create_error(
                ErrorType.API_RATE_LIMIT,
                stage=stage, 
                technical_details=str(exception),
                retry_possible=True
            )
        elif any(code in error_str for code in ["500", "502", "503", "504"]):
            return cls.create_error(
                ErrorType.API_SERVER,
                stage=stage,
                technical_details=str(exception),
                retry_possible=True
            )
        elif "timeout" in error_str:
            return cls.create_error(
                ErrorType.API_TIMEOUT,
                stage=stage,
                technical_details=str(exception),
                retry_possible=True
            )
        else:
            return cls.create_error(
                ErrorType.PROCESSING,
                message=f"API error: {str(exception)}",
                stage=stage,
                technical_details=str(exception),
                retry_possible=True
            )


def safe_execute(func: Callable, *args, logger: Optional[StreamlitLogger] = None, 
                stage: Optional[str] = None, **kwargs) -> tuple[Any, Optional[ErrorInfo]]:
    """
    Safely execute a function and return results with error information.
    
    This utility function wraps any operation to ensure errors are returned
    as data rather than raised as exceptions, following spec.md Section 8.
    
    Args:
        func: The function to execute
        *args: Positional arguments for the function
        logger: Optional logger for recording execution
        stage: Pipeline stage for error context
        **kwargs: Keyword arguments for the function
        
    Returns:
        Tuple of (result, error_info). If successful, error_info is None.
        If failed, result is None and error_info contains details.
    """
    try:
        if logger:
            logger.debug(f"Executing {func.__name__}", stage)
        
        result = func(*args, **kwargs)
        
        if logger:
            logger.debug(f"Successfully completed {func.__name__}", stage)
        
        return result, None
        
    except Exception as e:
        error_info = ErrorHandler.handle_api_error(e, stage or "unknown")
        
        if logger:
            logger.error(f"Error in {func.__name__}: {str(e)}", stage)
        
        return None, error_info


@contextmanager
def streamlit_error_handler(stage: str, logger: Optional[StreamlitLogger] = None):
    """
    Context manager for handling errors in Streamlit context.
    
    Usage:
        with streamlit_error_handler("entity_extraction", logger) as handler:
            result = some_operation()
            if handler.has_error():
                return handler.error_info
    """
    handler_state = {'error_info': None}
    
    try:
        yield handler_state
    except Exception as e:
        error_info = ErrorHandler.handle_api_error(e, stage)
        handler_state['error_info'] = error_info
        
        if logger:
            logger.error(f"Error in {stage}: {str(e)}", stage)
        
        # Display error in Streamlit
        error_info.display_in_streamlit()


# Progress indication utilities for Streamlit
class ProgressTracker:
    """
    Simple progress tracking for multi-stage operations.
    
    Provides user-friendly progress indication following spec.md Section 3 (FR-I2).
    """
    
    def __init__(self, total_stages: int = 3):
        """Initialize progress tracker."""
        self.total_stages = total_stages
        self.current_stage = 0
        self.stage_names = ["Entity Extraction", "Triple Generation", "Graph Judgment"]
        self.progress_bar = None
        self.status_text = None
    
    def start(self):
        """Initialize progress display in Streamlit."""
        self.progress_bar = st.progress(0)
        self.status_text = st.empty()
    
    def update(self, stage_index: int, message: str = None):
        """Update progress display."""
        self.current_stage = stage_index
        progress = (stage_index / self.total_stages)
        
        if self.progress_bar:
            self.progress_bar.progress(progress)
        
        if self.status_text:
            stage_name = self.stage_names[min(stage_index, len(self.stage_names) - 1)]
            status_msg = message or f"Processing {stage_name}..."
            self.status_text.text(status_msg)
    
    def complete(self):
        """Mark processing as complete."""
        if self.progress_bar:
            self.progress_bar.progress(1.0)
        if self.status_text:
            self.status_text.text("✅ Pipeline completed successfully!")
    
    def error(self, message: str):
        """Mark processing as failed."""
        if self.status_text:
            self.status_text.text(f"❌ {message}")


# Global logger instance for easy access
_global_logger: Optional[StreamlitLogger] = None

def get_logger() -> StreamlitLogger:
    """Get or create the global logger instance."""
    global _global_logger
    if _global_logger is None:
        _global_logger = StreamlitLogger()
    return _global_logger

def set_logger(logger: StreamlitLogger):
    """Set the global logger instance."""
    global _global_logger
    _global_logger = logger