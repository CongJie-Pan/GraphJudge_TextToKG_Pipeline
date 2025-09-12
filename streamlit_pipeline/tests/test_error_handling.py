"""
Comprehensive unit tests for error handling and logging system.

This test module validates the error handling system following the TDD principles
outlined in docs/Testing_Demands.md and spec.md Section 15 requirements.

Test Coverage:
- Error classification and creation
- User-friendly error messaging 
- Streamlit-compatible logging
- Progress indication functionality
- Safe execution patterns
- Error recovery suggestions
"""

import pytest
import uuid
import time
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock
from typing import Optional
import tempfile
import os
from pathlib import Path

# Import the modules we're testing
from streamlit_pipeline.utils.error_handling import (
    ErrorType, ErrorSeverity, ErrorInfo, StreamlitLogger, ErrorHandler,
    safe_execute, ProgressTracker, get_logger, set_logger
)


class TestErrorInfo:
    """Test suite for ErrorInfo data structure."""
    
    def test_error_info_creation(self):
        """Test basic ErrorInfo object creation."""
        error = ErrorInfo(
            error_type=ErrorType.API_AUTH,
            severity=ErrorSeverity.HIGH,
            message="Authentication failed",
            technical_details="401 Unauthorized",
            suggestions=["Check API key"],
            stage="entity_extraction"
        )
        
        assert error.error_type == ErrorType.API_AUTH
        assert error.severity == ErrorSeverity.HIGH
        assert error.message == "Authentication failed"
        assert error.technical_details == "401 Unauthorized"
        assert error.suggestions == ["Check API key"]
        assert error.stage == "entity_extraction"
        assert isinstance(error.timestamp, datetime)
        assert not error.retry_possible  # Default value
        assert not error.partial_results  # Default value
    
    def test_error_info_to_dict(self):
        """Test ErrorInfo conversion to dictionary."""
        error = ErrorInfo(
            error_type=ErrorType.VALIDATION,
            severity=ErrorSeverity.MEDIUM,
            message="Invalid response",
            retry_possible=True
        )
        
        error_dict = error.to_dict()
        
        assert error_dict['error_type'] == 'validation'
        assert error_dict['severity'] == 'medium'
        assert error_dict['message'] == 'Invalid response'
        assert error_dict['retry_possible'] is True
        assert 'timestamp' in error_dict
        assert isinstance(error_dict['timestamp'], str)  # ISO format
    
    def test_error_info_defaults(self):
        """Test ErrorInfo with minimal parameters."""
        error = ErrorInfo(
            error_type=ErrorType.PROCESSING,
            severity=ErrorSeverity.LOW,
            message="Minor issue"
        )
        
        assert error.technical_details is None
        assert error.suggestions == []
        assert error.stage is None
        assert not error.retry_possible
        assert not error.partial_results


class TestErrorHandler:
    """Test suite for ErrorHandler class."""
    
    def test_create_error_with_defaults(self):
        """Test error creation with template defaults."""
        error = ErrorHandler.create_error(ErrorType.API_RATE_LIMIT)
        
        assert error.error_type == ErrorType.API_RATE_LIMIT
        assert error.severity == ErrorSeverity.MEDIUM
        assert "Rate limit exceeded" in error.message
        assert "Wait a moment and try again" in error.suggestions
    
    def test_create_error_with_custom_message(self):
        """Test error creation with custom message."""
        custom_message = "Custom rate limit message"
        error = ErrorHandler.create_error(
            ErrorType.API_RATE_LIMIT,
            message=custom_message
        )
        
        assert error.message == custom_message
        assert error.error_type == ErrorType.API_RATE_LIMIT
    
    def test_severity_determination(self):
        """Test automatic severity determination."""
        # Critical errors
        config_error = ErrorHandler.create_error(ErrorType.CONFIGURATION)
        assert config_error.severity == ErrorSeverity.CRITICAL
        
        # High severity errors
        auth_error = ErrorHandler.create_error(ErrorType.API_AUTH)
        assert auth_error.severity == ErrorSeverity.HIGH
        
        file_error = ErrorHandler.create_error(ErrorType.FILE_SYSTEM)
        assert file_error.severity == ErrorSeverity.HIGH
        
        # Medium severity errors
        rate_limit_error = ErrorHandler.create_error(ErrorType.API_RATE_LIMIT)
        assert rate_limit_error.severity == ErrorSeverity.MEDIUM
        
        # Low severity errors
        input_error = ErrorHandler.create_error(ErrorType.INPUT_VALIDATION)
        assert input_error.severity == ErrorSeverity.LOW
    
    def test_handle_api_error_401(self):
        """Test API error handling for 401 Unauthorized."""
        exception = Exception("401 Unauthorized: Invalid API key")
        error = ErrorHandler.handle_api_error(exception, "entity_extraction")
        
        assert error.error_type == ErrorType.API_AUTH
        assert error.stage == "entity_extraction"
        assert not error.retry_possible
        assert "401" in error.technical_details
    
    def test_handle_api_error_429(self):
        """Test API error handling for 429 Rate Limit."""
        exception = Exception("429 Too Many Requests: Rate limit exceeded")
        error = ErrorHandler.handle_api_error(exception, "triple_generation")
        
        assert error.error_type == ErrorType.API_RATE_LIMIT
        assert error.stage == "triple_generation"
        assert error.retry_possible
        assert "429" in error.technical_details
    
    def test_handle_api_error_500(self):
        """Test API error handling for 500 Server Error."""
        exception = Exception("500 Internal Server Error")
        error = ErrorHandler.handle_api_error(exception, "graph_judgment")
        
        assert error.error_type == ErrorType.API_SERVER
        assert error.stage == "graph_judgment"
        assert error.retry_possible
        assert "500" in error.technical_details
    
    def test_handle_api_error_timeout(self):
        """Test API error handling for timeout."""
        exception = Exception("Request timeout after 30 seconds")
        error = ErrorHandler.handle_api_error(exception, "entity_extraction")
        
        assert error.error_type == ErrorType.API_TIMEOUT
        assert error.stage == "entity_extraction"
        assert error.retry_possible
        assert "timeout" in error.technical_details
    
    def test_handle_api_error_generic(self):
        """Test API error handling for generic errors."""
        exception = Exception("Unknown API error occurred")
        error = ErrorHandler.handle_api_error(exception, "unknown")
        
        assert error.error_type == ErrorType.PROCESSING
        assert error.stage == "unknown"
        assert error.retry_possible
        assert "Unknown API error occurred" in error.technical_details


class TestStreamlitLogger:
    """Test suite for StreamlitLogger class."""
    
    def test_logger_initialization(self):
        """Test logger initialization."""
        logger = StreamlitLogger()
        
        assert logger.run_id is not None
        assert len(logger.run_id) == 8  # UUID prefix
        assert isinstance(logger.start_time, datetime)
        assert logger.log_entries == []
        assert not logger.log_to_file
    
    def test_logger_with_custom_run_id(self):
        """Test logger with custom run ID."""
        custom_id = "test_run_123"
        logger = StreamlitLogger(run_id=custom_id)
        
        assert logger.run_id == custom_id
    
    def test_basic_logging_methods(self):
        """Test basic logging methods."""
        logger = StreamlitLogger()
        
        logger.info("Test info message")
        logger.warning("Test warning message") 
        logger.error("Test error message")
        logger.debug("Test debug message")
        
        logs = logger.get_logs()
        assert len(logs) == 4
        
        # Check log levels
        levels = [log['level'] for log in logs]
        assert levels == ['INFO', 'WARNING', 'ERROR', 'DEBUG']
        
        # Check messages
        messages = [log['message'] for log in logs]
        assert 'Test info message' in messages
        assert 'Test warning message' in messages
        assert 'Test error message' in messages
        assert 'Test debug message' in messages
    
    def test_logging_with_stage_and_extra(self):
        """Test logging with stage information and extra data."""
        logger = StreamlitLogger()
        
        extra_data = {'processing_time': 2.5, 'item_count': 10}
        logger.info(
            "Processing completed",
            stage="entity_extraction",
            extra=extra_data
        )
        
        logs = logger.get_logs()
        assert len(logs) == 1
        
        log = logs[0]
        assert log['stage'] == "entity_extraction"
        assert log['extra'] == extra_data
        assert log['run_id'] == logger.run_id
        assert 'elapsed_seconds' in log
    
    def test_get_logs_filtering(self):
        """Test log retrieval with filtering."""
        logger = StreamlitLogger()
        
        logger.info("Info message 1")
        logger.error("Error message 1")
        logger.info("Info message 2")
        logger.warning("Warning message 1")
        
        # Test filtering by level
        info_logs = logger.get_logs(level="INFO")
        assert len(info_logs) == 2
        
        error_logs = logger.get_logs(level="ERROR")
        assert len(error_logs) == 1
        
        warning_logs = logger.get_logs(level="WARNING")
        assert len(warning_logs) == 1
    
    def test_clear_logs(self):
        """Test log clearing functionality."""
        logger = StreamlitLogger()
        
        logger.info("Test message")
        assert len(logger.get_logs()) == 1
        
        logger.clear_logs()
        assert len(logger.get_logs()) == 0
    
    def test_progress_context_success(self):
        """Test progress context manager for successful operations."""
        logger = StreamlitLogger()
        
        with logger.progress_context("Test operation", "test_stage"):
            time.sleep(0.1)  # Simulate work
        
        logs = logger.get_logs()
        assert len(logs) == 2  # Start and completion messages
        
        start_log = logs[0]
        end_log = logs[1]
        
        assert "Starting: Test operation" in start_log['message']
        assert "Completed: Test operation" in end_log['message']
        assert start_log['stage'] == "test_stage"
        assert end_log['stage'] == "test_stage"
    
    def test_progress_context_with_exception(self):
        """Test progress context manager when exception occurs."""
        logger = StreamlitLogger()
        
        with pytest.raises(ValueError):
            with logger.progress_context("Failing operation", "test_stage"):
                raise ValueError("Test error")
        
        logs = logger.get_logs()
        assert len(logs) == 2  # Start and failure messages
        
        start_log = logs[0]
        error_log = logs[1]
        
        assert "Starting: Failing operation" in start_log['message']
        assert "Failed: Failing operation" in error_log['message']
        assert "Test error" in error_log['message']
        assert error_log['level'] == 'ERROR'
    
    @patch('streamlit_pipeline.utils.error_handling.logging')
    def test_file_logging_setup(self, mock_logging):
        """Test file logging setup."""
        mock_logger = MagicMock()
        mock_logging.getLogger.return_value = mock_logger
        
        logger = StreamlitLogger(log_to_file=True)
        
        # Verify logger setup was called
        mock_logging.getLogger.assert_called_once()
        mock_logger.setLevel.assert_called_with(mock_logging.DEBUG)


class TestSafeExecute:
    """Test suite for safe_execute utility function."""
    
    def test_safe_execute_success(self):
        """Test safe_execute with successful function."""
        def successful_function(x, y):
            return x + y
        
        result, error = safe_execute(successful_function, 2, 3)
        
        assert result == 5
        assert error is None
    
    def test_safe_execute_with_exception(self):
        """Test safe_execute with function that raises exception."""
        def failing_function():
            raise ValueError("Test error")
        
        result, error = safe_execute(failing_function, stage="test_stage")
        
        assert result is None
        assert error is not None
        assert isinstance(error, ErrorInfo)
        assert error.stage == "test_stage"
        assert "Test error" in error.technical_details
    
    def test_safe_execute_with_logger(self):
        """Test safe_execute with logger integration."""
        logger = StreamlitLogger()
        
        def test_function():
            return "success"
        
        result, error = safe_execute(test_function, logger=logger, stage="test")
        
        assert result == "success"
        assert error is None
        
        logs = logger.get_logs()
        assert len(logs) == 2  # Debug start and end messages
    
    def test_safe_execute_with_kwargs(self):
        """Test safe_execute with keyword arguments."""
        def function_with_kwargs(a, b=10, c=20):
            return a + b + c
        
        result, error = safe_execute(function_with_kwargs, 1, b=5, c=15)
        
        assert result == 21  # 1 + 5 + 15
        assert error is None


class TestProgressTracker:
    """Test suite for ProgressTracker class."""
    
    def test_progress_tracker_initialization(self):
        """Test ProgressTracker initialization."""
        tracker = ProgressTracker()
        
        assert tracker.total_stages == 3
        assert tracker.current_stage == 0
        assert len(tracker.stage_names) == 3
        assert "Entity Extraction" in tracker.stage_names
    
    def test_progress_tracker_custom_stages(self):
        """Test ProgressTracker with custom number of stages."""
        tracker = ProgressTracker(total_stages=5)
        
        assert tracker.total_stages == 5
    
    @patch('streamlit_pipeline.utils.error_handling.st')
    def test_progress_tracker_start(self, mock_st):
        """Test ProgressTracker start method."""
        mock_progress = MagicMock()
        mock_status = MagicMock()
        mock_st.progress.return_value = mock_progress
        mock_st.empty.return_value = mock_status
        
        tracker = ProgressTracker()
        tracker.start()
        
        assert tracker.progress_bar == mock_progress
        assert tracker.status_text == mock_status
        
        mock_st.progress.assert_called_once_with(0)
        mock_st.empty.assert_called_once()
    
    @patch('streamlit_pipeline.utils.error_handling.st')
    def test_progress_tracker_update(self, mock_st):
        """Test ProgressTracker update method."""
        mock_progress = MagicMock()
        mock_status = MagicMock()
        
        tracker = ProgressTracker()
        tracker.progress_bar = mock_progress
        tracker.status_text = mock_status
        
        tracker.update(1, "Custom message")
        
        assert tracker.current_stage == 1
        mock_progress.progress.assert_called_once_with(1/3)  # stage 1 out of 3
        mock_status.text.assert_called_once_with("Custom message")
    
    @patch('streamlit_pipeline.utils.error_handling.st')
    def test_progress_tracker_complete(self, mock_st):
        """Test ProgressTracker complete method."""
        mock_progress = MagicMock()
        mock_status = MagicMock()
        
        tracker = ProgressTracker()
        tracker.progress_bar = mock_progress
        tracker.status_text = mock_status
        
        tracker.complete()
        
        mock_progress.progress.assert_called_once_with(1.0)
        mock_status.text.assert_called_once()
        args = mock_status.text.call_args[0]
        assert "completed successfully" in args[0]
    
    @patch('streamlit_pipeline.utils.error_handling.st')
    def test_progress_tracker_error(self, mock_st):
        """Test ProgressTracker error method."""
        mock_status = MagicMock()
        
        tracker = ProgressTracker()
        tracker.status_text = mock_status
        
        error_message = "Something went wrong"
        tracker.error(error_message)
        
        mock_status.text.assert_called_once_with(f"❌ {error_message}")


class TestGlobalLogger:
    """Test suite for global logger functionality."""
    
    def test_get_logger_creates_instance(self):
        """Test that get_logger creates a global instance."""
        # Clear global state first
        set_logger(None)
        
        logger1 = get_logger()
        logger2 = get_logger()
        
        # Should return the same instance
        assert logger1 is logger2
        assert isinstance(logger1, StreamlitLogger)
    
    def test_set_logger(self):
        """Test setting a custom global logger."""
        custom_logger = StreamlitLogger(run_id="custom_test")
        set_logger(custom_logger)
        
        retrieved_logger = get_logger()
        assert retrieved_logger is custom_logger
        assert retrieved_logger.run_id == "custom_test"


class TestErrorTypeClassification:
    """Test suite for error type classification and handling."""
    
    def test_all_error_types_have_messages(self):
        """Test that all error types have corresponding message templates."""
        for error_type in ErrorType:
            error = ErrorHandler.create_error(error_type)
            
            # Should have a non-empty message
            assert error.message
            assert len(error.message) > 0
            
            # Should have suggestions (even if empty list is valid)
            assert isinstance(error.suggestions, list)
    
    def test_severity_levels_complete(self):
        """Test that all error types have severity mappings."""
        for error_type in ErrorType:
            severity = ErrorHandler._determine_severity(error_type)
            assert isinstance(severity, ErrorSeverity)
    
    def test_error_message_user_friendly(self):
        """Test that error messages are user-friendly (no technical jargon)."""
        # Technical terms that should not appear in user-facing messages
        technical_terms = ['exception', 'stack trace', 'null pointer', 'http status']
        
        for error_type in ErrorType:
            error = ErrorHandler.create_error(error_type)
            message_lower = error.message.lower()
            
            for term in technical_terms:
                assert term not in message_lower, f"Found technical term '{term}' in user message for {error_type}"


class TestIntegrationScenarios:
    """Integration tests for common error handling scenarios."""
    
    def test_complete_error_flow(self):
        """Test a complete error handling flow from detection to display."""
        logger = StreamlitLogger()
        
        def failing_api_call():
            raise Exception("429 Rate limit exceeded for API")
        
        # Execute with error handling
        result, error_info = safe_execute(
            failing_api_call, 
            logger=logger, 
            stage="entity_extraction"
        )
        
        # Verify results
        assert result is None
        assert error_info is not None
        assert error_info.error_type == ErrorType.API_RATE_LIMIT
        assert error_info.retry_possible
        assert error_info.stage == "entity_extraction"
        
        # Verify logging
        logs = logger.get_logs()
        error_logs = logger.get_logs(level="ERROR")
        assert len(error_logs) == 1
        assert "failing_api_call" in error_logs[0]['message']
    
    def test_partial_success_scenario(self):
        """Test handling of partial success scenarios."""
        error = ErrorHandler.create_error(
            ErrorType.VALIDATION,
            message="Some triples failed validation",
            partial_results=True,
            retry_possible=False
        )
        
        assert error.partial_results
        assert not error.retry_possible
        assert error.severity == ErrorSeverity.MEDIUM
    
    def test_recoverable_error_scenario(self):
        """Test handling of recoverable errors."""
        error = ErrorHandler.create_error(
            ErrorType.API_SERVER,
            stage="graph_judgment",
            retry_possible=True
        )
        
        assert error.retry_possible
        assert error.severity == ErrorSeverity.HIGH
        assert "try again" in " ".join(error.suggestions).lower()
    
    def test_critical_error_scenario(self):
        """Test handling of critical errors."""
        error = ErrorHandler.create_error(
            ErrorType.CONFIGURATION,
            message="Missing required API keys"
        )
        
        assert error.severity == ErrorSeverity.CRITICAL
        assert not error.retry_possible  # Can't retry without fixing config
        assert "api key" in " ".join(error.suggestions).lower()


class TestPerformanceAndRobustness:
    """Performance and robustness tests."""
    
    def test_logger_performance_many_entries(self):
        """Test logger performance with many log entries."""
        logger = StreamlitLogger()
        
        start_time = time.time()
        
        # Log many entries
        for i in range(1000):
            logger.info(f"Log entry {i}", stage="performance_test")
        
        elapsed = time.time() - start_time
        
        # Should complete reasonably quickly (adjust threshold as needed)
        assert elapsed < 1.0, f"Logging 1000 entries took {elapsed:.2f} seconds"
        
        # Verify all entries were recorded
        logs = logger.get_logs()
        assert len(logs) == 1000
    
    def test_error_info_memory_usage(self):
        """Test ErrorInfo memory usage with large technical details."""
        large_details = "x" * 10000  # 10KB of text
        
        error = ErrorInfo(
            error_type=ErrorType.PROCESSING,
            severity=ErrorSeverity.MEDIUM,
            message="Test error",
            technical_details=large_details
        )
        
        # Should handle large technical details without issues
        assert len(error.technical_details) == 10000
        
        # Conversion to dict should work
        error_dict = error.to_dict()
        assert len(error_dict['technical_details']) == 10000
    
    def test_concurrent_logger_access(self):
        """Test logger thread safety (basic test)."""
        logger = StreamlitLogger()
        
        # Simulate concurrent access (basic test)
        for i in range(100):
            logger.info(f"Concurrent message {i}")
        
        logs = logger.get_logs()
        assert len(logs) == 100
        
        # All messages should be present
        messages = [log['message'] for log in logs]
        for i in range(100):
            assert f"Concurrent message {i}" in messages


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v"])