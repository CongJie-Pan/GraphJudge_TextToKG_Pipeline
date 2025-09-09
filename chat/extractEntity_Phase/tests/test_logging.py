"""
Tests for logging system.
"""

import pytest
import os
import tempfile
from datetime import datetime
from unittest.mock import patch, mock_open, MagicMock
from extractEntity_Phase.infrastructure.logging import (
    LogEntry, LogFormatter, BaseLogger, NullLogger,
    TerminalLogger, Logger, get_logger, set_logger,
    temporary_logger
)


class TestLogEntry:
    """Test log entry dataclass."""
    
    def test_log_entry_creation(self):
        """Test log entry creation with all fields."""
        timestamp = datetime.now()
        entry = LogEntry(
            level="INFO",
            message="Test log message",
            timestamp=timestamp,
            module="test_module",
            function="test_function",
            line_number=42
        )
        assert entry.level == "INFO"
        assert entry.message == "Test log message"
        assert entry.timestamp == timestamp
        assert entry.module == "test_module"
        assert entry.function == "test_function"
        assert entry.line_number == 42
    
    def test_log_entry_defaults(self):
        """Test log entry default values."""
        entry = LogEntry(level="INFO", message="Test message")
        assert entry.level == "INFO"
        assert entry.message == "Test message"
        assert entry.timestamp is not None
        assert entry.module is None
        assert entry.function is None
        assert entry.line_number is None
    
    def test_log_entry_string_representation(self):
        """Test log entry string representation."""
        entry = LogEntry(level="INFO", message="Test message")
        entry_str = str(entry)
        assert "INFO" in entry_str
        assert "Test message" in entry_str
        assert "test_logging.py" in entry_str  # Current module name


class TestLogFormatter:
    """Test log formatter class."""
    
    def test_log_formatter_creation(self):
        """Test log formatter creation."""
        formatter = LogFormatter()
        assert formatter is not None
    
    def test_format_log_entry(self):
        """Test log entry formatting."""
        formatter = LogFormatter()
        entry = LogEntry(
            level="INFO",
            message="Test message",
            timestamp=datetime(2025, 1, 27, 10, 0, 0),
            module="test_module",
            function="test_function",
            line_number=42
        )
        
        formatted = formatter.format(entry)
        assert "INFO" in formatted
        assert "Test message" in formatted
        assert "test_module" in formatted
        assert "test_function" in formatted
        assert "42" in formatted
        assert "2025-01-27" in formatted
    
    def test_format_log_entry_minimal(self):
        """Test log entry formatting with minimal fields."""
        formatter = LogFormatter()
        entry = LogEntry(level="ERROR", message="Error message")
        
        formatted = formatter.format(entry)
        assert "ERROR" in formatted
        assert "Error message" in formatted
        assert "None" not in formatted  # Should not show None values
    
    def test_format_log_entry_with_colors(self):
        """Test log entry formatting with colors."""
        formatter = LogFormatter(use_colors=True)
        entry = LogEntry(level="ERROR", message="Error message")
        
        formatted = formatter.format(entry)
        assert "ERROR" in formatted
        assert "Error message" in formatted
        # Note: Color codes may not be visible in test output


class TestBaseLogger:
    """Test base logger abstract class."""
    
    def test_base_logger_instantiation(self):
        """Test that base logger cannot be instantiated."""
        with pytest.raises(TypeError):
            BaseLogger()
    
    def test_base_logger_abstract_methods(self):
        """Test that base logger has required abstract methods."""
        # Check that required methods exist
        assert hasattr(BaseLogger, 'log')
        assert hasattr(BaseLogger, 'info')
        assert hasattr(BaseLogger, 'warning')
        assert hasattr(BaseLogger, 'error')
        assert hasattr(BaseLogger, 'debug')
        assert hasattr(BaseLogger, 'success')


class TestNullLogger:
    """Test null logger for testing environments."""
    
    def test_null_logger_creation(self):
        """Test null logger creation."""
        logger = NullLogger()
        assert logger is not None
    
    def test_null_logger_methods(self):
        """Test null logger methods do nothing."""
        logger = NullLogger()
        
        # All methods should return None and not raise exceptions
        assert logger.log("INFO", "Test message") is None
        assert logger.info("Test info") is None
        assert logger.warning("Test warning") is None
        assert logger.error("Test error") is None
        assert logger.debug("Test debug") is None
        assert logger.success("Test success") is None
    
    def test_null_logger_context_manager(self):
        """Test null logger context manager behavior."""
        logger = NullLogger()
        
        with logger:
            # Should not raise exceptions
            logger.info("Test message")
        
        # Should not raise exceptions after context
        logger.info("Test message after context")


class TestTerminalLogger:
    """Test terminal logger for active logging."""
    
    def test_terminal_logger_creation(self):
        """Test terminal logger creation."""
        logger = TerminalLogger()
        assert logger is not None
    
    @patch('builtins.print')
    def test_terminal_logger_log_method(self, mock_print):
        """Test terminal logger log method."""
        logger = TerminalLogger()
        logger.log("INFO", "Test message")
        
        # Verify print was called
        mock_print.assert_called_once()
        call_args = mock_print.call_args[0][0]
        assert "INFO" in call_args
        assert "Test message" in call_args
    
    @patch('builtins.print')
    def test_terminal_logger_info_method(self, mock_print):
        """Test terminal logger info method."""
        logger = TerminalLogger()
        logger.info("Test info message")
        
        mock_print.assert_called_once()
        call_args = mock_print.call_args[0][0]
        assert "INFO" in call_args
        assert "Test info message" in call_args
    
    @patch('builtins.print')
    def test_terminal_logger_warning_method(self, mock_print):
        """Test terminal logger warning method."""
        logger = TerminalLogger()
        logger.warning("Test warning message")
        
        mock_print.assert_called_once()
        call_args = mock_print.call_args[0][0]
        assert "WARNING" in call_args
        assert "Test warning message" in call_args
    
    @patch('builtins.print')
    def test_terminal_logger_error_method(self, mock_print):
        """Test terminal logger error method."""
        logger = TerminalLogger()
        logger.error("Test error message")
        
        mock_print.assert_called_once()
        call_args = mock_print.call_args[0][0]
        assert "ERROR" in call_args
        assert "Test error message" in call_args
    
    @patch('builtins.print')
    def test_terminal_logger_debug_method(self, mock_print):
        """Test terminal logger debug method."""
        logger = TerminalLogger()
        logger.debug("Test debug message")
        
        mock_print.assert_called_once()
        call_args = mock_print.call_args[0][0]
        assert "DEBUG" in call_args
        assert "Test debug message" in call_args
    
    @patch('builtins.print')
    def test_terminal_logger_success_method(self, mock_print):
        """Test terminal logger success method."""
        logger = TerminalLogger()
        logger.success("Test success message")
        
        mock_print.assert_called_once()
        call_args = mock_print.call_args[0][0]
        assert "SUCCESS" in call_args
        assert "Test success message" in call_args
    
    def test_terminal_logger_context_manager(self):
        """Test terminal logger context manager behavior."""
        logger = TerminalLogger()
        
        with logger:
            # Should not raise exceptions
            logger.info("Test message")
        
        # Should not raise exceptions after context
        logger.info("Test message after context")


class TestLogger:
    """Test main logger class."""
    
    def test_logger_creation(self):
        """Test logger creation."""
        logger = Logger()
        assert logger is not None
        assert hasattr(logger, 'logger')
        assert hasattr(logger, 'log_file')
    
    def test_logger_creation_with_config(self):
        """Test logger creation with configuration."""
        from extractEntity_Phase.infrastructure.config import LoggingConfig
        
        config = LoggingConfig(
            level="DEBUG",
            log_file="test.log",
            max_file_size_mb=10,
            backup_count=5
        )
        
        logger = Logger(config=config)
        assert logger.log_file == "test.log"
    
    @patch('extractEntity_Phase.infrastructure.logging.os.getenv')
    def test_logger_creation_testing_environment(self, mock_getenv):
        """Test logger creation in testing environment."""
        mock_getenv.return_value = "test"
        
        logger = Logger()
        assert isinstance(logger.logger, NullLogger)
    
    @patch('extractEntity_Phase.infrastructure.logging.os.getenv')
    def test_logger_creation_production_environment(self, mock_getenv):
        """Test logger creation in production environment."""
        mock_getenv.return_value = "production"
        
        logger = Logger()
        assert isinstance(logger.logger, TerminalLogger)
    
    @patch('builtins.open', new_callable=mock_open)
    def test_logger_file_creation(self, mock_file):
        """Test logger file creation."""
        logger = Logger()
        
        # Trigger file creation
        with logger:
            logger.info("Test message")
        
        # Verify file was opened
        mock_file.assert_called()
    
    @patch('builtins.open', new_callable=mock_open)
    def test_logger_file_write(self, mock_file):
        """Test logger file writing."""
        mock_file.return_value.write = MagicMock()
        
        logger = Logger()
        
        with logger:
            logger.info("Test message")
        
        # Verify write was called
        mock_file.return_value.write.assert_called()
    
    @patch('builtins.open', new_callable=mock_open)
    def test_logger_file_close(self, mock_file):
        """Test logger file closing."""
        mock_file.return_value.close = MagicMock()
        
        logger = Logger()
        
        with logger:
            logger.info("Test message")
        
        # Verify close was called
        mock_file.return_value.close.assert_called()
    
    def test_logger_context_manager(self):
        """Test logger context manager behavior."""
        logger = Logger()
        
        with logger:
            # Should not raise exceptions
            logger.info("Test message")
            logger.warning("Test warning")
            logger.error("Test error")
        
        # Should not raise exceptions after context
        logger.info("Test message after context")
    
    def test_logger_convenience_methods(self):
        """Test logger convenience methods."""
        logger = Logger()
        
        # All convenience methods should work
        logger.info("Test info")
        logger.warning("Test warning")
        logger.error("Test error")
        logger.debug("Test debug")
        logger.success("Test success")
    
    def test_logger_log_method(self):
        """Test logger log method."""
        logger = Logger()
        
        # Test with custom level
        logger.log("CUSTOM", "Custom level message")
        
        # Test with standard levels
        logger.log("INFO", "Info message")
        logger.log("WARNING", "Warning message")
        logger.log("ERROR", "Error message")


class TestLoggerFunctions:
    """Test logger utility functions."""
    
    def test_get_logger_singleton(self):
        """Test get_logger singleton behavior."""
        # Clear any existing logger
        set_logger(None)
        
        logger1 = get_logger()
        logger2 = get_logger()
        
        assert logger1 is logger2
        assert isinstance(logger1, Logger)
    
    def test_set_logger(self):
        """Test set_logger function."""
        # Create a custom logger
        custom_logger = Logger()
        
        # Set the custom logger
        set_logger(custom_logger)
        
        # Get the logger and verify it's the custom one
        retrieved_logger = get_logger()
        assert retrieved_logger is custom_logger
    
    def test_set_logger_none(self):
        """Test set_logger with None."""
        # Set logger to None
        set_logger(None)
        
        # Should create a new default logger
        logger = get_logger()
        assert isinstance(logger, Logger)
    
    def test_temporary_logger(self):
        """Test temporary_logger context manager."""
        with temporary_logger() as temp_logger:
            assert isinstance(temp_logger, Logger)
            temp_logger.info("Test message")
        
        # Should not affect the global logger
        global_logger = get_logger()
        assert global_logger is not temp_logger


class TestErrorHandling:
    """Test error handling scenarios."""
    
    @patch('builtins.open')
    def test_logger_file_creation_error(self, mock_open):
        """Test logger handling of file creation errors."""
        mock_open.side_effect = IOError("Permission denied")
        
        logger = Logger()
        
        # Should handle file creation errors gracefully
        with logger:
            logger.info("Test message")
        
        # Should not raise exceptions
    
    @patch('builtins.open', new_callable=mock_open)
    def test_logger_file_write_error(self, mock_file):
        """Test logger handling of file write errors."""
        mock_file.return_value.write.side_effect = IOError("Disk full")
        
        logger = Logger()
        
        # Should handle file write errors gracefully
        with logger:
            logger.info("Test message")
        
        # Should not raise exceptions
    
    @patch('builtins.open', new_callable=mock_open)
    def test_logger_file_close_error(self, mock_file):
        """Test logger handling of file close errors."""
        mock_file.return_value.close.side_effect = IOError("File already closed")
        
        logger = Logger()
        
        # Should handle file close errors gracefully
        with logger:
            logger.info("Test message")
        
        # Should not raise exceptions
    
    def test_logger_invalid_level(self):
        """Test logger handling of invalid log levels."""
        logger = Logger()
        
        # Should handle invalid levels gracefully
        logger.log("INVALID_LEVEL", "Test message")
        
        # Should not raise exceptions
    
    def test_logger_empty_message(self):
        """Test logger handling of empty messages."""
        logger = Logger()
        
        # Should handle empty messages gracefully
        logger.info("")
        logger.info(None)
        
        # Should not raise exceptions


class TestEdgeCases:
    """Test edge cases and boundary conditions."""
    
    def test_logger_empty_message(self):
        """Test logger with empty message."""
        logger = Logger()
        
        # Should handle empty string
        logger.info("")
        
        # Should handle None
        logger.info(None)
        
        # Should handle whitespace-only
        logger.info("   ")
    
    def test_logger_very_long_message(self):
        """Test logger with very long message."""
        logger = Logger()
        
        # Create very long message
        long_message = "A" * 10000
        
        # Should handle without errors
        logger.info(long_message)
    
    def test_logger_special_characters(self):
        """Test logger with special characters."""
        logger = Logger()
        
        # Test various special characters
        special_message = "!@#$%^&*()_+-=[]{}|;':\",./<>?"
        
        # Should handle without errors
        logger.info(special_message)
    
    def test_logger_unicode_characters(self):
        """Test logger with unicode characters."""
        logger = Logger()
        
        # Test Chinese characters
        chinese_message = "賈寶玉在大觀園中讀書"
        
        # Should handle without errors
        logger.info(chinese_message)
    
    def test_logger_multiple_contexts(self):
        """Test logger with multiple context managers."""
        logger = Logger()
        
        # Nested contexts
        with logger:
            logger.info("Outer context")
            with logger:
                logger.info("Inner context")
                logger.info("Another inner message")
            logger.info("Back to outer context")
        
        # Should not raise exceptions
    
    def test_logger_concurrent_access(self):
        """Test logger with concurrent-like access patterns."""
        logger = Logger()
        
        # Simulate rapid successive calls
        for i in range(100):
            logger.info(f"Message {i}")
        
        # Should handle without errors
    
    def test_logger_mixed_levels(self):
        """Test logger with mixed log levels."""
        logger = Logger()
        
        # Mix different levels
        logger.debug("Debug message")
        logger.info("Info message")
        logger.warning("Warning message")
        logger.error("Error message")
        logger.success("Success message")
        
        # Should handle all levels without errors

