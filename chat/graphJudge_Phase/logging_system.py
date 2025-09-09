"""
Logging system for the GraphJudge system.

This module provides comprehensive logging functionality including
terminal logging with file capture and structured logging utilities.
"""

import os
import logging
from datetime import datetime
from typing import Optional
from .config import LOG_DIR


def setup_terminal_logging() -> str:
    """
    Set up terminal logging to capture output to a timestamped log file.
    
    Creates the log directory if it doesn't exist and generates a
    timestamped log filename for the current session.
    
    Returns:
        str: Path to the created log file
    """
    # Create log directory if it doesn't exist
    os.makedirs(LOG_DIR, exist_ok=True)
    
    # Create timestamped log filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f"run_gj_log_{timestamp}.txt"
    log_filepath = os.path.join(LOG_DIR, log_filename)
    
    return log_filepath


class TerminalLogger:
    """
    Simple terminal logger that captures output to file.
    
    This class provides a comprehensive logging solution that:
    1. Captures all print statements to a log file
    2. Maintains console output for real-time monitoring
    3. Provides structured logging with timestamps and levels
    4. Handles session management with start/end markers
    """
    
    def __init__(self, log_filepath: str):
        """
        Initialize the terminal logger.
        
        Args:
            log_filepath (str): Path to the log file
        """
        self.log_filepath = log_filepath
        self.original_print = print
        
        # Initialize log file with header
        self.write_to_log("=" * 80)
        self.write_to_log(f"Perplexity API Graph Judge Execution Log")
        self.write_to_log(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        self.write_to_log("=" * 80)
        
        # Replace print function with logged version
        import builtins
        builtins.print = self.logged_print
        
    def write_to_log(self, message: str) -> None:
        """
        Write a message directly to the log file.
        
        Args:
            message (str): Message to write to log
        """
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_entry = f"[{timestamp}] {message}\n"
        
        try:
            with open(self.log_filepath, 'a', encoding='utf-8') as f:
                f.write(log_entry)
        except Exception as e:
            # If logging fails, at least print to console
            self.original_print(f"Logging error: {e}")
    
    def logged_print(self, *args, **kwargs) -> None:
        """
        Custom print function that logs to file and prints to console.
        
        This function replaces the built-in print function to ensure
        all output is captured in the log file while maintaining
        console output for real-time monitoring.
        
        Args:
            *args: Arguments to print
            **kwargs: Keyword arguments for print
        """
        # Convert arguments to string message
        message = ' '.join(str(arg) for arg in args)
        
        # Write to log file
        self.write_to_log(message)
        
        # Print to console using original print
        self.original_print(*args, **kwargs)
    
    def log_message(self, message: str, level: str = "INFO") -> None:
        """
        Log a message with level indicator.
        
        Args:
            message (str): Message to log
            level (str): Log level (INFO, WARNING, ERROR, etc.)
        """
        log_entry = f"[{level}] {message}"
        self.write_to_log(log_entry)
        self.original_print(log_entry)
    
    def start_session(self) -> None:
        """Start a new logging session."""
        self.log_message("🎯 Perplexity API Graph Judge - Session Started", "INFO")
    
    def end_session(self) -> None:
        """End the logging session."""
        self.log_message("🎯 Perplexity API Graph Judge - Session Ended", "INFO")
        self.write_to_log("=" * 80)
        
        # Restore original print function
        import builtins
        builtins.print = self.original_print
    
    def log_error(self, error: Exception, context: str = "") -> None:
        """
        Log an error with context information.
        
        Args:
            error (Exception): The error that occurred
            context (str): Additional context about the error
        """
        error_message = f"ERROR in {context}: {str(error)}" if context else f"ERROR: {str(error)}"
        self.log_message(error_message, "ERROR")
    
    def log_warning(self, message: str) -> None:
        """
        Log a warning message.
        
        Args:
            message (str): Warning message
        """
        self.log_message(message, "WARNING")
    
    def log_info(self, message: str) -> None:
        """
        Log an info message.
        
        Args:
            message (str): Info message
        """
        self.log_message(message, "INFO")
    
    def log_debug(self, message: str) -> None:
        """
        Log a debug message.
        
        Args:
            message (str): Debug message
        """
        self.log_message(message, "DEBUG")
    
    def log_section(self, title: str) -> None:
        """
        Log a section header for better organization.
        
        Args:
            title (str): Section title
        """
        self.write_to_log("")
        self.write_to_log("-" * 60)
        self.write_to_log(f"Section: {title}")
        self.write_to_log("-" * 60)
        self.original_print(f"\n{'='*20} {title} {'='*20}")
    
    def log_statistics(self, stats: dict) -> None:
        """
        Log statistics in a formatted way.
        
        Args:
            stats (dict): Dictionary of statistics to log
        """
        self.write_to_log("📊 Statistics:")
        for key, value in stats.items():
            if isinstance(value, float):
                formatted_value = f"{value:.2f}"
            else:
                formatted_value = str(value)
            self.write_to_log(f"   - {key}: {formatted_value}")
        self.original_print("📊 Statistics logged to file")
    
    def get_log_filepath(self) -> str:
        """
        Get the current log file path.
        
        Returns:
            str: Path to the current log file
        """
        return self.log_filepath


def create_logger(name: str, log_filepath: Optional[str] = None) -> logging.Logger:
    """
    Create a standard Python logger with file and console handlers.
    
    Args:
        name (str): Logger name
        log_filepath (Optional[str]): Path to log file (auto-generated if None)
        
    Returns:
        logging.Logger: Configured logger instance
    """
    if log_filepath is None:
        log_filepath = setup_terminal_logging()
    
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    
    # Create formatters
    file_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    console_formatter = logging.Formatter(
        '%(levelname)s - %(message)s'
    )
    
    # Create file handler
    file_handler = logging.FileHandler(log_filepath, encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(file_formatter)
    
    # Create console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(console_formatter)
    
    # Add handlers to logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger
