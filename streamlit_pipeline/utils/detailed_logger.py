"""
Comprehensive Logging System for GraphJudge Streamlit Pipeline.

This module provides detailed logging capabilities for all pipeline stages,
storing logs in the streamlit_pipeline/logs directory with timestamped files
and comprehensive debugging information.
"""

import os
import sys
import json
import traceback
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List
from enum import Enum


class LogLevel(Enum):
    """Log levels for different types of messages."""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    SUCCESS = "SUCCESS"
    PIPELINE = "PIPELINE"
    API = "API"
    STORAGE = "STORAGE"


class DetailedLogger:
    """
    Comprehensive logger for the GraphJudge pipeline with file storage.

    Features:
    - Timestamped log files in logs/ directory
    - Multiple log levels (DEBUG, INFO, WARNING, ERROR, etc.)
    - JSON structured logs for easy parsing
    - Console and file output
    - Phase-specific logging
    - API call tracking
    - Error trace logging
    """

    def __init__(self, phase: str = "pipeline", create_subdirs: bool = True):
        """
        Initialize the detailed logger.

        Args:
            phase: The pipeline phase (ectd, triples, judgment, main)
            create_subdirs: Whether to create phase-specific subdirectories
        """
        self.phase = phase
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Create date-based directory structure
        current_dir = Path(__file__).parent.parent
        self.logs_base_dir = current_dir / "logs"
        self.logs_base_dir.mkdir(exist_ok=True)

        # Create date-specific subdirectory (format: YYYY_M_DD)
        date_folder = datetime.now().strftime("%Y_%m_%d")
        self.date_logs_dir = self.logs_base_dir / date_folder
        self.date_logs_dir.mkdir(exist_ok=True)

        # Create phase-specific subdirectory if requested
        if create_subdirs:
            self.logs_dir = self.date_logs_dir / phase
            self.logs_dir.mkdir(exist_ok=True)
        else:
            self.logs_dir = self.date_logs_dir

        # Create timestamped log files
        self.log_filename = f"{phase}_{self.session_id}.log"
        self.json_log_filename = f"{phase}_{self.session_id}.json"
        self.error_log_filename = f"{phase}_{self.session_id}_errors.log"

        self.log_file_path = self.logs_dir / self.log_filename
        self.json_log_path = self.logs_dir / self.json_log_filename
        self.error_log_path = self.logs_dir / self.error_log_filename

        # Initialize log files
        self._initialize_log_files()

        # Store log entries for summary
        self.log_entries: List[Dict[str, Any]] = []
        self.start_time = datetime.now()

        # Statistics tracking
        self.stats = {
            'total_logs': 0,
            'debug_logs': 0,
            'info_logs': 0,
            'warning_logs': 0,
            'error_logs': 0,
            'api_calls': 0,
            'storage_operations': 0,
            'pipeline_events': 0
        }

        self.log(LogLevel.INFO, f"Logger initialized for phase: {phase}")
        self.log(LogLevel.INFO, f"Log files created in: {self.logs_dir}")

    def _initialize_log_files(self):
        """Initialize log files with headers."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # Text log header
        header = f"""
{'='*80}
GraphJudge Streamlit Pipeline - Detailed Debug Log
Phase: {self.phase.upper()}
Session ID: {self.session_id}
Started: {timestamp}
Log File: {self.log_filename}
{'='*80}

"""

        with open(self.log_file_path, 'w', encoding='utf-8') as f:
            f.write(header)

        # JSON log initialization
        json_header = {
            "session_info": {
                "phase": self.phase,
                "session_id": self.session_id,
                "start_time": timestamp,
                "log_file": str(self.log_file_path),
                "json_file": str(self.json_log_path)
            },
            "entries": []
        }

        with open(self.json_log_path, 'w', encoding='utf-8') as f:
            json.dump(json_header, f, ensure_ascii=False, indent=2)

    def log(self, level: LogLevel, message: str, details: Dict[str, Any] = None,
            print_to_console: bool = True):
        """
        Log a message with specified level and optional details.

        Args:
            level: Log level (DEBUG, INFO, WARNING, ERROR, etc.)
            message: Main log message
            details: Additional structured details
            print_to_console: Whether to print to console
        """
        timestamp = datetime.now()
        timestamp_str = timestamp.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]

        # Create log entry
        log_entry = {
            "timestamp": timestamp_str,
            "level": level.value,
            "phase": self.phase,
            "message": message,
            "details": details or {},
            "session_id": self.session_id
        }

        # Add to internal storage
        self.log_entries.append(log_entry)
        self.stats['total_logs'] += 1
        self.stats[f'{level.value.lower()}_logs'] = self.stats.get(f'{level.value.lower()}_logs', 0) + 1

        # Format for text log
        details_str = ""
        if details:
            try:
                details_str = f" | Details: {json.dumps(details, ensure_ascii=False)}"
            except Exception:
                details_str = f" | Details: {str(details)}"

        text_log_line = f"[{timestamp_str}] [{level.value:8}] {message}{details_str}\n"

        # Write to text log
        with open(self.log_file_path, 'a', encoding='utf-8') as f:
            f.write(text_log_line)

        # Update JSON log
        self._update_json_log(log_entry)

        # Print to console if requested
        if print_to_console:
            # Use different formats for different levels (no emoji to avoid encoding issues)
            if level == LogLevel.ERROR:
                print(f"[ERROR] [{level.value}] {message}", file=sys.stderr)
            elif level == LogLevel.WARNING:
                print(f"[WARN] [{level.value}] {message}")
            elif level == LogLevel.SUCCESS:
                print(f"[SUCCESS] [{level.value}] {message}")
            elif level == LogLevel.DEBUG:
                print(f"[DEBUG] [{level.value}] {message}")
            elif level == LogLevel.API:
                print(f"[API] [{level.value}] {message}")
            elif level == LogLevel.STORAGE:
                print(f"[STORAGE] [{level.value}] {message}")
            elif level == LogLevel.PIPELINE:
                print(f"[PIPELINE] [{level.value}] {message}")
            else:
                print(f"[INFO] [{level.value}] {message}")

        # Special handling for errors
        if level == LogLevel.ERROR:
            self._log_error_details(message, details)

    def _update_json_log(self, log_entry: Dict[str, Any]):
        """Update the JSON log file with new entry."""
        try:
            with open(self.json_log_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            data['entries'].append(log_entry)

            with open(self.json_log_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            # Fallback - write to error log
            with open(self.error_log_path, 'a', encoding='utf-8') as f:
                f.write(f"[{datetime.now()}] Failed to update JSON log: {e}\n")

    def _log_error_details(self, message: str, details: Dict[str, Any] = None):
        """Log error details to separate error log file."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]

        error_info = {
            "timestamp": timestamp,
            "phase": self.phase,
            "error_message": message,
            "details": details or {},
            "traceback": traceback.format_exc() if sys.exc_info()[0] else None
        }

        with open(self.error_log_path, 'a', encoding='utf-8') as f:
            f.write(f"\n{'='*50}\n")
            f.write(f"ERROR at {timestamp}\n")
            f.write(f"Phase: {self.phase}\n")
            f.write(f"Message: {message}\n")
            if details:
                f.write(f"Details: {json.dumps(details, ensure_ascii=False, indent=2)}\n")
            if error_info["traceback"]:
                f.write(f"Traceback:\n{error_info['traceback']}\n")
            f.write(f"{'='*50}\n")

    def log_debug(self, category: str, message: str, details: Dict[str, Any] = None):
        """Log debug message with category."""
        self.log(LogLevel.DEBUG, f"[{category}] {message}", details)

    def log_info(self, category: str, message: str, details: Dict[str, Any] = None):
        """Log info message with category."""
        self.log(LogLevel.INFO, f"[{category}] {message}", details)

    def log_warning(self, category: str, message: str, details: Dict[str, Any] = None):
        """Log warning message with category."""
        self.log(LogLevel.WARNING, f"[{category}] {message}", details)

    def log_error(self, category: str, message: str, details: Dict[str, Any] = None):
        """Log error message with category."""
        self.log(LogLevel.ERROR, f"[{category}] {message}", details)

    def log_pipeline_start(self, details: Dict[str, Any] = None):
        """Log the start of the entire pipeline."""
        self.log(LogLevel.PIPELINE, "Pipeline execution started", details)
        self.stats['pipeline_events'] += 1

    def log_pipeline_complete(self, details: Dict[str, Any] = None):
        """Log the completion of the entire pipeline."""
        self.log(LogLevel.PIPELINE, "Pipeline execution completed", details)
        self.stats['pipeline_events'] += 1

    def log_phase_start(self, phase_name: str, description: str = ""):
        """Log the start of a pipeline phase."""
        self.log(LogLevel.PIPELINE, f"Starting {phase_name}", {
            "phase_name": phase_name,
            "description": description,
            "event": "phase_start"
        })
        self.stats['pipeline_events'] += 1

    def log_phase_end(self, phase_name: str, success: bool, duration: float = None):
        """Log the end of a pipeline phase."""
        self.log(LogLevel.PIPELINE, f"Completed {phase_name} - Success: {success}", {
            "phase_name": phase_name,
            "success": success,
            "duration": duration,
            "event": "phase_end"
        })

    def log_api_call(self, api_name: str, prompt_length: int, response_length: int = None,
                     success: bool = True, error: str = None):
        """Log API call details."""
        details = {
            "api_name": api_name,
            "prompt_length": prompt_length,
            "response_length": response_length,
            "success": success,
            "error": error
        }

        level = LogLevel.API if success else LogLevel.ERROR
        message = f"API call to {api_name}" + (" succeeded" if success else f" failed: {error}")

        self.log(level, message, details)
        self.stats['api_calls'] += 1

    def log_storage_operation(self, operation: str, file_path: str, success: bool = True,
                             error: str = None):
        """Log storage operations."""
        details = {
            "operation": operation,
            "file_path": file_path,
            "success": success,
            "error": error
        }

        level = LogLevel.STORAGE if success else LogLevel.ERROR
        message = f"Storage {operation}" + (" succeeded" if success else f" failed: {error}")

        self.log(level, message, details)
        self.stats['storage_operations'] += 1

    def log_data_flow(self, stage: str, input_data: Dict[str, Any], output_data: Dict[str, Any]):
        """Log detailed data flow between stages."""
        self.log(LogLevel.DEBUG, f"Data flow for {stage}", {
            "stage": stage,
            "input_data": input_data,
            "output_data": output_data,
            "event": "data_flow"
        })

    def log_processing_stats(self, stage: str, stats: Dict[str, Any]):
        """Log processing statistics for a stage."""
        self.log(LogLevel.INFO, f"Processing stats for {stage}", {
            "stage": stage,
            "statistics": stats,
            "event": "processing_stats"
        })

    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of all logged activities."""
        end_time = datetime.now()
        duration = (end_time - self.start_time).total_seconds()

        return {
            "session_info": {
                "phase": self.phase,
                "session_id": self.session_id,
                "start_time": self.start_time.isoformat(),
                "end_time": end_time.isoformat(),
                "duration_seconds": duration
            },
            "statistics": self.stats.copy(),
            "log_files": {
                "text_log": str(self.log_file_path),
                "json_log": str(self.json_log_path),
                "error_log": str(self.error_log_path)
            },
            "total_entries": len(self.log_entries)
        }

    def save_summary(self):
        """Save a summary report to the logs directory."""
        summary = self.get_summary()
        summary_file = self.logs_dir / f"{self.phase}_{self.session_id}_summary.json"

        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)

        self.log(LogLevel.INFO, f"Summary saved to {summary_file}")
        return summary_file

    def close(self):
        """Close the logger and save final summary."""
        self.log(LogLevel.INFO, f"Closing logger for phase: {self.phase}")

        # Save summary
        summary_file = self.save_summary()

        # Final stats
        duration = (datetime.now() - self.start_time).total_seconds()
        self.log(LogLevel.INFO, f"Session completed in {duration:.2f} seconds")
        self.log(LogLevel.INFO, f"Total log entries: {len(self.log_entries)}")

        return summary_file


# Global logger instances for different phases
_loggers: Dict[str, DetailedLogger] = {}


def get_phase_logger(phase: str) -> DetailedLogger:
    """
    Get or create a logger for a specific phase.

    Args:
        phase: Phase name (ectd, triples, judgment, main)

    Returns:
        DetailedLogger instance for the phase
    """
    if phase not in _loggers:
        _loggers[phase] = DetailedLogger(phase)

    return _loggers[phase]


def close_all_loggers():
    """Close all active loggers and save summaries."""
    summary_files = []
    for phase, logger in _loggers.items():
        summary_file = logger.close()
        summary_files.append(summary_file)

    _loggers.clear()
    return summary_files


def log_to_phase(phase: str, level: LogLevel, message: str, details: Dict[str, Any] = None):
    """Convenience function to log to a specific phase."""
    logger = get_phase_logger(phase)
    logger.log(level, message, details)