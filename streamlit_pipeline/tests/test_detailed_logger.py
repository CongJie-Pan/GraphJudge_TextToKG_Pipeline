#!/usr/bin/env python3
"""
Comprehensive tests for DetailedLogger functionality.

This test module ensures the logging system works correctly across all
pipeline stages and handles various error conditions gracefully.
"""

import pytest
import tempfile
import shutil
from pathlib import Path
from datetime import datetime
import json
import os

from streamlit_pipeline.utils.detailed_logger import DetailedLogger, LogLevel


class TestDetailedLogger:
    """Test suite for DetailedLogger class."""

    def setup_method(self):
        """Set up test fixtures before each test method."""
        # Create a temporary directory for test logs
        self.test_logs_dir = tempfile.mkdtemp()

        # Override the default logs directory
        self.original_logs_dir = None

    def teardown_method(self):
        """Clean up test fixtures after each test method."""
        # Clean up temporary directory
        if os.path.exists(self.test_logs_dir):
            shutil.rmtree(self.test_logs_dir)

    def test_logger_initialization(self):
        """Test that DetailedLogger initializes correctly."""
        logger = DetailedLogger(phase="test_phase")

        assert logger.phase == "test_phase"
        assert logger.logs_dir.exists()
        assert logger.session_id is not None
        assert len(logger.log_entries) == 2  # Initialization messages

    def test_date_based_directory_creation(self):
        """Test that logs are stored in date-based directories."""
        logger = DetailedLogger(phase="test_phase")

        expected_date = datetime.now().strftime("%Y_%m_%d")
        assert expected_date in str(logger.logs_dir)
        assert "test_phase" in str(logger.logs_dir)

    def test_log_debug_method(self):
        """Test log_debug method functionality."""
        logger = DetailedLogger(phase="test_phase")

        test_details = {"key": "value", "count": 42}
        logger.log_debug("TEST", "Debug message", test_details)

        # Check that log entry was created
        assert len(logger.log_entries) == 3  # 2 init + 1 debug

        # Check log file content
        debug_file = logger.logs_dir / f"{logger.phase}_{logger.session_id}.log"
        assert debug_file.exists()

        content = debug_file.read_text(encoding='utf-8')
        assert "DEBUG" in content
        assert "Debug message" in content

    def test_log_info_method(self):
        """Test log_info method functionality."""
        logger = DetailedLogger(phase="test_phase")

        test_details = {"entity_count": 5, "processing_time": 1.23}
        logger.log_info("ENTITY", "Entity extraction completed", test_details)

        # Check that log entry was created
        debug_file = logger.logs_dir / f"{logger.phase}_{logger.session_id}.log"
        content = debug_file.read_text(encoding='utf-8')

        assert "INFO" in content
        assert "Entity extraction completed" in content
        assert "entity_count" in content

    def test_log_warning_method(self):
        """Test log_warning method functionality."""
        logger = DetailedLogger(phase="test_phase")

        logger.log_warning("API", "API rate limit approaching")

        # Check warning file was created
        warning_file = logger.logs_dir / f"{logger.phase}_{logger.session_id}_warnings.log"
        assert warning_file.exists()

        content = warning_file.read_text(encoding='utf-8')
        assert "WARNING" in content
        assert "API rate limit approaching" in content

    def test_log_error_method(self):
        """Test log_error method functionality."""
        logger = DetailedLogger(phase="test_phase")

        test_details = {"error_type": "CONNECTION_FAILED", "retry_count": 3}
        logger.log_error("API", "Failed to connect to API", test_details)

        # Check error file was created
        error_file = logger.logs_dir / f"{logger.phase}_{logger.session_id}_errors.log"
        assert error_file.exists()

        content = error_file.read_text(encoding='utf-8')
        assert "ERROR" in content
        assert "Failed to connect to API" in content
        assert "error_type" in content

    def test_log_pipeline_start(self):
        """Test log_pipeline_start method functionality."""
        logger = DetailedLogger(phase="test_phase")

        config = {"model": "gpt-5-mini", "temperature": 0.7}
        logger.log_pipeline_start(config)

        debug_file = logger.logs_dir / f"{logger.phase}_{logger.session_id}.log"
        content = debug_file.read_text(encoding='utf-8')

        assert "Pipeline execution started" in content
        assert "gpt-5-mini" in content

    def test_log_pipeline_complete(self):
        """Test log_pipeline_complete method functionality."""
        logger = DetailedLogger(phase="test_phase")

        results = {"entities_count": 15, "triples_count": 42, "success": True}
        logger.log_pipeline_complete(results)

        debug_file = logger.logs_dir / f"{logger.phase}_{logger.session_id}.log"
        content = debug_file.read_text(encoding='utf-8')

        assert "Pipeline execution completed" in content
        assert "entities_count" in content

    def test_log_phase_start(self):
        """Test log_phase_start method functionality."""
        logger = DetailedLogger(phase="test_phase")

        config = {"chunk_size": 1000}
        logger.log_phase_start("entity_extraction", config)

        debug_file = logger.logs_dir / f"{logger.phase}_{logger.session_id}.log"
        content = debug_file.read_text(encoding='utf-8')

        assert "Starting entity_extraction" in content
        assert "chunk_size" in content

    def test_log_phase_complete(self):
        """Test log_phase_complete method functionality."""
        logger = DetailedLogger(phase="test_phase")

        results = {"processing_time": 2.5, "entity_count": 10}
        logger.log_phase_complete("entity_extraction", True, results)

        debug_file = logger.logs_dir / f"{logger.phase}_{logger.session_id}.log"
        content = debug_file.read_text(encoding='utf-8')

        assert "Completed entity_extraction - Success: True" in content
        assert "processing_time" in content

    def test_log_api_call_success(self):
        """Test log_api_call method for successful API calls."""
        logger = DetailedLogger(phase="test_phase")

        logger.log_api_call("GPT-5-mini", 500, 150, success=True)

        debug_file = logger.logs_dir / f"{logger.phase}_{logger.session_id}.log"
        content = debug_file.read_text(encoding='utf-8')

        assert "API call to GPT-5-mini" in content
        assert "input_tokens: 500" in content
        assert "output_tokens: 150" in content
        assert "success: true" in content

    def test_log_api_call_failure(self):
        """Test log_api_call method for failed API calls."""
        logger = DetailedLogger(phase="test_phase")

        logger.log_api_call("GPT-5-mini", 500, 0, success=False, error="Rate limit exceeded")

        # Should create both debug log and error log entries
        debug_file = logger.logs_dir / f"{logger.phase}_{logger.session_id}.log"
        error_file = logger.logs_dir / f"{logger.phase}_{logger.session_id}_errors.log"

        debug_content = debug_file.read_text(encoding='utf-8')
        assert "API call to GPT-5-mini" in debug_content
        assert "success: false" in debug_content

        error_content = error_file.read_text(encoding='utf-8')
        assert "Rate limit exceeded" in error_content

    def test_log_storage_operation(self):
        """Test log_storage_operation method functionality."""
        logger = DetailedLogger(phase="test_phase")

        logger.log_storage_operation("save", "/path/to/file.json", success=True, size_bytes=2048)

        debug_file = logger.logs_dir / f"{logger.phase}_{logger.session_id}.log"
        content = debug_file.read_text(encoding='utf-8')

        assert "Storage operation: save" in content
        assert "/path/to/file.json" in content
        assert "size_bytes: 2048" in content

    def test_log_data_flow(self):
        """Test log_data_flow method functionality."""
        logger = DetailedLogger(phase="test_phase")

        data_info = {"input_size": 1000, "output_size": 800, "transformation": "text_cleaning"}
        logger.log_data_flow("preprocessing", data_info)

        debug_file = logger.logs_dir / f"{logger.phase}_{logger.session_id}.log"
        content = debug_file.read_text(encoding='utf-8')

        assert "Data flow for preprocessing" in content
        assert "transformation" in content

    def test_log_processing_stats(self):
        """Test log_processing_stats method functionality."""
        logger = DetailedLogger(phase="test_phase")

        stats = {
            "total_items": 100,
            "processed_items": 95,
            "failed_items": 5,
            "processing_rate": 10.5
        }
        logger.log_processing_stats("entity_extraction", stats)

        debug_file = logger.logs_dir / f"{logger.phase}_{logger.session_id}.log"
        content = debug_file.read_text(encoding='utf-8')

        assert "Processing stats for entity_extraction" in content
        assert "total_items: 100" in content
        assert "processing_rate: 10.5" in content

    def test_multiple_loggers_same_phase(self):
        """Test that multiple loggers for the same phase work correctly."""
        logger1 = DetailedLogger(phase="test_phase")
        logger2 = DetailedLogger(phase="test_phase")

        # Should have different session IDs
        assert logger1.session_id != logger2.session_id

        # Should create separate log files
        logger1.log_info("TEST", "Logger 1 message")
        logger2.log_info("TEST", "Logger 2 message")

        # Check both log files exist and contain correct content
        log1_file = logger1.logs_dir / f"{logger1.phase}_{logger1.session_id}.log"
        log2_file = logger2.logs_dir / f"{logger2.phase}_{logger2.session_id}.log"

        assert log1_file.exists()
        assert log2_file.exists()

        content1 = log1_file.read_text(encoding='utf-8')
        content2 = log2_file.read_text(encoding='utf-8')

        assert "Logger 1 message" in content1
        assert "Logger 1 message" not in content2
        assert "Logger 2 message" in content2
        assert "Logger 2 message" not in content1

    def test_unicode_handling(self):
        """Test that logger handles Unicode characters correctly."""
        logger = DetailedLogger(phase="test_phase")

        # Test with Chinese characters
        chinese_text = "測試中文字符處理"
        details = {"chinese_text": chinese_text, "length": len(chinese_text)}

        logger.log_info("UNICODE", "Unicode test message", details)

        debug_file = logger.logs_dir / f"{logger.phase}_{logger.session_id}.log"
        content = debug_file.read_text(encoding='utf-8')

        assert chinese_text in content
        assert "Unicode test message" in content

    def test_large_data_logging(self):
        """Test logging with large data structures."""
        logger = DetailedLogger(phase="test_phase")

        # Create a large details dictionary
        large_details = {
            "entities": [f"entity_{i}" for i in range(100)],
            "metadata": {"processing_notes": "x" * 1000}
        }

        logger.log_info("LARGE_DATA", "Processing large dataset", large_details)

        debug_file = logger.logs_dir / f"{logger.phase}_{logger.session_id}.log"
        assert debug_file.exists()

        content = debug_file.read_text(encoding='utf-8')
        assert "Processing large dataset" in content
        assert "entity_0" in content
        assert "entity_99" in content

    def test_close_logger(self):
        """Test that logger closes correctly and generates summary."""
        logger = DetailedLogger(phase="test_phase")

        # Add some log entries
        logger.log_info("TEST", "Test message 1")
        logger.log_warning("TEST", "Test warning")
        logger.log_error("TEST", "Test error", {"error_code": 500})

        # Close the logger
        logger.close()

        # Check that summary file was created
        summary_file = logger.logs_dir / f"{logger.phase}_{logger.session_id}_summary.json"
        assert summary_file.exists()

        # Check summary content
        with open(summary_file, 'r', encoding='utf-8') as f:
            summary = json.load(f)

        assert summary["phase"] == "test_phase"
        assert summary["session_id"] == logger.session_id
        assert "start_time" in summary
        assert "end_time" in summary
        assert "duration" in summary
        assert summary["total_entries"] == 5  # 2 init + 3 test entries

    def test_error_handling_in_logging(self):
        """Test that logger handles its own errors gracefully."""
        logger = DetailedLogger(phase="test_phase")

        # Test with invalid details (should not crash)
        invalid_details = {"circular_ref": None}
        invalid_details["circular_ref"] = invalid_details

        try:
            logger.log_info("ERROR_TEST", "Testing error handling", invalid_details)
            # Should not raise an exception
            assert True
        except Exception as e:
            pytest.fail(f"Logger should handle invalid details gracefully, but raised: {e}")


class TestLoggerIntegration:
    """Integration tests for DetailedLogger with pipeline components."""

    def setup_method(self):
        """Set up test fixtures."""
        self.test_logs_dir = tempfile.mkdtemp()

    def teardown_method(self):
        """Clean up test fixtures."""
        if os.path.exists(self.test_logs_dir):
            shutil.rmtree(self.test_logs_dir)

    def test_ectd_phase_logging_pattern(self):
        """Test typical logging pattern for ECTD phase."""
        logger = DetailedLogger(phase="ectd")

        # Simulate ECTD phase logging
        config = {"model": "gpt-5-mini", "temperature": 0.7}
        logger.log_pipeline_start(config)

        # Entity extraction phase
        logger.log_phase_start("entity_extraction", {"text_length": 5000})
        logger.log_info("ENTITY", "Starting entity extraction from text", {
            "text_length": 5000,
            "text_preview": "Sample text..."
        })
        logger.log_api_call("GPT-5-mini", 500, 150, success=True)
        logger.log_phase_complete("entity_extraction", True, {
            "entity_count": 25,
            "processing_time": 2.1
        })

        # Text denoising phase
        logger.log_phase_start("text_denoising", {"entity_count": 25})
        logger.log_api_call("GPT-5-mini", 600, 200, success=True)
        logger.log_phase_complete("text_denoising", True, {
            "original_length": 5000,
            "denoised_length": 4200,
            "processing_time": 1.8
        })

        logger.log_pipeline_complete({
            "entities_extracted": 25,
            "text_denoised": True,
            "total_processing_time": 3.9
        })

        # Verify log files were created correctly
        debug_file = logger.logs_dir / f"ectd_{logger.session_id}.log"
        assert debug_file.exists()

        content = debug_file.read_text(encoding='utf-8')
        assert "entity_extraction" in content
        assert "text_denoising" in content
        assert "entity_count: 25" in content

    def test_triple_generation_phase_logging_pattern(self):
        """Test typical logging pattern for triple generation phase."""
        logger = DetailedLogger(phase="triple_gen")

        # Simulate triple generation phase logging
        logger.log_phase_start("triple_generation", {
            "denoised_text_length": 4200,
            "chunk_size": 1000
        })

        # Log chunking process
        logger.log_info("TRIPLE", "Text chunked for processing", {
            "total_chunks": 5,
            "chunk_sizes": [1000, 1000, 1000, 1000, 200]
        })

        # Log API calls for each chunk
        for i in range(5):
            logger.log_api_call("GPT-5-mini", 800, 300, success=True)
            logger.log_info("TRIPLE", f"Processed chunk {i+1}", {
                "chunk_id": i+1,
                "triples_generated": 8
            })

        logger.log_phase_complete("triple_generation", True, {
            "total_triples": 40,
            "processing_time": 15.2
        })

        # Verify logging
        debug_file = logger.logs_dir / f"triple_gen_{logger.session_id}.log"
        content = debug_file.read_text(encoding='utf-8')
        assert "Text chunked for processing" in content
        assert "total_triples: 40" in content

    def test_concurrent_phase_logging(self):
        """Test logging from multiple phases simultaneously."""
        ectd_logger = DetailedLogger(phase="ectd")
        triple_logger = DetailedLogger(phase="triple_gen")
        gj_logger = DetailedLogger(phase="graph_judge")

        # Log from different phases
        ectd_logger.log_info("ENTITY", "ECTD phase message")
        triple_logger.log_info("TRIPLE", "Triple generation phase message")
        gj_logger.log_info("JUDGE", "Graph judge phase message")

        # Verify separate log files were created
        ectd_file = ectd_logger.logs_dir / f"ectd_{ectd_logger.session_id}.log"
        triple_file = triple_logger.logs_dir / f"triple_gen_{triple_logger.session_id}.log"
        gj_file = gj_logger.logs_dir / f"graph_judge_{gj_logger.session_id}.log"

        assert ectd_file.exists()
        assert triple_file.exists()
        assert gj_file.exists()

        # Verify content separation
        ectd_content = ectd_file.read_text(encoding='utf-8')
        triple_content = triple_file.read_text(encoding='utf-8')
        gj_content = gj_file.read_text(encoding='utf-8')

        assert "ECTD phase message" in ectd_content
        assert "ECTD phase message" not in triple_content
        assert "ECTD phase message" not in gj_content

        assert "Triple generation phase message" in triple_content
        assert "Graph judge phase message" in gj_content


if __name__ == "__main__":
    pytest.main([__file__, "-v"])