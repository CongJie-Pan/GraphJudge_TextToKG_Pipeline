"""
Unit tests for session state management utilities in GraphJudge Streamlit application.

This test module provides comprehensive testing for:
- SessionStateManager functionality
- StatePersistenceManager operations
- StateCleanupManager cleanup rules
- Data caching and retrieval
- State persistence and recovery

Following TDD principles from docs/Testing_Demands.md and test coverage requirements
from spec.md Section 15 (testing strategy).
"""

import unittest
import tempfile
import json
import time
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any

# Set up path for imports
import sys
import os
parent_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

import streamlit as st

# Import modules under test
from streamlit_pipeline.utils.session_state import (
    SessionStateManager, SessionStateKey, CacheEntry, CacheStats,
    get_session_manager, store_pipeline_result, get_current_pipeline_result
)
from streamlit_pipeline.utils.state_persistence import (
    StatePersistenceManager, StateSerializer, 
    get_persistence_manager, persist_pipeline_result
)
from streamlit_pipeline.utils.state_cleanup import (
    StateCleanupManager, CleanupStrategy, CleanupRule,
    get_cleanup_manager, cleanup_expired_data
)
from streamlit_pipeline.core.models import EntityResult, TripleResult, JudgmentResult, Triple
from streamlit_pipeline.core.pipeline import PipelineResult
from streamlit_pipeline.utils.error_handling import ErrorInfo, ErrorType, ErrorSeverity


class TestSessionStateManager(unittest.TestCase):
    """Test cases for SessionStateManager class."""
    
    def setUp(self):
        """Set up test environment before each test."""
        # Mock Streamlit session state
        self.mock_session_state = {}
        
        # Patch st.session_state
        self.session_state_patcher = patch('streamlit.session_state', self.mock_session_state)
        self.session_state_patcher.start()
        
        # Create manager instance
        self.manager = SessionStateManager()
    
    def tearDown(self):
        """Clean up after each test."""
        self.session_state_patcher.stop()
        
        # Reset global manager
        import streamlit_pipeline.utils.session_state
        streamlit_pipeline.utils.session_state._session_manager = None
    
    def test_session_state_initialization(self):
        """Test that session state is properly initialized."""
        # Check that core keys are initialized (exclude LOGGER which is handled by app.py)
        core_keys = [
            SessionStateKey.PIPELINE_RESULTS.value,
            SessionStateKey.CURRENT_RESULT.value,
            SessionStateKey.PIPELINE_STATE.value,
            SessionStateKey.PROCESSING.value,
            SessionStateKey.CURRENT_STAGE.value,
            SessionStateKey.PROGRESS_DATA.value,
            SessionStateKey.RUN_COUNT.value,
            SessionStateKey.CONFIG_OPTIONS.value,
            SessionStateKey.DATA_CACHE.value,
            SessionStateKey.CACHE_STATS.value,
            SessionStateKey.SHOW_DETAILED_RESULTS.value,
            SessionStateKey.SHOW_COMPARISON.value,
            SessionStateKey.ERROR_HISTORY.value,
            SessionStateKey.TEMP_INPUT.value,
            SessionStateKey.TEMP_SELECTIONS.value
        ]
        
        for key in core_keys:
            self.assertIn(key, self.mock_session_state, f"Key {key} not initialized")
        
        # Check that session metadata is initialized
        self.assertIn('session_metadata', self.mock_session_state)
        
        # Check default values
        self.assertEqual(self.mock_session_state[SessionStateKey.PIPELINE_RESULTS.value], [])
        self.assertIsNone(self.mock_session_state[SessionStateKey.CURRENT_RESULT.value])
        self.assertFalse(self.mock_session_state[SessionStateKey.PROCESSING.value])
        self.assertEqual(self.mock_session_state[SessionStateKey.RUN_COUNT.value], 0)
    
    def test_current_result_management(self):
        """Test setting and getting current pipeline results."""
        # Create a mock pipeline result
        result = PipelineResult(
            success=True,
            stage_reached=3,
            total_time=10.5,
            stats={'test': 'data'}
        )
        
        # Set current result
        self.manager.set_current_result(result)
        
        # Verify result is stored
        stored_result = self.manager.get_current_result()
        self.assertEqual(stored_result, result)
        
        # Verify it's added to results history
        results = self.manager.get_pipeline_results()
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0], result)
        
        # Verify run count incremented
        self.assertEqual(self.mock_session_state[SessionStateKey.RUN_COUNT.value], 1)
    
    def test_successful_results_filtering(self):
        """Test filtering of successful results."""
        # Create mixed results
        successful_result = PipelineResult(success=True, stage_reached=3, total_time=5.0)
        failed_result = PipelineResult(success=False, stage_reached=1, total_time=2.0, error="Test error")
        
        self.manager.set_current_result(successful_result)
        self.manager.set_current_result(failed_result)
        
        # Test successful results filtering
        successful_results = self.manager.get_successful_results()
        self.assertEqual(len(successful_results), 1)
        self.assertTrue(successful_results[0].success)
    
    def test_processing_state_management(self):
        """Test processing state tracking."""
        # Initially not processing
        self.assertFalse(self.manager.is_processing())
        self.assertEqual(self.manager.get_current_stage(), -1)
        
        # Start processing
        self.manager.set_processing_state(True, 1)
        self.assertTrue(self.manager.is_processing())
        self.assertEqual(self.manager.get_current_stage(), 1)
        
        # Stop processing
        self.manager.set_processing_state(False)
        self.assertFalse(self.manager.is_processing())
    
    def test_progress_data_tracking(self):
        """Test progress data updates."""
        # Update progress data
        self.manager.update_progress_data(2, "Processing triples", extra_data="test")
        
        progress_data = self.manager.get_progress_data()
        self.assertEqual(progress_data['stage'], 2)
        self.assertEqual(progress_data['message'], "Processing triples")
        self.assertEqual(progress_data['extra_data'], "test")
        self.assertIn('timestamp', progress_data)
    
    def test_data_caching(self):
        """Test data caching functionality."""
        test_data = {"key": "value", "number": 42}
        cache_key = "test_cache"
        
        # Cache data
        success = self.manager.cache_data(cache_key, test_data)
        self.assertTrue(success)
        
        # Retrieve cached data
        cached_data, hit = self.manager.get_cached_data(cache_key)
        self.assertTrue(hit)
        self.assertEqual(cached_data, test_data)
        
        # Test cache miss
        missing_data, hit = self.manager.get_cached_data("nonexistent")
        self.assertFalse(hit)
        self.assertIsNone(missing_data)
    
    def test_cache_eviction(self):
        """Test cache eviction when memory limits exceeded."""
        # Fill cache with data
        for i in range(5):
            self.manager.cache_data(f"key_{i}", f"data_{i}")
        
        # Trigger eviction
        self.manager._evict_cache_entries(evict_count=2)
        
        # Verify cache size reduced
        cache = self.mock_session_state.get(SessionStateKey.DATA_CACHE.value, {})
        self.assertEqual(len(cache), 3)  # Should have 3 remaining after evicting 2
    
    def test_ui_state_management(self):
        """Test UI state management."""
        # Set UI state
        self.manager.set_ui_state("custom_key", "custom_value")
        value = self.manager.get_ui_state("custom_key")
        self.assertEqual(value, "custom_value")
        
        # Test default value
        default_value = self.manager.get_ui_state("nonexistent", "default")
        self.assertEqual(default_value, "default")
        
        # Test toggle functions
        self.manager.toggle_detailed_results()
        detailed = self.mock_session_state[SessionStateKey.SHOW_DETAILED_RESULTS.value]
        self.assertTrue(detailed)
        
        self.manager.toggle_detailed_results()
        detailed = self.mock_session_state[SessionStateKey.SHOW_DETAILED_RESULTS.value]
        self.assertFalse(detailed)
    
    def test_error_tracking(self):
        """Test error tracking functionality."""
        error_info = ErrorInfo(
            error_type=ErrorType.API_AUTH,
            severity=ErrorSeverity.HIGH,
            message="Test error message",
            stage="test_stage",
            technical_details="Technical details"
        )
        
        # Add error
        self.manager.add_error(error_info)
        
        # Retrieve recent errors
        recent_errors = self.manager.get_recent_errors(1)
        self.assertEqual(len(recent_errors), 1)
        self.assertEqual(recent_errors[0]['message'], "Test error message")
        self.assertEqual(recent_errors[0]['error_type'], ErrorType.API_AUTH.value)
    
    def test_reset_functions(self):
        """Test various reset functions."""
        # Set up some data
        result = PipelineResult(success=True, stage_reached=3, total_time=5.0)
        self.manager.set_current_result(result)
        self.manager.set_processing_state(True, 2)
        self.manager.cache_data("test", "data")
        
        # Test pipeline data reset
        self.manager.reset_pipeline_data()
        self.assertEqual(len(self.manager.get_pipeline_results()), 0)
        self.assertIsNone(self.manager.get_current_result())
        self.assertFalse(self.manager.is_processing())
        
        # Test complete session reset
        self.manager.reset_session()
        self.assertEqual(self.mock_session_state[SessionStateKey.RUN_COUNT.value], 0)
        
        # Cache should be cleared
        cache = self.mock_session_state.get(SessionStateKey.DATA_CACHE.value, {})
        self.assertEqual(len(cache), 0)
    
    def test_session_metadata(self):
        """Test session metadata management."""
        metadata = self.manager.get_session_metadata()
        
        self.assertIsNotNone(metadata.session_id)
        self.assertIsInstance(metadata.created_at, datetime)
        self.assertIsInstance(metadata.last_activity, datetime)
        self.assertEqual(metadata.run_count, 0)
    
    def test_export_session_data(self):
        """Test session data export for debugging."""
        # Add some data
        result = PipelineResult(success=True, stage_reached=3, total_time=5.0)
        self.manager.set_current_result(result)
        
        # Export data
        exported_data = self.manager.export_session_data()
        
        self.assertIn('metadata', exported_data)
        self.assertIn('cache_stats', exported_data)
        self.assertIn('pipeline_results_count', exported_data)
        self.assertEqual(exported_data['pipeline_results_count'], 1)


class TestStatePersistenceManager(unittest.TestCase):
    """Test cases for StatePersistenceManager class."""
    
    def setUp(self):
        """Set up test environment."""
        self.temp_dir = Path(tempfile.mkdtemp())
        
        # Mock session state
        self.mock_session_state = {}
        self.session_state_patcher = patch('streamlit.session_state', self.mock_session_state)
        self.session_state_patcher.start()
        
        # Create manager with temporary directory
        self.manager = StatePersistenceManager(temp_dir=self.temp_dir)
    
    def tearDown(self):
        """Clean up test environment."""
        self.session_state_patcher.stop()
        
        # Clean up temporary directory
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_state_serialization(self):
        """Test pipeline result serialization and deserialization."""
        # Create a complex pipeline result
        entity_result = EntityResult(
            entities=["entity1", "entity2"],
            denoised_text="cleaned text",
            success=True,
            processing_time=2.5
        )
        
        triple = Triple(subject="s1", predicate="p1", object="o1", confidence=0.9)
        triple_result = TripleResult(
            triples=[triple],
            metadata={"source": "test"},
            success=True,
            processing_time=3.0
        )
        
        judgment_result = JudgmentResult(
            judgments=[True, False],
            confidence=[0.8, 0.6],
            success=True,
            processing_time=1.5
        )
        
        pipeline_result = PipelineResult(
            success=True,
            stage_reached=3,
            total_time=7.0,
            entity_result=entity_result,
            triple_result=triple_result,
            judgment_result=judgment_result
        )
        
        # Test serialization
        serializer = StateSerializer()
        serialized = serializer.serialize_pipeline_result(pipeline_result)
        
        self.assertEqual(serialized['__type__'], 'PipelineResult')
        self.assertEqual(serialized['success'], True)
        self.assertEqual(serialized['total_time'], 7.0)
        
        # Test deserialization
        deserialized = serializer.deserialize_pipeline_result(serialized)
        
        self.assertIsNotNone(deserialized)
        self.assertEqual(deserialized.success, True)
        self.assertEqual(deserialized.total_time, 7.0)
        self.assertEqual(len(deserialized.triple_result.triples), 1)
        self.assertEqual(deserialized.triple_result.triples[0].subject, "s1")
    
    def test_large_result_persistence(self):
        """Test persistence of large results."""
        # Create a result that should trigger file storage
        large_result = PipelineResult(
            success=True,
            stage_reached=3,
            total_time=15.0,
            stats={"large_data": "x" * 1000}  # Make it larger
        )
        
        # Persist result
        success = self.manager.persist_large_result("large_test", large_result, size_threshold_mb=0.001)
        self.assertTrue(success)
        
        # Retrieve result
        retrieved = self.manager.retrieve_result("large_test")
        self.assertIsNotNone(retrieved)
        self.assertEqual(retrieved.success, True)
        self.assertEqual(retrieved.total_time, 15.0)
    
    def test_intermediate_state_persistence(self):
        """Test intermediate state persistence."""
        state_data = {
            "current_entities": ["entity1", "entity2"],
            "progress": 0.5,
            "temp_results": {"key": "value"}
        }
        
        # Persist intermediate state
        success = self.manager.persist_intermediate_state("entity_extraction", state_data)
        self.assertTrue(success)
        
        # Retrieve intermediate state
        retrieved_state = self.manager.retrieve_intermediate_state("entity_extraction")
        self.assertIsNotNone(retrieved_state)
        self.assertEqual(retrieved_state["progress"], 0.5)
        self.assertEqual(len(retrieved_state["current_entities"]), 2)
    
    def test_file_based_persistence(self):
        """Test file-based persistence for large data."""
        # Create a proper pipeline result to test with
        pipeline_result = PipelineResult(
            success=True,
            stage_reached=3,
            total_time=5.0,
            stats={"large_data": "x" * 10000}  # Make it large
        )
        
        # Serialize the result first (as would happen in normal flow)
        serialized_data = self.manager.serializer.serialize_pipeline_result(pipeline_result)
        
        # Force file persistence with serialized data
        success = self.manager._persist_to_file("file_test", serialized_data, 5.0)
        self.assertTrue(success)
        
        # Check file was created
        json_files = list(self.temp_dir.glob("*.json"))
        self.assertEqual(len(json_files), 1)
        
        # Verify that the file reference was stored in session state
        file_ref_key = "temp_file_file_test"
        self.assertIn(file_ref_key, self.mock_session_state)
        
        # Retrieve from file (should work now that the data is properly serialized)
        retrieved = self.manager._retrieve_from_file("file_test")
        self.assertIsNotNone(retrieved)
        self.assertEqual(retrieved.success, True)
        self.assertEqual(retrieved.total_time, 5.0)
    
    def test_cleanup_expired_data(self):
        """Test cleanup of expired data."""
        # Create some test files with old timestamps
        old_file = self.temp_dir / "old_test_20200101_120000.json"
        with open(old_file, 'w') as f:
            json.dump({"test": "data"}, f)
        
        # Set old modification time using os.utime
        old_time = time.time() - 48 * 3600  # 48 hours ago
        os.utime(old_file, (old_time, old_time))
        
        # Run cleanup
        self.manager.cleanup_expired_data(max_age_hours=24)
        
        # File should be removed
        self.assertFalse(old_file.exists())
    
    def test_persistence_stats(self):
        """Test persistence statistics reporting."""
        # Add some data
        result = PipelineResult(success=True, stage_reached=3, total_time=5.0)
        self.manager.persist_large_result("stats_test", result)
        
        # Get statistics
        stats = self.manager.get_persistence_stats()
        
        self.assertIn('cache_stats', stats)
        self.assertIn('temp_files', stats)
        self.assertIn('session_manager', stats)
    
    def test_reset_all_persistence(self):
        """Test complete persistence reset."""
        # Add some data
        result = PipelineResult(success=True, stage_reached=3, total_time=5.0)
        self.manager.persist_large_result("reset_test", result)
        
        # Create a temp file
        test_file = self.temp_dir / "test.json"
        test_file.write_text('{"test": "data"}')
        
        # Reset all persistence
        self.manager.reset_all_persistence()
        
        # Verify cleanup
        self.assertFalse(test_file.exists())
    
    def test_data_integrity_checking(self):
        """Test data integrity verification."""
        serializer = StateSerializer()
        
        # Test hash calculation
        test_data = {"key": "value", "number": 42}
        hash1 = serializer.calculate_data_hash(test_data)
        hash2 = serializer.calculate_data_hash(test_data)
        
        # Same data should produce same hash
        self.assertEqual(hash1, hash2)
        
        # Different data should produce different hash
        different_data = {"key": "different", "number": 42}
        hash3 = serializer.calculate_data_hash(different_data)
        self.assertNotEqual(hash1, hash3)


class TestStateCleanupManager(unittest.TestCase):
    """Test cases for StateCleanupManager class."""
    
    def setUp(self):
        """Set up test environment."""
        # Mock session state
        self.mock_session_state = {}
        self.session_state_patcher = patch('streamlit.session_state', self.mock_session_state)
        self.session_state_patcher.start()
        
        # Create manager instance
        self.manager = StateCleanupManager()
    
    def tearDown(self):
        """Clean up test environment."""
        self.session_state_patcher.stop()
    
    def test_cleanup_rules_initialization(self):
        """Test that cleanup rules are properly initialized."""
        rules = self.manager.cleanup_rules
        
        self.assertGreater(len(rules), 0)
        
        # Check for required rules
        rule_names = [rule.name for rule in rules]
        expected_rules = ["expired_cache", "large_results", "temp_files", "error_history", "old_results"]
        
        for expected_rule in expected_rules:
            self.assertIn(expected_rule, rule_names)
    
    def test_cleanup_rule_conditions(self):
        """Test cleanup rule condition evaluation."""
        # Create a conditional rule
        test_rule = CleanupRule(
            name="test_rule",
            description="Test rule",
            strategy=CleanupStrategy.CONDITIONAL,
            condition_func=lambda: True
        )
        
        # Test should execute conditions
        self.assertTrue(self.manager._should_execute_rule(test_rule, force=False))
        
        # Test force execution
        manual_rule = CleanupRule(
            name="manual_rule",
            description="Manual rule",
            strategy=CleanupStrategy.MANUAL
        )
        
        self.assertFalse(self.manager._should_execute_rule(manual_rule, force=False))
        self.assertTrue(self.manager._should_execute_rule(manual_rule, force=True))
    
    def test_cache_cleanup(self):
        """Test expired cache cleanup."""
        # Mock cache with expired entries
        from streamlit_pipeline.utils.session_state import SessionStateKey, CacheEntry
        from datetime import datetime, timedelta
        
        old_time = datetime.now() - timedelta(hours=2)
        recent_time = datetime.now() - timedelta(minutes=5)
        
        expired_entry = CacheEntry(
            key="expired",
            data="old_data",
            created_at=old_time,
            last_accessed=old_time,
            size_bytes=100
        )
        
        recent_entry = CacheEntry(
            key="recent",
            data="new_data",
            created_at=recent_time,
            last_accessed=recent_time,
            size_bytes=50
        )
        
        self.mock_session_state[SessionStateKey.DATA_CACHE.value] = {
            "expired": expired_entry,
            "recent": recent_entry
        }
        
        # Initialize cache stats to ensure consistency
        from streamlit_pipeline.utils.session_state import CacheStats
        self.mock_session_state[SessionStateKey.CACHE_STATS.value] = CacheStats(
            total_entries=2,
            total_size_bytes=150,
            hit_count=0,
            miss_count=0,
            eviction_count=0
        )
        
        # Create rule for cache cleanup
        cache_rule = CleanupRule(
            name="expired_cache",
            description="Test cache cleanup",
            strategy=CleanupStrategy.IMMEDIATE,
            max_age_minutes=60
        )
        
        # Execute cleanup
        result = self.manager._execute_cleanup_rule(cache_rule)
        
        self.assertEqual(result['status'], 'success')
        # Should clean up at least the expired entry
        self.assertGreaterEqual(result['items_cleaned'], 1)
    
    def test_large_results_cleanup(self):
        """Test cleanup of large pipeline results."""
        from streamlit_pipeline.utils.session_state import SessionStateKey
        
        # Create multiple results
        results = []
        for i in range(15):  # More than max_items in rule
            result = PipelineResult(
                success=True,
                stage_reached=3,
                total_time=5.0,
                stats={"data": "x" * (1000 * i)}  # Varying sizes
            )
            results.append(result)
        
        self.mock_session_state[SessionStateKey.PIPELINE_RESULTS.value] = results
        
        # Create rule for large results cleanup
        large_results_rule = CleanupRule(
            name="large_results",
            description="Test large results cleanup",
            strategy=CleanupStrategy.IMMEDIATE,
            max_items=10
        )
        
        # Execute cleanup
        result = self.manager._execute_cleanup_rule(large_results_rule)
        
        self.assertEqual(result['status'], 'success')
        self.assertEqual(result['items_cleaned'], 5)  # Should remove 5 items
        
        # Verify results were removed
        remaining_results = self.mock_session_state[SessionStateKey.PIPELINE_RESULTS.value]
        self.assertEqual(len(remaining_results), 10)
    
    def test_error_history_cleanup(self):
        """Test error history cleanup."""
        from streamlit_pipeline.utils.session_state import SessionStateKey
        
        # Create large error history
        error_history = []
        for i in range(100):
            error_history.append({
                'timestamp': datetime.now().isoformat(),
                'message': f'Error {i}',
                'type': 'test'
            })
        
        self.mock_session_state[SessionStateKey.ERROR_HISTORY.value] = error_history
        
        # Create rule for error history cleanup
        error_rule = CleanupRule(
            name="error_history",
            description="Test error cleanup",
            strategy=CleanupStrategy.IMMEDIATE,
            max_items=50
        )
        
        # Execute cleanup
        result = self.manager._execute_cleanup_rule(error_rule)
        
        self.assertEqual(result['status'], 'success')
        self.assertEqual(result['items_cleaned'], 50)  # Should remove 50 items
        
        # Verify errors were trimmed
        remaining_errors = self.mock_session_state[SessionStateKey.ERROR_HISTORY.value]
        self.assertEqual(len(remaining_errors), 50)
    
    def test_cleanup_execution(self):
        """Test complete cleanup execution."""
        # Execute cleanup with no specific rules (all applicable rules)
        result = self.manager.execute_cleanup(force=True)
        
        self.assertEqual(result['status'], 'success')
        self.assertIn('executed_rules', result)
        self.assertIn('statistics', result)
        self.assertIn('duration_seconds', result['statistics'])
    
    def test_scheduled_cleanup(self):
        """Test scheduled cleanup functionality."""
        # Schedule cleanup
        self.manager.schedule_automatic_cleanup(interval_minutes=1)
        
        # Check schedule was created
        schedule = self.mock_session_state.get('cleanup_schedule')
        self.assertIsNotNone(schedule)
        self.assertTrue(schedule['enabled'])
        self.assertEqual(schedule['interval_minutes'], 1)
        
        # Simulate time passage and check cleanup
        # Mock the next cleanup time to be in the past
        schedule['next_cleanup'] = datetime.now() - timedelta(minutes=5)
        
        with patch.object(self.manager, 'execute_cleanup') as mock_cleanup:
            mock_cleanup.return_value = {'status': 'success'}
            
            self.manager.check_scheduled_cleanup()
            
            # Verify cleanup was called
            mock_cleanup.assert_called_once()
    
    def test_force_complete_cleanup(self):
        """Test force complete cleanup."""
        with patch.object(self.manager, 'execute_cleanup') as mock_cleanup, \
             patch.object(self.manager.session_manager, 'reset_session') as mock_reset_session, \
             patch.object(self.manager.persistence_manager, 'reset_all_persistence') as mock_reset_persistence:
            
            mock_cleanup.return_value = {'status': 'success'}
            
            self.manager.force_complete_cleanup()
            
            # Verify all cleanup methods were called
            mock_cleanup.assert_called_once_with(force=True)
            mock_reset_session.assert_called_once()
            mock_reset_persistence.assert_called_once()
    
    def test_cleanup_stats(self):
        """Test cleanup statistics reporting."""
        stats = self.manager.get_cleanup_stats()
        
        self.assertIn('cleanup_stats', stats)
        self.assertIn('session_stats', stats)
        self.assertIn('cleanup_rules', stats)
        
        # Verify rule information is included
        rules_info = stats['cleanup_rules']
        self.assertGreater(len(rules_info), 0)
        
        for rule_info in rules_info:
            self.assertIn('name', rule_info)
            self.assertIn('description', rule_info)
            self.assertIn('strategy', rule_info)
    
    def test_memory_pressure_detection(self):
        """Test memory pressure detection."""
        # Test with mocked cache stats
        from streamlit_pipeline.utils.session_state import CacheStats
        
        # Mock high memory usage
        high_usage_stats = CacheStats()
        high_usage_stats.total_size_bytes = 200 * 1024 * 1024  # 200MB
        
        with patch.object(self.manager.session_manager, 'get_cache_stats', return_value=high_usage_stats):
            self.assertTrue(self.manager._check_memory_pressure())
        
        # Mock low memory usage
        low_usage_stats = CacheStats()
        low_usage_stats.total_size_bytes = 10 * 1024 * 1024  # 10MB
        
        with patch.object(self.manager.session_manager, 'get_cache_stats', return_value=low_usage_stats):
            self.assertFalse(self.manager._check_memory_pressure())


class TestConvenienceFunctions(unittest.TestCase):
    """Test cases for module convenience functions."""
    
    def setUp(self):
        """Set up test environment."""
        # Mock session state
        self.mock_session_state = {}
        self.session_state_patcher = patch('streamlit.session_state', self.mock_session_state)
        self.session_state_patcher.start()
    
    def tearDown(self):
        """Clean up test environment."""
        self.session_state_patcher.stop()
        
        # Reset global managers
        import streamlit_pipeline.utils.session_state
        import streamlit_pipeline.utils.state_persistence
        import streamlit_pipeline.utils.state_cleanup
        
        streamlit_pipeline.utils.session_state._session_manager = None
        streamlit_pipeline.utils.state_persistence._persistence_manager = None
        streamlit_pipeline.utils.state_cleanup._cleanup_manager = None
    
    def test_store_and_get_pipeline_result(self):
        """Test convenience functions for pipeline result management."""
        result = PipelineResult(
            success=True,
            stage_reached=3,
            total_time=5.0
        )
        
        # Store result
        store_pipeline_result(result)
        
        # Get result
        retrieved = get_current_pipeline_result()
        self.assertEqual(retrieved, result)
    
    def test_cleanup_expired_data(self):
        """Test cleanup convenience function."""
        with patch('streamlit_pipeline.utils.state_cleanup.get_cleanup_manager') as mock_get_manager:
            mock_manager = Mock()
            mock_manager.execute_cleanup.return_value = {'status': 'success'}
            mock_get_manager.return_value = mock_manager
            
            result = cleanup_expired_data()
            
            self.assertEqual(result['status'], 'success')
            mock_manager.execute_cleanup.assert_called_once_with(['expired_cache', 'temp_files'])
    
    def test_singleton_managers(self):
        """Test that managers follow singleton pattern."""
        # Test session manager singleton
        manager1 = get_session_manager()
        manager2 = get_session_manager()
        self.assertIs(manager1, manager2)
        
        # Test persistence manager singleton
        pers_manager1 = get_persistence_manager()
        pers_manager2 = get_persistence_manager()
        self.assertIs(pers_manager1, pers_manager2)
        
        # Test cleanup manager singleton
        cleanup_manager1 = get_cleanup_manager()
        cleanup_manager2 = get_cleanup_manager()
        self.assertIs(cleanup_manager1, cleanup_manager2)


if __name__ == '__main__':
    # Run tests with verbose output
    unittest.main(verbosity=2)