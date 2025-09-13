"""
Session State Management for GraphJudge Streamlit Application

This module provides comprehensive session state management utilities for managing
pipeline data flow, intermediate results, and application state persistence across
Streamlit reruns.

Implements requirements from spec.md Section 2 (target architecture), Section 8 
(data models), and Section 9 (state machines) for efficient data handling.
"""

import time
import uuid
import logging
from typing import Dict, Any, Optional, List, Union, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, field, asdict
from enum import Enum

import streamlit as st

from ..core.models import (
    EntityResult, TripleResult, JudgmentResult, Triple, PipelineState
)
from ..core.pipeline import PipelineResult
from .error_handling import ErrorHandler, ErrorInfo, ErrorType


class SessionStateKey(Enum):
    """Enumeration of all session state keys used in the application."""
    
    # Core pipeline data
    PIPELINE_RESULTS = "pipeline_results"
    CURRENT_RESULT = "current_result"
    PIPELINE_STATE = "pipeline_state"
    
    # Processing state
    PROCESSING = "processing"
    CURRENT_STAGE = "current_stage"
    PROGRESS_DATA = "progress_data"
    
    # Application state
    RUN_COUNT = "run_count"
    SESSION_ID = "session_id"
    CONFIG_OPTIONS = "config_options"
    
    # Data cache and performance
    DATA_CACHE = "data_cache"
    CACHE_STATS = "cache_stats"
    
    # UI state
    SHOW_DETAILED_RESULTS = "show_detailed_results"
    SHOW_COMPARISON = "show_comparison"
    SELECTED_RESULT_INDEX = "selected_result_index"
    
    # Logging and debugging
    LOGGER = "logger"
    DEBUG_MODE = "debug_mode"
    ERROR_HISTORY = "error_history"
    
    # Temporary storage
    TEMP_INPUT = "temp_input"
    TEMP_SELECTIONS = "temp_selections"


@dataclass
class SessionMetadata:
    """Metadata about the current session."""
    session_id: str
    created_at: datetime
    last_activity: datetime
    run_count: int = 0
    total_processing_time: float = 0.0
    successful_runs: int = 0
    failed_runs: int = 0


@dataclass
class CacheEntry:
    """Entry in the data cache for performance optimization."""
    key: str
    data: Any
    created_at: datetime
    last_accessed: datetime
    access_count: int = 0
    size_bytes: int = 0
    
    def update_access(self):
        """Update access statistics."""
        self.last_accessed = datetime.now()
        self.access_count += 1


@dataclass 
class CacheStats:
    """Statistics about cache performance."""
    total_entries: int = 0
    total_size_bytes: int = 0
    hit_count: int = 0
    miss_count: int = 0
    eviction_count: int = 0
    
    @property
    def hit_rate(self) -> float:
        """Calculate cache hit rate."""
        total = self.hit_count + self.miss_count
        return self.hit_count / total if total > 0 else 0.0


class SessionStateManager:
    """
    Comprehensive session state manager for the GraphJudge Streamlit application.
    
    Provides utilities for:
    - Managing pipeline data flow and intermediate results
    - Caching expensive operations and API responses
    - Persisting application state across reruns
    - Cleanup and memory management
    - Performance monitoring and debugging
    
    Following spec.md requirements for efficient session state management.
    """
    
    def __init__(self):
        """Initialize the session state manager."""
        self.error_handler = ErrorHandler()
        self.logger = logging.getLogger(__name__)
        self._initialize_session_state()
    
    def _initialize_session_state(self):
        """Initialize all session state variables with default values."""
        
        # Generate session ID if not exists
        if SessionStateKey.SESSION_ID.value not in st.session_state:
            st.session_state[SessionStateKey.SESSION_ID.value] = str(uuid.uuid4())
        
        # Initialize metadata
        if 'session_metadata' not in st.session_state:
            st.session_state['session_metadata'] = SessionMetadata(
                session_id=st.session_state[SessionStateKey.SESSION_ID.value],
                created_at=datetime.now(),
                last_activity=datetime.now()
            )
        
        # Core pipeline data
        self._ensure_key_exists(SessionStateKey.PIPELINE_RESULTS, [])
        self._ensure_key_exists(SessionStateKey.CURRENT_RESULT, None)
        self._ensure_key_exists(SessionStateKey.PIPELINE_STATE, None)
        
        # Processing state
        self._ensure_key_exists(SessionStateKey.PROCESSING, False)
        self._ensure_key_exists(SessionStateKey.CURRENT_STAGE, -1)
        self._ensure_key_exists(SessionStateKey.PROGRESS_DATA, {})
        
        # Application state
        self._ensure_key_exists(SessionStateKey.RUN_COUNT, 0)
        self._ensure_key_exists(SessionStateKey.CONFIG_OPTIONS, {})
        
        # Data cache and performance
        self._ensure_key_exists(SessionStateKey.DATA_CACHE, {})
        self._ensure_key_exists(SessionStateKey.CACHE_STATS, CacheStats())
        
        # UI state
        self._ensure_key_exists(SessionStateKey.SHOW_DETAILED_RESULTS, False)
        self._ensure_key_exists(SessionStateKey.SHOW_COMPARISON, False)
        self._ensure_key_exists(SessionStateKey.SELECTED_RESULT_INDEX, None)
        
        # Error tracking
        self._ensure_key_exists(SessionStateKey.ERROR_HISTORY, [])
        self._ensure_key_exists(SessionStateKey.DEBUG_MODE, False)
        
        # Temporary storage
        self._ensure_key_exists(SessionStateKey.TEMP_INPUT, "")
        self._ensure_key_exists(SessionStateKey.TEMP_SELECTIONS, {})
    
    def _ensure_key_exists(self, key: SessionStateKey, default_value: Any):
        """Ensure a session state key exists with default value."""
        key_name = key.value
        if key_name not in st.session_state:
            st.session_state[key_name] = default_value
    
    # Core pipeline data management
    
    def set_current_result(self, result: PipelineResult):
        """
        Set the current pipeline result and update metadata.
        
        Args:
            result: The pipeline result to store
        """
        st.session_state[SessionStateKey.CURRENT_RESULT.value] = result
        
        # Add to results history
        pipeline_results = st.session_state[SessionStateKey.PIPELINE_RESULTS.value]
        pipeline_results.append(result)
        
        # Update session metadata
        metadata = st.session_state['session_metadata']
        metadata.last_activity = datetime.now()
        metadata.run_count += 1
        metadata.total_processing_time += result.total_time
        
        if result.success:
            metadata.successful_runs += 1
        else:
            metadata.failed_runs += 1
        
        # Increment run count
        st.session_state[SessionStateKey.RUN_COUNT.value] += 1
        
        self.logger.info(f"Pipeline result stored. Run #{metadata.run_count}, Success: {result.success}")
    
    def get_current_result(self) -> Optional[PipelineResult]:
        """Get the current pipeline result."""
        return st.session_state.get(SessionStateKey.CURRENT_RESULT.value)
    
    def get_pipeline_results(self) -> List[PipelineResult]:
        """Get all pipeline results from session history."""
        return st.session_state.get(SessionStateKey.PIPELINE_RESULTS.value, [])
    
    def get_successful_results(self) -> List[PipelineResult]:
        """Get only successful pipeline results."""
        return [r for r in self.get_pipeline_results() if r.success]
    
    def set_pipeline_state(self, state: PipelineState):
        """Set the current pipeline state."""
        st.session_state[SessionStateKey.PIPELINE_STATE.value] = state
    
    def get_pipeline_state(self) -> Optional[PipelineState]:
        """Get the current pipeline state."""
        return st.session_state.get(SessionStateKey.PIPELINE_STATE.value)
    
    # Processing state management
    
    def set_processing_state(self, processing: bool, stage: int = -1):
        """
        Set the processing state and current stage.
        
        Args:
            processing: Whether pipeline is currently processing
            stage: Current processing stage (-1 for not started)
        """
        st.session_state[SessionStateKey.PROCESSING.value] = processing
        st.session_state[SessionStateKey.CURRENT_STAGE.value] = stage
        
        if processing:
            self.logger.info(f"Processing started at stage {stage}")
        else:
            self.logger.info("Processing completed/stopped")
    
    def is_processing(self) -> bool:
        """Check if pipeline is currently processing."""
        return st.session_state.get(SessionStateKey.PROCESSING.value, False)
    
    def get_current_stage(self) -> int:
        """Get the current processing stage."""
        return st.session_state.get(SessionStateKey.CURRENT_STAGE.value, -1)
    
    def update_progress_data(self, stage: int, message: str, **kwargs):
        """
        Update progress tracking data.
        
        Args:
            stage: Current stage number
            message: Progress message
            **kwargs: Additional progress data
        """
        progress_data = st.session_state.get(SessionStateKey.PROGRESS_DATA.value, {})
        progress_data.update({
            'stage': stage,
            'message': message,
            'timestamp': datetime.now().isoformat(),
            **kwargs
        })
        st.session_state[SessionStateKey.PROGRESS_DATA.value] = progress_data
    
    def get_progress_data(self) -> Dict[str, Any]:
        """Get current progress tracking data."""
        return st.session_state.get(SessionStateKey.PROGRESS_DATA.value, {})
    
    # Data caching for performance optimization
    
    def cache_data(self, key: str, data: Any, max_size_mb: int = 50) -> bool:
        """
        Cache data for performance optimization.
        
        Args:
            key: Cache key
            data: Data to cache
            max_size_mb: Maximum cache size in MB
            
        Returns:
            True if successfully cached, False otherwise
        """
        try:
            import sys
            data_size = sys.getsizeof(data)
            
            # Check cache size limits
            cache_stats = st.session_state.get(SessionStateKey.CACHE_STATS.value, CacheStats())
            if cache_stats.total_size_bytes + data_size > max_size_mb * 1024 * 1024:
                self._evict_cache_entries()
            
            # Create cache entry
            cache_entry = CacheEntry(
                key=key,
                data=data,
                created_at=datetime.now(),
                last_accessed=datetime.now(),
                size_bytes=data_size
            )
            
            # Store in cache
            cache = st.session_state.get(SessionStateKey.DATA_CACHE.value, {})
            cache[key] = cache_entry
            st.session_state[SessionStateKey.DATA_CACHE.value] = cache
            
            # Update cache stats
            cache_stats.total_entries += 1
            cache_stats.total_size_bytes += data_size
            st.session_state[SessionStateKey.CACHE_STATS.value] = cache_stats
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to cache data: {str(e)}")
            return False
    
    def get_cached_data(self, key: str) -> Tuple[Optional[Any], bool]:
        """
        Retrieve data from cache.
        
        Args:
            key: Cache key
            
        Returns:
            Tuple of (data, hit) where hit indicates if cache was hit
        """
        cache = st.session_state.get(SessionStateKey.DATA_CACHE.value, {})
        cache_stats = st.session_state.get(SessionStateKey.CACHE_STATS.value, CacheStats())
        
        if key in cache:
            entry = cache[key]
            entry.update_access()
            cache_stats.hit_count += 1
            st.session_state[SessionStateKey.CACHE_STATS.value] = cache_stats
            return entry.data, True
        else:
            cache_stats.miss_count += 1
            st.session_state[SessionStateKey.CACHE_STATS.value] = cache_stats
            return None, False
    
    def _evict_cache_entries(self, evict_count: int = 3):
        """
        Evict least recently used cache entries.
        
        Args:
            evict_count: Number of entries to evict
        """
        cache = st.session_state.get(SessionStateKey.DATA_CACHE.value, {})
        cache_stats = st.session_state.get(SessionStateKey.CACHE_STATS.value, CacheStats())
        
        if len(cache) <= evict_count:
            return
        
        # Sort by last access time
        sorted_entries = sorted(cache.values(), key=lambda x: x.last_accessed)
        
        # Remove oldest entries
        for entry in sorted_entries[:evict_count]:
            if entry.key in cache:
                cache_stats.total_size_bytes -= entry.size_bytes
                cache_stats.total_entries -= 1
                cache_stats.eviction_count += 1
                del cache[entry.key]
        
        st.session_state[SessionStateKey.DATA_CACHE.value] = cache
        st.session_state[SessionStateKey.CACHE_STATS.value] = cache_stats
    
    # UI state management
    
    def set_ui_state(self, key: str, value: Any):
        """Set UI-related state."""
        if key in [k.value for k in SessionStateKey]:
            st.session_state[key] = value
        else:
            # Store in temp selections
            temp_selections = st.session_state.get(SessionStateKey.TEMP_SELECTIONS.value, {})
            temp_selections[key] = value
            st.session_state[SessionStateKey.TEMP_SELECTIONS.value] = temp_selections
    
    def get_ui_state(self, key: str, default: Any = None) -> Any:
        """Get UI-related state."""
        if key in [k.value for k in SessionStateKey]:
            return st.session_state.get(key, default)
        else:
            temp_selections = st.session_state.get(SessionStateKey.TEMP_SELECTIONS.value, {})
            return temp_selections.get(key, default)
    
    def toggle_detailed_results(self):
        """Toggle detailed results view."""
        current = st.session_state.get(SessionStateKey.SHOW_DETAILED_RESULTS.value, False)
        st.session_state[SessionStateKey.SHOW_DETAILED_RESULTS.value] = not current
    
    def toggle_comparison_view(self):
        """Toggle comparison view."""
        current = st.session_state.get(SessionStateKey.SHOW_COMPARISON.value, False)
        st.session_state[SessionStateKey.SHOW_COMPARISON.value] = not current
    
    # Error handling and logging
    
    def add_error(self, error_info: ErrorInfo):
        """Add error to session error history."""
        error_history = st.session_state.get(SessionStateKey.ERROR_HISTORY.value, [])
        error_history.append({
            'timestamp': datetime.now().isoformat(),
            'error_type': error_info.error_type.value,
            'message': error_info.message,
            'stage': error_info.stage,
            'technical_details': error_info.technical_details
        })
        
        # Keep only last 50 errors
        if len(error_history) > 50:
            error_history = error_history[-50:]
        
        st.session_state[SessionStateKey.ERROR_HISTORY.value] = error_history
    
    def get_recent_errors(self, count: int = 10) -> List[Dict[str, Any]]:
        """Get recent errors from session history."""
        error_history = st.session_state.get(SessionStateKey.ERROR_HISTORY.value, [])
        return error_history[-count:]
    
    def clear_error_history(self):
        """Clear all error history."""
        st.session_state[SessionStateKey.ERROR_HISTORY.value] = []
    
    # State cleanup and reset
    
    def reset_pipeline_data(self):
        """Reset all pipeline-related data."""
        st.session_state[SessionStateKey.PIPELINE_RESULTS.value] = []
        st.session_state[SessionStateKey.CURRENT_RESULT.value] = None
        st.session_state[SessionStateKey.PIPELINE_STATE.value] = None
        st.session_state[SessionStateKey.PROCESSING.value] = False
        st.session_state[SessionStateKey.CURRENT_STAGE.value] = -1
        st.session_state[SessionStateKey.PROGRESS_DATA.value] = {}
        
        self.logger.info("Pipeline data reset")
    
    def reset_ui_state(self):
        """Reset UI-related state."""
        st.session_state[SessionStateKey.SHOW_DETAILED_RESULTS.value] = False
        st.session_state[SessionStateKey.SHOW_COMPARISON.value] = False
        st.session_state[SessionStateKey.SELECTED_RESULT_INDEX.value] = None
        st.session_state[SessionStateKey.TEMP_SELECTIONS.value] = {}
        
        self.logger.info("UI state reset")
    
    def clear_cache(self):
        """Clear all cached data."""
        st.session_state[SessionStateKey.DATA_CACHE.value] = {}
        st.session_state[SessionStateKey.CACHE_STATS.value] = CacheStats()
        
        self.logger.info("Cache cleared")
    
    def reset_session(self):
        """Complete session reset - clear all data."""
        # Reset metadata but keep session ID
        session_id = st.session_state.get(SessionStateKey.SESSION_ID.value)
        st.session_state['session_metadata'] = SessionMetadata(
            session_id=session_id,
            created_at=datetime.now(),
            last_activity=datetime.now()
        )
        
        # Reset all state
        self.reset_pipeline_data()
        self.reset_ui_state()
        self.clear_cache()
        self.clear_error_history()
        
        st.session_state[SessionStateKey.RUN_COUNT.value] = 0
        st.session_state[SessionStateKey.CONFIG_OPTIONS.value] = {}
        
        self.logger.info("Complete session reset performed")
    
    # Utility methods
    
    def get_session_metadata(self) -> SessionMetadata:
        """Get session metadata."""
        return st.session_state.get('session_metadata', SessionMetadata(
            session_id=str(uuid.uuid4()),
            created_at=datetime.now(),
            last_activity=datetime.now()
        ))
    
    def get_cache_stats(self) -> CacheStats:
        """Get cache performance statistics."""
        return st.session_state.get(SessionStateKey.CACHE_STATS.value, CacheStats())
    
    def export_session_data(self) -> Dict[str, Any]:
        """
        Export session data for debugging or analysis.
        
        Returns:
            Dictionary containing serializable session data
        """
        try:
            metadata = self.get_session_metadata()
            cache_stats = self.get_cache_stats()
            
            return {
                'metadata': asdict(metadata),
                'cache_stats': asdict(cache_stats),
                'pipeline_results_count': len(self.get_pipeline_results()),
                'successful_runs': len(self.get_successful_results()),
                'error_count': len(st.session_state.get(SessionStateKey.ERROR_HISTORY.value, [])),
                'current_processing': self.is_processing(),
                'current_stage': self.get_current_stage(),
                'session_keys': list(st.session_state.keys())
            }
        except Exception as e:
            self.logger.error(f"Failed to export session data: {str(e)}")
            return {'error': str(e)}


# Global session state manager instance
_session_manager = None

def get_session_manager() -> SessionStateManager:
    """Get the global session state manager instance."""
    global _session_manager
    if _session_manager is None:
        _session_manager = SessionStateManager()
    return _session_manager


# Convenience functions for common operations

def store_pipeline_result(result: PipelineResult):
    """Store a pipeline result in session state."""
    get_session_manager().set_current_result(result)

def get_current_pipeline_result() -> Optional[PipelineResult]:
    """Get the current pipeline result."""
    return get_session_manager().get_current_result()

def cache_expensive_operation(key: str, data: Any) -> bool:
    """Cache result of expensive operation."""
    return get_session_manager().cache_data(key, data)

def get_cached_operation(key: str) -> Tuple[Optional[Any], bool]:
    """Get cached operation result."""
    return get_session_manager().get_cached_data(key)

def reset_all_data():
    """Reset all session data."""
    get_session_manager().reset_session()

def track_error(error_info: ErrorInfo):
    """Track error in session state."""
    get_session_manager().add_error(error_info)