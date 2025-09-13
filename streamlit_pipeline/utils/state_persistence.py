"""
State Persistence Utilities for GraphJudge Streamlit Application

This module provides utilities for persisting application state across Streamlit
reruns, handling large intermediate results efficiently, and managing state
synchronization with external storage when needed.

Implements requirements from spec.md Section 2 (target architecture) for
in-memory data passing and Section 8 (data models) for efficient data flow.
"""

import json
import pickle
import hashlib
import time
import logging
from typing import Any, Dict, Optional, Tuple, List, Union
from datetime import datetime, timedelta
from pathlib import Path
from dataclasses import asdict, is_dataclass

import streamlit as st

from ..core.models import EntityResult, TripleResult, JudgmentResult, PipelineState
from ..core.pipeline import PipelineResult
from .session_state import SessionStateManager, get_session_manager


class StateSerializer:
    """
    Handles serialization and deserialization of complex state objects.
    
    Provides efficient serialization for large pipeline results and maintains
    compatibility with Streamlit's session state requirements.
    """
    
    @staticmethod
    def serialize_pipeline_result(result: PipelineResult) -> Dict[str, Any]:
        """
        Serialize a PipelineResult for storage.
        
        Args:
            result: PipelineResult to serialize
            
        Returns:
            Serializable dictionary representation
        """
        try:
            # Convert dataclass to dict, handling nested objects
            data = asdict(result)
            
            # Add type information for reconstruction
            data['__type__'] = 'PipelineResult'
            data['__serialized_at__'] = datetime.now().isoformat()
            
            return data
            
        except Exception as e:
            logging.error(f"Failed to serialize PipelineResult: {str(e)}")
            return {'error': f'Serialization failed: {str(e)}'}
    
    @staticmethod
    def deserialize_pipeline_result(data: Dict[str, Any]) -> Optional[PipelineResult]:
        """
        Deserialize a PipelineResult from storage.
        
        Args:
            data: Serialized data dictionary
            
        Returns:
            Reconstructed PipelineResult or None if failed
        """
        try:
            if data.get('__type__') != 'PipelineResult':
                return None
            
            # Remove metadata
            clean_data = {k: v for k, v in data.items() if not k.startswith('__')}
            
            # Reconstruct nested objects
            if clean_data.get('entity_result'):
                clean_data['entity_result'] = EntityResult(**clean_data['entity_result'])
            
            if clean_data.get('triple_result'):
                triple_data = clean_data['triple_result']
                # Reconstruct Triple objects
                if 'triples' in triple_data and triple_data['triples']:
                    from ..core.models import Triple
                    triple_data['triples'] = [
                        Triple(**t) if isinstance(t, dict) else t 
                        for t in triple_data['triples']
                    ]
                clean_data['triple_result'] = TripleResult(**triple_data)
            
            if clean_data.get('judgment_result'):
                clean_data['judgment_result'] = JudgmentResult(**clean_data['judgment_result'])
            
            return PipelineResult(**clean_data)
            
        except Exception as e:
            logging.error(f"Failed to deserialize PipelineResult: {str(e)}")
            return None
    
    @staticmethod
    def calculate_data_hash(data: Any) -> str:
        """
        Calculate hash for data integrity checking.
        
        Args:
            data: Data to hash
            
        Returns:
            SHA-256 hash string
        """
        try:
            if isinstance(data, dict):
                serialized = json.dumps(data, sort_keys=True, default=str)
            else:
                serialized = str(data)
            
            return hashlib.sha256(serialized.encode()).hexdigest()
            
        except Exception:
            return "unknown"


class StatePersistenceManager:
    """
    Manages state persistence across Streamlit reruns with support for
    large data sets, automatic cleanup, and performance optimization.
    
    Handles both in-memory session state and optional temporary file
    storage for very large intermediate results.
    """
    
    def __init__(self, temp_dir: Optional[Path] = None):
        """
        Initialize the state persistence manager.
        
        Args:
            temp_dir: Optional directory for temporary file storage
        """
        self.session_manager = get_session_manager()
        self.logger = logging.getLogger(__name__)
        self.serializer = StateSerializer()
        
        # Setup temporary storage if needed
        self.temp_dir = temp_dir or Path.cwd() / "temp" / "streamlit_state"
        self.temp_dir.mkdir(parents=True, exist_ok=True)
        
        # Track temporary files for cleanup
        self.temp_files: Dict[str, Path] = {}
    
    def persist_large_result(self, key: str, result: PipelineResult, 
                           size_threshold_mb: float = 10.0) -> bool:
        """
        Persist large pipeline results with automatic storage strategy selection.
        
        Args:
            key: Storage key for the result
            result: PipelineResult to persist
            size_threshold_mb: Threshold for file storage vs memory storage
            
        Returns:
            True if successfully persisted
        """
        try:
            # Serialize the result
            serialized_data = self.serializer.serialize_pipeline_result(result)
            
            # Estimate size
            estimated_size = len(json.dumps(serialized_data, default=str).encode('utf-8'))
            size_mb = estimated_size / (1024 * 1024)
            
            if size_mb > size_threshold_mb:
                # Use file storage for large results
                return self._persist_to_file(key, serialized_data, result.total_time)
            else:
                # Use session state for smaller results
                return self._persist_to_session(key, serialized_data)
                
        except Exception as e:
            self.logger.error(f"Failed to persist result {key}: {str(e)}")
            return False
    
    def _persist_to_file(self, key: str, data: Dict[str, Any], processing_time: float) -> bool:
        """
        Persist data to temporary file storage.
        
        Args:
            key: Storage key
            data: Serialized data
            processing_time: Original processing time for metadata
            
        Returns:
            True if successful
        """
        try:
            # Create file path with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{key}_{timestamp}.json"
            file_path = self.temp_dir / filename
            
            # Add metadata
            file_data = {
                'key': key,
                'stored_at': datetime.now().isoformat(),
                'processing_time': processing_time,
                'data_hash': self.serializer.calculate_data_hash(data),
                'data': data
            }
            
            # Write to file
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(file_data, f, indent=2, default=str)
            
            # Store file reference in session state
            self.temp_files[key] = file_path
            st.session_state[f"temp_file_{key}"] = str(file_path)
            
            self.logger.info(f"Large result {key} persisted to file: {file_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to persist to file {key}: {str(e)}")
            return False
    
    def _persist_to_session(self, key: str, data: Dict[str, Any]) -> bool:
        """
        Persist data to session state.
        
        Args:
            key: Storage key
            data: Serialized data
            
        Returns:
            True if successful
        """
        try:
            # Use session manager caching for performance
            cache_key = f"persisted_{key}"
            return self.session_manager.cache_data(cache_key, data)
            
        except Exception as e:
            self.logger.error(f"Failed to persist to session {key}: {str(e)}")
            return False
    
    def retrieve_result(self, key: str) -> Optional[PipelineResult]:
        """
        Retrieve a persisted pipeline result.
        
        Args:
            key: Storage key
            
        Returns:
            Retrieved PipelineResult or None if not found
        """
        try:
            # Try session cache first
            cache_key = f"persisted_{key}"
            cached_data, hit = self.session_manager.get_cached_data(cache_key)
            
            if hit and cached_data:
                return self.serializer.deserialize_pipeline_result(cached_data)
            
            # Try file storage
            return self._retrieve_from_file(key)
            
        except Exception as e:
            self.logger.error(f"Failed to retrieve result {key}: {str(e)}")
            return None
    
    def _retrieve_from_file(self, key: str) -> Optional[PipelineResult]:
        """
        Retrieve data from file storage.
        
        Args:
            key: Storage key
            
        Returns:
            Retrieved PipelineResult or None if not found
        """
        try:
            # Check if we have a file reference
            file_path_str = st.session_state.get(f"temp_file_{key}")
            if not file_path_str:
                return None
            
            file_path = Path(file_path_str)
            if not file_path.exists():
                return None
            
            # Read and deserialize
            with open(file_path, 'r', encoding='utf-8') as f:
                file_data = json.load(f)
            
            # Verify data integrity
            stored_hash = file_data.get('data_hash', '')
            current_hash = self.serializer.calculate_data_hash(file_data['data'])
            
            if stored_hash != current_hash:
                self.logger.warning(f"Data integrity check failed for {key}")
                return None
            
            result = self.serializer.deserialize_pipeline_result(file_data['data'])
            
            # Cache the result for future access
            if result:
                cache_key = f"persisted_{key}"
                self.session_manager.cache_data(cache_key, file_data['data'])
            
            return result
            
        except Exception as e:
            self.logger.error(f"Failed to retrieve from file {key}: {str(e)}")
            return None
    
    def persist_intermediate_state(self, stage: str, state_data: Dict[str, Any]) -> bool:
        """
        Persist intermediate processing state for recovery.
        
        Args:
            stage: Processing stage name
            state_data: State data to persist
            
        Returns:
            True if successful
        """
        try:
            key = f"intermediate_{stage}"
            
            # Add timestamp and recovery metadata
            state_with_meta = {
                'stage': stage,
                'timestamp': datetime.now().isoformat(),
                'session_id': self.session_manager.get_session_metadata().session_id,
                'data': state_data
            }
            
            return self.session_manager.cache_data(key, state_with_meta)
            
        except Exception as e:
            self.logger.error(f"Failed to persist intermediate state {stage}: {str(e)}")
            return False
    
    def retrieve_intermediate_state(self, stage: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve intermediate processing state.
        
        Args:
            stage: Processing stage name
            
        Returns:
            Retrieved state data or None
        """
        try:
            key = f"intermediate_{stage}"
            state_data, hit = self.session_manager.get_cached_data(key)
            
            if hit and state_data:
                return state_data.get('data')
            
            return None
            
        except Exception as e:
            self.logger.error(f"Failed to retrieve intermediate state {stage}: {str(e)}")
            return None
    
    def cleanup_expired_data(self, max_age_hours: int = 24):
        """
        Clean up expired temporary files and cached data.
        
        Args:
            max_age_hours: Maximum age in hours for data retention
        """
        try:
            cutoff_time = datetime.now() - timedelta(hours=max_age_hours)
            cleanup_count = 0
            
            # Clean up temporary files
            for file_path in self.temp_dir.glob("*.json"):
                try:
                    if file_path.stat().st_mtime < cutoff_time.timestamp():
                        file_path.unlink()
                        cleanup_count += 1
                        
                        # Remove from session state
                        for key, path in list(self.temp_files.items()):
                            if path == file_path:
                                if f"temp_file_{key}" in st.session_state:
                                    del st.session_state[f"temp_file_{key}"]
                                del self.temp_files[key]
                                break
                                
                except Exception as file_error:
                    self.logger.warning(f"Failed to clean up file {file_path}: {str(file_error)}")
            
            self.logger.info(f"Cleaned up {cleanup_count} expired files")
            
            # Clean up old cache entries (handled by session manager)
            self.session_manager._evict_cache_entries()
            
        except Exception as e:
            self.logger.error(f"Failed to cleanup expired data: {str(e)}")
    
    def get_persistence_stats(self) -> Dict[str, Any]:
        """
        Get statistics about data persistence.
        
        Returns:
            Dictionary with persistence statistics
        """
        try:
            cache_stats = self.session_manager.get_cache_stats()
            
            # Count temporary files
            temp_file_count = len(list(self.temp_dir.glob("*.json")))
            temp_file_size = sum(f.stat().st_size for f in self.temp_dir.glob("*.json"))
            
            return {
                'cache_stats': {
                    'entries': cache_stats.total_entries,
                    'hit_rate': cache_stats.hit_rate,
                    'size_bytes': cache_stats.total_size_bytes
                },
                'temp_files': {
                    'count': temp_file_count,
                    'size_bytes': temp_file_size,
                    'directory': str(self.temp_dir)
                },
                'session_manager': self.session_manager.export_session_data()
            }
            
        except Exception as e:
            return {'error': str(e)}
    
    def reset_all_persistence(self):
        """Reset all persisted data."""
        try:
            # Clear session state
            self.session_manager.clear_cache()
            
            # Remove temporary files
            for file_path in self.temp_dir.glob("*.json"):
                try:
                    file_path.unlink()
                except Exception:
                    pass  # Best effort cleanup
            
            # Clear temp file tracking
            self.temp_files.clear()
            
            # Remove temp file references from session state
            keys_to_remove = [k for k in st.session_state.keys() if k.startswith("temp_file_")]
            for key in keys_to_remove:
                del st.session_state[key]
            
            self.logger.info("All persistence data reset")
            
        except Exception as e:
            self.logger.error(f"Failed to reset persistence: {str(e)}")


# Global persistence manager instance
_persistence_manager = None

def get_persistence_manager() -> StatePersistenceManager:
    """Get the global state persistence manager instance."""
    global _persistence_manager
    if _persistence_manager is None:
        _persistence_manager = StatePersistenceManager()
    return _persistence_manager


# Convenience functions

def persist_pipeline_result(key: str, result: PipelineResult) -> bool:
    """Persist a pipeline result with automatic storage strategy."""
    return get_persistence_manager().persist_large_result(key, result)

def retrieve_pipeline_result(key: str) -> Optional[PipelineResult]:
    """Retrieve a persisted pipeline result."""
    return get_persistence_manager().retrieve_result(key)

def save_processing_state(stage: str, state_data: Dict[str, Any]) -> bool:
    """Save intermediate processing state for recovery."""
    return get_persistence_manager().persist_intermediate_state(stage, state_data)

def load_processing_state(stage: str) -> Optional[Dict[str, Any]]:
    """Load intermediate processing state."""
    return get_persistence_manager().retrieve_intermediate_state(stage)

def cleanup_old_data(max_age_hours: int = 24):
    """Clean up old persisted data."""
    get_persistence_manager().cleanup_expired_data(max_age_hours)