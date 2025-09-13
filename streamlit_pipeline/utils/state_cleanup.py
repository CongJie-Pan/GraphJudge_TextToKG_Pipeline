"""
State Cleanup Utilities for GraphJudge Streamlit Application

This module provides comprehensive cleanup utilities for managing session state,
temporary files, cached data, and memory usage in the Streamlit application.

Implements cleanup requirements from spec.md Section 9 (state machines) and 
provides automated maintenance for long-running sessions.
"""

import time
import logging
import gc
import threading
from typing import Dict, Any, List, Optional, Callable
from datetime import datetime, timedelta
from pathlib import Path
from dataclasses import dataclass
from enum import Enum

import streamlit as st

from .session_state import SessionStateManager, SessionStateKey, get_session_manager
from .state_persistence import StatePersistenceManager, get_persistence_manager


class CleanupStrategy(Enum):
    """Cleanup strategies for different types of data."""
    
    IMMEDIATE = "immediate"          # Clean up immediately
    DELAYED = "delayed"              # Clean up after delay
    CONDITIONAL = "conditional"      # Clean up based on conditions
    SCHEDULED = "scheduled"          # Clean up on schedule
    MANUAL = "manual"               # Clean up only when requested


@dataclass
class CleanupRule:
    """Defines a cleanup rule for specific data types."""
    
    name: str
    description: str
    strategy: CleanupStrategy
    max_age_minutes: int = 60
    max_size_mb: int = 100
    max_items: int = 1000
    condition_func: Optional[Callable[[], bool]] = None
    priority: int = 5  # 1 = highest priority, 10 = lowest


class StateCleanupManager:
    """
    Comprehensive state cleanup manager for the GraphJudge Streamlit application.
    
    Provides automated and manual cleanup functionality for:
    - Session state data
    - Cached API responses
    - Temporary files
    - Large intermediate results
    - Memory optimization
    
    Ensures optimal performance and prevents memory bloat in long-running sessions.
    """
    
    def __init__(self):
        """Initialize the state cleanup manager."""
        self.session_manager = get_session_manager()
        self.persistence_manager = get_persistence_manager()
        self.logger = logging.getLogger(__name__)
        
        # Define default cleanup rules
        self.cleanup_rules = self._initialize_default_rules()
        
        # Track cleanup statistics
        self.cleanup_stats = {
            'last_cleanup': None,
            'total_cleanups': 0,
            'bytes_cleaned': 0,
            'items_cleaned': 0,
            'errors': 0
        }
        
        # Cleanup lock to prevent concurrent cleanups
        self._cleanup_lock = threading.Lock()
    
    def _initialize_default_rules(self) -> List[CleanupRule]:
        """Initialize default cleanup rules."""
        return [
            CleanupRule(
                name="expired_cache",
                description="Remove expired cache entries",
                strategy=CleanupStrategy.SCHEDULED,
                max_age_minutes=30,
                priority=1
            ),
            CleanupRule(
                name="large_results",
                description="Clean up large pipeline results",
                strategy=CleanupStrategy.CONDITIONAL,
                max_size_mb=50,
                max_items=5,
                priority=2,
                condition_func=lambda: self._check_memory_pressure()
            ),
            CleanupRule(
                name="temp_files",
                description="Remove temporary files",
                strategy=CleanupStrategy.SCHEDULED,
                max_age_minutes=120,
                priority=3
            ),
            CleanupRule(
                name="error_history",
                description="Trim error history",
                strategy=CleanupStrategy.CONDITIONAL,
                max_items=50,
                priority=4,
                condition_func=lambda: len(st.session_state.get(SessionStateKey.ERROR_HISTORY.value, [])) > 100
            ),
            CleanupRule(
                name="old_results",
                description="Remove old pipeline results",
                strategy=CleanupStrategy.CONDITIONAL,
                max_items=10,
                max_age_minutes=240,
                priority=5,
                condition_func=lambda: len(st.session_state.get(SessionStateKey.PIPELINE_RESULTS.value, [])) > 20
            )
        ]
    
    def _check_memory_pressure(self) -> bool:
        """Check if system is under memory pressure."""
        try:
            # Simple memory pressure check based on cache size
            cache_stats = self.session_manager.get_cache_stats()
            return cache_stats.total_size_bytes > 100 * 1024 * 1024  # 100MB
        except Exception:
            return False
    
    def execute_cleanup(self, rule_names: Optional[List[str]] = None, 
                       force: bool = False) -> Dict[str, Any]:
        """
        Execute cleanup based on rules.
        
        Args:
            rule_names: Specific rules to execute (None for all applicable rules)
            force: Force cleanup regardless of conditions
            
        Returns:
            Dictionary with cleanup results and statistics
        """
        if not self._cleanup_lock.acquire(blocking=False):
            return {'status': 'skipped', 'reason': 'cleanup_in_progress'}
        
        try:
            start_time = time.time()
            results = {
                'status': 'success',
                'executed_rules': [],
                'skipped_rules': [],
                'errors': [],
                'statistics': {
                    'items_cleaned': 0,
                    'bytes_cleaned': 0,
                    'duration_seconds': 0
                }
            }
            
            # Determine which rules to execute
            rules_to_execute = self.cleanup_rules
            if rule_names:
                rules_to_execute = [r for r in self.cleanup_rules if r.name in rule_names]
            
            # Sort by priority
            rules_to_execute.sort(key=lambda r: r.priority)
            
            # Execute each rule
            for rule in rules_to_execute:
                try:
                    if self._should_execute_rule(rule, force):
                        rule_result = self._execute_cleanup_rule(rule)
                        results['executed_rules'].append({
                            'rule': rule.name,
                            'result': rule_result
                        })
                        
                        # Update statistics
                        results['statistics']['items_cleaned'] += rule_result.get('items_cleaned', 0)
                        results['statistics']['bytes_cleaned'] += rule_result.get('bytes_cleaned', 0)
                        
                    else:
                        results['skipped_rules'].append(rule.name)
                        
                except Exception as e:
                    error_msg = f"Failed to execute rule {rule.name}: {str(e)}"
                    self.logger.error(error_msg)
                    results['errors'].append(error_msg)
                    self.cleanup_stats['errors'] += 1
            
            # Update global statistics
            duration = time.time() - start_time
            results['statistics']['duration_seconds'] = duration
            
            self.cleanup_stats['last_cleanup'] = datetime.now()
            self.cleanup_stats['total_cleanups'] += 1
            self.cleanup_stats['bytes_cleaned'] += results['statistics']['bytes_cleaned']
            self.cleanup_stats['items_cleaned'] += results['statistics']['items_cleaned']
            
            # Force garbage collection if significant cleanup occurred
            if results['statistics']['bytes_cleaned'] > 10 * 1024 * 1024:  # 10MB
                gc.collect()
            
            self.logger.info(f"Cleanup completed: {len(results['executed_rules'])} rules executed in {duration:.2f}s")
            return results
            
        finally:
            self._cleanup_lock.release()
    
    def _should_execute_rule(self, rule: CleanupRule, force: bool) -> bool:
        """
        Determine if a cleanup rule should be executed.
        
        Args:
            rule: The cleanup rule to evaluate
            force: Force execution regardless of conditions
            
        Returns:
            True if rule should be executed
        """
        if force:
            return True
        
        if rule.strategy == CleanupStrategy.MANUAL:
            return False
        
        if rule.strategy == CleanupStrategy.IMMEDIATE:
            return True
        
        if rule.strategy == CleanupStrategy.CONDITIONAL:
            return rule.condition_func() if rule.condition_func else True
        
        if rule.strategy == CleanupStrategy.SCHEDULED:
            # Check if enough time has passed since last cleanup
            last_cleanup = self.cleanup_stats.get('last_cleanup')
            if not last_cleanup:
                return True
            
            time_since_cleanup = datetime.now() - last_cleanup
            return time_since_cleanup > timedelta(minutes=rule.max_age_minutes)
        
        return False
    
    def _execute_cleanup_rule(self, rule: CleanupRule) -> Dict[str, Any]:
        """
        Execute a specific cleanup rule.
        
        Args:
            rule: The cleanup rule to execute
            
        Returns:
            Dictionary with cleanup results
        """
        result = {
            'rule_name': rule.name,
            'items_cleaned': 0,
            'bytes_cleaned': 0,
            'status': 'success'
        }
        
        try:
            if rule.name == "expired_cache":
                result.update(self._cleanup_expired_cache(rule))
            
            elif rule.name == "large_results":
                result.update(self._cleanup_large_results(rule))
            
            elif rule.name == "temp_files":
                result.update(self._cleanup_temp_files(rule))
            
            elif rule.name == "error_history":
                result.update(self._cleanup_error_history(rule))
            
            elif rule.name == "old_results":
                result.update(self._cleanup_old_results(rule))
            
            else:
                result['status'] = 'unknown_rule'
        
        except Exception as e:
            result['status'] = 'error'
            result['error'] = str(e)
            self.logger.error(f"Error executing cleanup rule {rule.name}: {str(e)}")
        
        return result
    
    def _cleanup_expired_cache(self, rule: CleanupRule) -> Dict[str, Any]:
        """Clean up expired cache entries."""
        try:
            # Get current cache
            cache = st.session_state.get(SessionStateKey.DATA_CACHE.value, {})
            
            cutoff_time = datetime.now() - timedelta(minutes=rule.max_age_minutes)
            items_removed = 0
            bytes_removed = 0
            
            # Identify expired entries
            expired_keys = []
            for key, entry in cache.items():
                if entry.last_accessed < cutoff_time:
                    expired_keys.append(key)
                    bytes_removed += getattr(entry, 'size_bytes', 0)
            
            # Remove expired entries
            for key in expired_keys:
                del cache[key]
                items_removed += 1
            
            # Update cache stats
            if items_removed > 0:
                cache_stats = st.session_state.get(SessionStateKey.CACHE_STATS.value)
                cache_stats.total_entries -= items_removed
                cache_stats.total_size_bytes -= bytes_removed
                cache_stats.eviction_count += items_removed
            
            return {
                'items_cleaned': items_removed,
                'bytes_cleaned': bytes_removed
            }
            
        except Exception as e:
            raise Exception(f"Cache cleanup failed: {str(e)}")
    
    def _cleanup_large_results(self, rule: CleanupRule) -> Dict[str, Any]:
        """Clean up large pipeline results."""
        try:
            results = st.session_state.get(SessionStateKey.PIPELINE_RESULTS.value, [])
            
            if len(results) <= rule.max_items:
                return {'items_cleaned': 0, 'bytes_cleaned': 0}
            
            # Sort by size (approximate) and age, remove largest/oldest
            results_with_size = []
            for i, result in enumerate(results):
                # Estimate size based on content
                estimated_size = self._estimate_result_size(result)
                results_with_size.append((i, result, estimated_size))
            
            # Sort by size descending, then by age
            results_with_size.sort(key=lambda x: (-x[2], -x[0]))
            
            # Remove excess items
            items_to_remove = len(results) - rule.max_items
            removed_indices = []
            bytes_removed = 0
            
            for i in range(items_to_remove):
                idx, result, size = results_with_size[i]
                removed_indices.append(idx)
                bytes_removed += size
            
            # Remove items (in reverse order to maintain indices)
            for idx in sorted(removed_indices, reverse=True):
                del results[idx]
            
            return {
                'items_cleaned': items_to_remove,
                'bytes_cleaned': bytes_removed
            }
            
        except Exception as e:
            raise Exception(f"Large results cleanup failed: {str(e)}")
    
    def _cleanup_temp_files(self, rule: CleanupRule) -> Dict[str, Any]:
        """Clean up temporary files."""
        return self.persistence_manager.cleanup_expired_data(rule.max_age_minutes // 60)
    
    def _cleanup_error_history(self, rule: CleanupRule) -> Dict[str, Any]:
        """Clean up error history."""
        try:
            error_history = st.session_state.get(SessionStateKey.ERROR_HISTORY.value, [])
            
            if len(error_history) <= rule.max_items:
                return {'items_cleaned': 0, 'bytes_cleaned': 0}
            
            # Keep only the most recent errors
            items_to_remove = len(error_history) - rule.max_items
            error_history[:] = error_history[-rule.max_items:]
            
            return {
                'items_cleaned': items_to_remove,
                'bytes_cleaned': items_to_remove * 100  # Rough estimate
            }
            
        except Exception as e:
            raise Exception(f"Error history cleanup failed: {str(e)}")
    
    def _cleanup_old_results(self, rule: CleanupRule) -> Dict[str, Any]:
        """Clean up old pipeline results."""
        try:
            results = st.session_state.get(SessionStateKey.PIPELINE_RESULTS.value, [])
            
            if len(results) <= rule.max_items:
                return {'items_cleaned': 0, 'bytes_cleaned': 0}
            
            # Remove oldest results beyond the limit
            items_to_remove = len(results) - rule.max_items
            removed_results = results[:items_to_remove]
            results[:] = results[items_to_remove:]
            
            bytes_removed = sum(self._estimate_result_size(r) for r in removed_results)
            
            return {
                'items_cleaned': items_to_remove,
                'bytes_cleaned': bytes_removed
            }
            
        except Exception as e:
            raise Exception(f"Old results cleanup failed: {str(e)}")
    
    def _estimate_result_size(self, result) -> int:
        """Estimate the size of a pipeline result."""
        try:
            # Rough estimation based on content
            size = 1000  # Base size
            
            if hasattr(result, 'entity_result') and result.entity_result:
                size += len(result.entity_result.entities or []) * 100
                size += len(result.entity_result.denoised_text or '') * 2
            
            if hasattr(result, 'triple_result') and result.triple_result:
                size += len(result.triple_result.triples or []) * 200
            
            if hasattr(result, 'judgment_result') and result.judgment_result:
                size += len(result.judgment_result.judgments or []) * 50
                size += len(result.judgment_result.explanations or []) * 300
            
            return size
            
        except Exception:
            return 1000  # Default estimate
    
    def schedule_automatic_cleanup(self, interval_minutes: int = 30):
        """
        Schedule automatic cleanup to run periodically.
        
        Args:
            interval_minutes: Cleanup interval in minutes
        """
        try:
            # Store cleanup schedule in session state
            st.session_state['cleanup_schedule'] = {
                'enabled': True,
                'interval_minutes': interval_minutes,
                'next_cleanup': datetime.now() + timedelta(minutes=interval_minutes)
            }
            
            self.logger.info(f"Automatic cleanup scheduled every {interval_minutes} minutes")
            
        except Exception as e:
            self.logger.error(f"Failed to schedule automatic cleanup: {str(e)}")
    
    def check_scheduled_cleanup(self):
        """Check if scheduled cleanup should run."""
        try:
            schedule = st.session_state.get('cleanup_schedule')
            if not schedule or not schedule.get('enabled'):
                return
            
            next_cleanup = schedule.get('next_cleanup')
            if next_cleanup and datetime.now() >= next_cleanup:
                # Execute scheduled cleanup
                result = self.execute_cleanup()
                
                # Schedule next cleanup
                schedule['next_cleanup'] = datetime.now() + timedelta(
                    minutes=schedule['interval_minutes']
                )
                
                self.logger.info("Scheduled cleanup executed successfully")
                
        except Exception as e:
            self.logger.error(f"Scheduled cleanup check failed: {str(e)}")
    
    def force_complete_cleanup(self):
        """Force a complete cleanup of all data."""
        try:
            # Execute all cleanup rules
            self.execute_cleanup(force=True)
            
            # Additional complete cleanup
            self.session_manager.reset_session()
            self.persistence_manager.reset_all_persistence()
            
            # Force garbage collection
            gc.collect()
            
            self.logger.info("Complete cleanup executed")
            
        except Exception as e:
            self.logger.error(f"Complete cleanup failed: {str(e)}")
    
    def get_cleanup_stats(self) -> Dict[str, Any]:
        """Get cleanup statistics and current state."""
        try:
            return {
                'cleanup_stats': self.cleanup_stats.copy(),
                'session_stats': self.session_manager.export_session_data(),
                'persistence_stats': self.persistence_manager.get_persistence_stats(),
                'memory_info': self._get_memory_info(),
                'cleanup_rules': [
                    {
                        'name': rule.name,
                        'description': rule.description,
                        'strategy': rule.strategy.value,
                        'priority': rule.priority
                    }
                    for rule in self.cleanup_rules
                ]
            }
            
        except Exception as e:
            return {'error': str(e)}
    
    def _get_memory_info(self) -> Dict[str, Any]:
        """Get memory usage information."""
        try:
            import psutil
            import os
            
            process = psutil.Process(os.getpid())
            memory_info = process.memory_info()
            
            return {
                'rss_mb': memory_info.rss / 1024 / 1024,
                'vms_mb': memory_info.vms / 1024 / 1024,
                'percent': process.memory_percent()
            }
            
        except ImportError:
            return {'error': 'psutil not available'}
        except Exception as e:
            return {'error': str(e)}


# Global cleanup manager instance
_cleanup_manager = None

def get_cleanup_manager() -> StateCleanupManager:
    """Get the global state cleanup manager instance."""
    global _cleanup_manager
    if _cleanup_manager is None:
        _cleanup_manager = StateCleanupManager()
    return _cleanup_manager


# Convenience functions

def cleanup_expired_data():
    """Clean up expired data using default rules."""
    return get_cleanup_manager().execute_cleanup(['expired_cache', 'temp_files'])

def force_cleanup_all():
    """Force cleanup of all data."""
    get_cleanup_manager().force_complete_cleanup()

def schedule_cleanup(interval_minutes: int = 30):
    """Schedule automatic cleanup."""
    get_cleanup_manager().schedule_automatic_cleanup(interval_minutes)

def check_and_run_cleanup():
    """Check if cleanup should run and execute if needed."""
    get_cleanup_manager().check_scheduled_cleanup()