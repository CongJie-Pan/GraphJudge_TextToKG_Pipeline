"""
Disk caching for API responses.

This module provides intelligent disk caching for GPT-5-mini API responses,
reducing redundant API calls and improving performance for repeated requests.
"""

import json
import hashlib
import os
import shutil
from pathlib import Path
from typing import Dict, Optional, Any, List, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import asyncio
from extractEntity_Phase.infrastructure.logging import get_logger
from extractEntity_Phase.infrastructure.config import get_config


@dataclass
class CacheConfig:
    """Configuration for caching."""
    enabled: bool = True
    cache_dir: str = "./cache"
    max_size_mb: int = 100
    ttl_hours: int = 24
    cleanup_interval_hours: int = 6
    compression_enabled: bool = False
    metadata_enabled: bool = True


@dataclass
class CacheEntry:
    """A single cache entry."""
    key: str
    response: str
    prompt: str
    system_prompt: Optional[str] = None
    timestamp: float = field(default_factory=asyncio.get_event_loop().time)
    model: str = "gpt-5-mini"
    temperature: float = 1.0
    max_tokens: int = 4000
    token_count: int = 0
    file_size: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "key": self.key,
            "response": self.response,
            "prompt": self.prompt,
            "system_prompt": self.system_prompt,
            "timestamp": self.timestamp,
            "model": self.model,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "token_count": self.token_count,
            "file_size": self.file_size
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'CacheEntry':
        """Create from dictionary."""
        return cls(**data)


class CacheManager:
    """
    Intelligent disk cache manager for API responses.
    
    Features:
    - Disk-based caching with JSON storage
    - TTL-based expiration
    - Size-based cleanup
    - Metadata tracking
    - Cache hit/miss statistics
    - Automatic cleanup
    """
    
    def __init__(self, config: Optional[CacheConfig] = None):
        """
        Initialize cache manager.
        
        Args:
            config: Caching configuration
        """
        self.config = config or CacheConfig()
        self.logger = get_logger()
        
        # Initialize cache directory
        self.cache_dir = Path(self.config.cache_dir)
        self._setup_cache_directory()
        
        # Statistics
        self.cache_hits = 0
        self.cache_misses = 0
        self.cache_writes = 0
        self.cache_errors = 0
        
        # Cache index for fast lookups
        self._cache_index: Dict[str, CacheEntry] = {}
        self._load_cache_index()
    
    def _setup_cache_directory(self) -> None:
        """Set up cache directory structure."""
        if not self.config.enabled:
            # When disabled, ensure cache_dir points to a non-existent path for testing
            self.cache_dir = Path("./disabled_cache")
            return
        
        # Handle empty cache directory path
        if not self.config.cache_dir or self.config.cache_dir.strip() == "":
            self.config.enabled = False
            self.cache_dir = Path("./disabled_cache")
            return
        
        try:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            self.logger.info(f"Cache directory initialized: {self.cache_dir}")
        except OSError as e:
            self.logger.error(f"Failed to create cache directory: {e}")
            self.config.enabled = False
    
    def _load_cache_index(self) -> None:
        """Load cache index from disk."""
        if not self.config.enabled:
            return
        
        try:
            index_file = self.cache_dir / "cache_index.json"
            if index_file.exists():
                with open(index_file, 'r', encoding='utf-8') as f:
                    index_data = json.load(f)
                    self._cache_index = {
                        key: CacheEntry.from_dict(entry_data)
                        for key, entry_data in index_data.items()
                    }
                self.logger.info(f"Loaded {len(self._cache_index)} cache entries")
        except Exception as e:
            self.logger.warning(f"Failed to load cache index: {e}")
            self._cache_index = {}
    
    def _save_cache_index(self) -> None:
        """Save cache index to disk."""
        if not self.config.enabled:
            return
        
        try:
            index_file = self.cache_dir / "cache_index.json"
            index_data = {
                key: entry.to_dict()
                for key, entry in self._cache_index.items()
            }
            
            with open(index_file, 'w', encoding='utf-8') as f:
                json.dump(index_data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            self.logger.error(f"Failed to save cache index: {e}")
    
    def generate_cache_key(self, prompt: str, system_prompt: Optional[str] = None, **kwargs) -> str:
        """
        Generate a unique cache key for API requests.
        
        Args:
            prompt: User prompt content
            system_prompt: System prompt content
            **kwargs: Additional API parameters
            
        Returns:
            SHA256 hash as cache key
        """
        # Combine all inputs that affect the API response
        cache_content = {
            "prompt": prompt,
            "system_prompt": system_prompt or "",
            "temperature": kwargs.get("temperature", 1.0),
            "max_tokens": kwargs.get("max_tokens", 4000),
            "model": kwargs.get("model", "gpt-5-mini"),
            **{k: v for k, v in kwargs.items() if k not in ["temperature", "max_tokens", "model"]}
        }
        
        # Create deterministic hash from content
        content_str = json.dumps(cache_content, sort_keys=True, ensure_ascii=False)
        hash_obj = hashlib.sha256(content_str.encode('utf-8'))
        return hash_obj.hexdigest()
    
    def get_cache_file_path(self, cache_key: str) -> Path:
        """
        Get the file path for a cache key.
        
        Args:
            cache_key: Cache key
            
        Returns:
            Path to cache file
        """
        return self.cache_dir / f"{cache_key}.json"
    
    def get(self, cache_key: str) -> Optional[str]:
        """
        Retrieve cached response.
        
        Args:
            cache_key: Cache key
            
        Returns:
            Cached response or None if not found/expired
        """
        if not self.config.enabled:
            self.cache_misses += 1
            return None
        
        # Check in-memory index first
        if cache_key in self._cache_index:
            entry = self._cache_index[cache_key]
            
            # Check if expired
            if self._is_expired(entry):
                self._remove_entry(cache_key)
                self.cache_misses += 1
                return None
            
            # Check if file exists
            cache_file = self.get_cache_file_path(cache_key)
            if not cache_file.exists():
                self._remove_entry(cache_key)
                self.cache_misses += 1
                return None
            
            try:
                with open(cache_file, 'r', encoding='utf-8') as f:
                    cache_data = json.load(f)
                    response = cache_data.get('response')
                    if response:
                        self.cache_hits += 1
                        self.logger.debug(f"Cache hit for key {cache_key[:8]}...")
                        return response
            except Exception as e:
                self.logger.warning(f"Cache read error for key {cache_key[:8]}...: {e}")
                self._remove_entry(cache_key)
                self.cache_errors += 1
        
        self.cache_misses += 1
        return None
    
    def set(self, cache_key: str, prompt: str, response: str, 
            system_prompt: Optional[str] = None, **kwargs) -> bool:
        """
        Store response in cache.
        
        Args:
            cache_key: Cache key
            prompt: Original user prompt
            response: API response content
            system_prompt: System prompt if used
            **kwargs: Additional metadata
            
        Returns:
            True if successfully cached, False otherwise
        """
        if not self.config.enabled:
            return False
        
        try:
            # Create cache entry
            entry = CacheEntry(
                key=cache_key,
                response=response,
                prompt=prompt,
                system_prompt=system_prompt,
                model=kwargs.get("model", "gpt-5-mini"),
                temperature=kwargs.get("temperature", 1.0),
                max_tokens=kwargs.get("max_tokens", 4000),
                token_count=kwargs.get("token_count", 0)
            )
            
            # Save to disk
            cache_file = self.get_cache_file_path(cache_key)
            cache_data = entry.to_dict()
            
            with open(cache_file, 'w', encoding='utf-8') as f:
                json.dump(cache_data, f, ensure_ascii=False, indent=2)
            
            # Update file size
            entry.file_size = cache_file.stat().st_size
            
            # Update in-memory index
            self._cache_index[cache_key] = entry
            self._save_cache_index()
            
            self.cache_writes += 1
            self.logger.debug(f"Cached response for key {cache_key[:8]}...")
            
            # Check if cleanup is needed (only for large entries)
            if len(response) > 1024 * 1024:  # 1MB threshold
                self._check_cleanup_needed()
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to cache response for key {cache_key[:8]}...: {e}")
            self.cache_errors += 1
            return False
    
    def _is_expired(self, entry: CacheEntry) -> bool:
        """
        Check if cache entry is expired.
        
        Args:
            entry: Cache entry to check
            
        Returns:
            True if expired, False otherwise
        """
        current_time = asyncio.get_event_loop().time()
        expiration_time = entry.timestamp + (self.config.ttl_hours * 3600)
        return current_time > expiration_time
    
    def _remove_entry(self, cache_key: str) -> None:
        """
        Remove cache entry.
        
        Args:
            cache_key: Cache key to remove
        """
        try:
            # Remove from index
            if cache_key in self._cache_index:
                del self._cache_index[cache_key]
            
            # Remove file
            cache_file = self.get_cache_file_path(cache_key)
            if cache_file.exists():
                cache_file.unlink()
            
            self._save_cache_index()
            
        except Exception as e:
            self.logger.warning(f"Failed to remove cache entry {cache_key[:8]}...: {e}")
    
    def _check_cleanup_needed(self) -> None:
        """Check if cache cleanup is needed."""
        if not self.config.enabled:
            return
            
        current_time = asyncio.get_event_loop().time()
        
        # Check if cleanup interval has passed
        if hasattr(self, '_last_cleanup'):
            if current_time - self._last_cleanup < (self.config.cleanup_interval_hours * 3600):
                return
        else:
            self._last_cleanup = current_time
        
        # Perform cleanup
        self.cleanup()
        self._last_cleanup = current_time
    
    def cleanup(self, force: bool = False) -> None:
        """
        Clean up expired and oversized cache entries.
        
        Args:
            force: Force cleanup even if not due
        """
        if not self.config.enabled:
            return
        
        try:
            self.logger.info("Starting cache cleanup...")
            
            # Remove expired entries
            expired_keys = [
                key for key, entry in self._cache_index.items()
                if self._is_expired(entry)
            ]
            
            for key in expired_keys:
                self._remove_entry(key)
            
            if expired_keys:
                self.logger.info(f"Removed {len(expired_keys)} expired cache entries")
            
            # Check size limits
            self._enforce_size_limits()
            
            self.logger.info("Cache cleanup completed")
            
        except Exception as e:
            self.logger.error(f"Cache cleanup failed: {e}")
    
    def _enforce_size_limits(self) -> None:
        """Enforce cache size limits."""
        if not self.config.enabled:
            return
        
        try:
            # Calculate current cache size
            total_size = 0
            entry_sizes = []
            
            for key, entry in self._cache_index.items():
                cache_file = self.get_cache_file_path(key)
                if cache_file.exists():
                    file_size = cache_file.stat().st_size
                    total_size += file_size
                    entry_sizes.append((key, file_size, entry.timestamp))
            
            # Convert to MB
            total_size_mb = total_size / (1024 * 1024)
            
            if total_size_mb <= self.config.max_size_mb:
                return
            
            # Sort by timestamp (oldest first)
            entry_sizes.sort(key=lambda x: x[2])
            
            # Remove oldest entries until under limit
            target_size = self.config.max_size_mb * 1024 * 1024  # Convert back to bytes
            removed_count = 0
            
            for key, file_size, _ in entry_sizes:
                if total_size <= target_size:
                    break
                
                self._remove_entry(key)
                total_size -= file_size
                removed_count += 1
            
            if removed_count > 0:
                self.logger.info(f"Removed {removed_count} cache entries to enforce size limit")
                
        except Exception as e:
            self.logger.error(f"Failed to enforce size limits: {e}")
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics.
        
        Returns:
            Dictionary with cache statistics
        """
        if not self.config.enabled:
            return {"enabled": False}
        
        # Calculate current cache size
        total_size = 0
        entry_count = len(self._cache_index)
        
        for key in self._cache_index:
            cache_file = self.get_cache_file_path(key)
            if cache_file.exists():
                total_size += cache_file.stat().st_size
        
        total_size_mb = total_size / (1024 * 1024)
        
        # Calculate hit rate
        total_requests = self.cache_hits + self.cache_misses
        hit_rate = (self.cache_hits / total_requests * 100) if total_requests > 0 else 0
        
        return {
            "enabled": True,
            "cache_dir": str(self.cache_dir),
            "entry_count": entry_count,
            "total_size_mb": round(total_size_mb, 2),
            "max_size_mb": self.config.max_size_mb,
            "ttl_hours": self.config.ttl_hours,
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "cache_writes": self.cache_writes,
            "cache_errors": self.cache_errors,
            "hit_rate_percent": round(hit_rate, 2),
            "total_requests": total_requests
        }
    
    def clear(self) -> None:
        """Clear all cache entries."""
        if not self.config.enabled:
            return
        
        try:
            # Remove all cache files including index
            for cache_file in self.cache_dir.glob("*.json"):
                cache_file.unlink()
            
            # Clear index
            self._cache_index.clear()
            
            # Reset statistics
            self.cache_hits = 0
            self.cache_misses = 0
            self.cache_writes = 0
            self.cache_errors = 0
            
            self.logger.info("Cache cleared")
            
        except Exception as e:
            self.logger.error(f"Failed to clear cache: {e}")
    
    def get_cache_info(self, cache_key: str) -> Optional[Dict[str, Any]]:
        """
        Get information about a specific cache entry.
        
        Args:
            cache_key: Cache key
            
        Returns:
            Cache entry information or None if not found
        """
        if not self.config.enabled or cache_key not in self._cache_index:
            return None
        
        entry = self._cache_index[cache_key]
        cache_file = self.get_cache_file_path(cache_key)
        
        info = entry.to_dict()
        info["exists"] = cache_file.exists()
        info["expired"] = self._is_expired(entry)
        
        if cache_file.exists():
            info["file_size"] = cache_file.stat().st_size
            info["last_modified"] = datetime.fromtimestamp(cache_file.stat().st_mtime).isoformat()
        
        return info


# Convenience functions for backward compatibility
def create_cache_manager(config: Optional[CacheConfig] = None) -> CacheManager:
    """Create a cache manager instance."""
    return CacheManager(config)


def get_default_cache_manager() -> CacheManager:
    """Get a cache manager with default configuration."""
    return CacheManager()


def generate_cache_key(prompt: str, system_prompt: Optional[str] = None, **kwargs) -> str:
    """Generate a cache key for backward compatibility."""
    cache_manager = get_default_cache_manager()
    return cache_manager.generate_cache_key(prompt, system_prompt, **kwargs)
