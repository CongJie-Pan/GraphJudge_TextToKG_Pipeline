"""
Tests for disk caching for API responses.
"""

import pytest
import json
import os
import tempfile
import shutil
from pathlib import Path
from unittest.mock import patch, MagicMock, mock_open
from extractEntity_Phase.api.cache_manager import (
    CacheConfig, CacheEntry, CacheManager,
    create_cache_manager, get_default_cache_manager, generate_cache_key
)


class TestCacheConfig:
    """Test cache configuration dataclass."""
    
    def test_cache_config_creation(self):
        """Test cache config creation with all fields."""
        config = CacheConfig(
            enabled=True,
            cache_dir="./test_cache",
            max_size_mb=200,
            ttl_hours=48,
            cleanup_interval_hours=12,
            compression_enabled=True,
            metadata_enabled=False
        )
        
        assert config.enabled == True
        assert config.cache_dir == "./test_cache"
        assert config.max_size_mb == 200
        assert config.ttl_hours == 48
        assert config.cleanup_interval_hours == 12
        assert config.compression_enabled == True
        assert config.metadata_enabled == False
    
    def test_cache_config_defaults(self):
        """Test cache config default values."""
        config = CacheConfig()
        
        assert config.enabled == True
        assert config.cache_dir == "./cache"
        assert config.max_size_mb == 100
        assert config.ttl_hours == 24
        assert config.cleanup_interval_hours == 6
        assert config.compression_enabled == False
        assert config.metadata_enabled == True


class TestCacheEntry:
    """Test cache entry dataclass."""
    
    def test_cache_entry_creation(self):
        """Test cache entry creation with all fields."""
        entry = CacheEntry(
            key="test_key_123",
            response="test response content",
            prompt="test prompt",
            system_prompt="test system prompt",
            timestamp=1234567890.0,
            model="gpt-5-mini",
            temperature=0.8,
            max_tokens=2000,
            token_count=1500,
            file_size=1024
        )
        
        assert entry.key == "test_key_123"
        assert entry.response == "test response content"
        assert entry.prompt == "test prompt"
        assert entry.system_prompt == "test system prompt"
        assert entry.timestamp == 1234567890.0
        assert entry.model == "gpt-5-mini"
        assert entry.temperature == 0.8
        assert entry.max_tokens == 2000
        assert entry.token_count == 1500
        assert entry.file_size == 1024
    
    def test_cache_entry_defaults(self):
        """Test cache entry creation with default values."""
        entry = CacheEntry(
            key="test_key",
            response="test response",
            prompt="test prompt"
        )
        
        assert entry.key == "test_key"
        assert entry.response == "test response"
        assert entry.prompt == "test prompt"
        assert entry.system_prompt is None
        assert entry.timestamp > 0
        assert entry.model == "gpt-5-mini"
        assert entry.temperature == 1.0
        assert entry.max_tokens == 4000
        assert entry.token_count == 0
        assert entry.file_size == 0
    
    def test_cache_entry_to_dict(self):
        """Test converting cache entry to dictionary."""
        entry = CacheEntry(
            key="test_key",
            response="test response",
            prompt="test prompt",
            system_prompt="test system",
            timestamp=1234567890.0,
            model="gpt-5-mini",
            temperature=0.7,
            max_tokens=3000,
            token_count=2000,
            file_size=2048
        )
        
        entry_dict = entry.to_dict()
        
        assert isinstance(entry_dict, dict)
        assert entry_dict["key"] == "test_key"
        assert entry_dict["response"] == "test response"
        assert entry_dict["prompt"] == "test prompt"
        assert entry_dict["system_prompt"] == "test system"
        assert entry_dict["timestamp"] == 1234567890.0
        assert entry_dict["model"] == "gpt-5-mini"
        assert entry_dict["temperature"] == 0.7
        assert entry_dict["max_tokens"] == 3000
        assert entry_dict["token_count"] == 2000
        assert entry_dict["file_size"] == 2048
    
    def test_cache_entry_from_dict(self):
        """Test creating cache entry from dictionary."""
        entry_data = {
            "key": "test_key",
            "response": "test response",
            "prompt": "test prompt",
            "system_prompt": "test system",
            "timestamp": 1234567890.0,
            "model": "gpt-5-mini",
            "temperature": 0.7,
            "max_tokens": 3000,
            "token_count": 2000,
            "file_size": 2048
        }
        
        entry = CacheEntry.from_dict(entry_data)
        
        assert entry.key == "test_key"
        assert entry.response == "test response"
        assert entry.prompt == "test prompt"
        assert entry.system_prompt == "test system"
        assert entry.timestamp == 1234567890.0
        assert entry.model == "gpt-5-mini"
        assert entry.temperature == 0.7
        assert entry.max_tokens == 3000
        assert entry.token_count == 2000
        assert entry.file_size == 2048


class TestCacheManager:
    """Test cache manager class."""
    
    @pytest.fixture
    def temp_cache_dir(self):
        """Create temporary cache directory for testing."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    def test_cache_manager_creation(self):
        """Test cache manager creation."""
        config = CacheConfig(cache_dir="./test_cache", max_size_mb=200)
        cache_manager = CacheManager(config)
        
        assert cache_manager.config == config
        assert cache_manager.config.cache_dir == "./test_cache"
        assert cache_manager.config.max_size_mb == 200
    
    def test_cache_manager_default_creation(self):
        """Test cache manager creation with default config."""
        cache_manager = CacheManager()
        
        assert cache_manager.config.enabled == True
        assert cache_manager.config.cache_dir == "./cache"
        assert cache_manager.config.max_size_mb == 100
        assert cache_manager.config.ttl_hours == 24
    
    def test_cache_manager_disabled(self):
        """Test cache manager when caching is disabled."""
        config = CacheConfig(enabled=False)
        cache_manager = CacheManager(config)
        
        # Should not create cache directory
        assert not cache_manager.cache_dir.exists()
        
        # Should not cache responses
        result = cache_manager.set("test_key", "test_prompt", "test_response")
        assert result == False
        
        # Should always miss cache
        cached = cache_manager.get("test_key")
        assert cached is None
    
    def test_setup_cache_directory(self, temp_cache_dir):
        """Test cache directory setup."""
        config = CacheConfig(cache_dir=temp_cache_dir)
        cache_manager = CacheManager(config)
        
        # Directory should be created
        assert Path(temp_cache_dir).exists()
        assert Path(temp_cache_dir).is_dir()
    
    def test_setup_cache_directory_error(self):
        """Test cache directory setup with error."""
        # Use invalid path that should cause error
        invalid_path = "/invalid/path/that/should/fail"
        
        with patch('pathlib.Path.mkdir') as mock_mkdir:
            mock_mkdir.side_effect = OSError("Permission denied")
            
            config = CacheConfig(cache_dir=invalid_path)
            cache_manager = CacheManager(config)
            
            # Should disable caching on error
            assert cache_manager.config.enabled == False
    
    def test_generate_cache_key(self):
        """Test cache key generation."""
        cache_manager = CacheManager()
        
        # Test basic key generation
        key1 = cache_manager.generate_cache_key("test prompt")
        key2 = cache_manager.generate_cache_key("test prompt")
        
        # Same inputs should generate same key
        assert key1 == key2
        assert len(key1) == 64  # SHA256 hash length
        
        # Different inputs should generate different keys
        key3 = cache_manager.generate_cache_key("different prompt")
        assert key1 != key3
    
    def test_generate_cache_key_with_system_prompt(self):
        """Test cache key generation with system prompt."""
        cache_manager = CacheManager()
        
        key1 = cache_manager.generate_cache_key("test prompt", "system prompt")
        key2 = cache_manager.generate_cache_key("test prompt", "system prompt")
        key3 = cache_manager.generate_cache_key("test prompt", "different system")
        
        # Same inputs should generate same key
        assert key1 == key2
        
        # Different system prompt should generate different key
        assert key1 != key3
    
    def test_generate_cache_key_with_kwargs(self):
        """Test cache key generation with additional parameters."""
        cache_manager = CacheManager()
        
        key1 = cache_manager.generate_cache_key("test prompt", temperature=0.7, max_tokens=2000)
        key2 = cache_manager.generate_cache_key("test prompt", temperature=0.7, max_tokens=2000)
        key3 = cache_manager.generate_cache_key("test prompt", temperature=0.8, max_tokens=2000)
        
        # Same parameters should generate same key
        assert key1 == key2
        
        # Different parameters should generate different key
        assert key1 != key3
    
    def test_get_cache_file_path(self):
        """Test getting cache file path."""
        cache_manager = CacheManager()
        
        cache_key = "test_key_123"
        file_path = cache_manager.get_cache_file_path(cache_key)
        
        expected_path = Path(cache_manager.config.cache_dir) / f"{cache_key}.json"
        assert file_path == expected_path
    
    def test_set_and_get_cache(self, temp_cache_dir):
        """Test setting and getting cache entries."""
        config = CacheConfig(cache_dir=temp_cache_dir)
        cache_manager = CacheManager(config)
        
        # Set cache entry
        cache_key = "test_key_123"
        prompt = "test prompt"
        response = "test response"
        system_prompt = "test system prompt"
        
        result = cache_manager.set(
            cache_key, prompt, response, system_prompt,
            model="gpt-5-mini", temperature=0.7, max_tokens=2000, token_count=1500
        )
        
        assert result == True
        
        # Get cached response
        cached_response = cache_manager.get(cache_key)
        
        assert cached_response == response
        
        # Check cache file exists
        cache_file = cache_manager.get_cache_file_path(cache_key)
        assert cache_file.exists()
        
        # Check cache index was updated
        assert cache_key in cache_manager._cache_index
    
    def test_get_cache_miss(self, temp_cache_dir):
        """Test cache miss scenario."""
        config = CacheConfig(cache_dir=temp_cache_dir)
        cache_manager = CacheManager(config)
        
        # Try to get non-existent cache
        cached_response = cache_manager.get("non_existent_key")
        
        assert cached_response is None
        assert cache_manager.cache_misses == 1
    
    def test_get_cache_expired(self, temp_cache_dir):
        """Test getting expired cache entry."""
        config = CacheConfig(cache_dir=temp_cache_dir, ttl_hours=1)
        cache_manager = CacheManager(config)
        
        # Create expired cache entry
        cache_key = "expired_key"
        entry = CacheEntry(
            key=cache_key,
            response="expired response",
            prompt="expired prompt",
            timestamp=0.0  # Very old timestamp
        )
        
        cache_manager._cache_index[cache_key] = entry
        
        # Try to get expired cache
        cached_response = cache_manager.get(cache_key)
        
        assert cached_response is None
        assert cache_key not in cache_manager._cache_index
    
    def test_get_cache_file_missing(self, temp_cache_dir):
        """Test getting cache when file is missing."""
        config = CacheConfig(cache_dir=temp_cache_dir)
        cache_manager = CacheManager(config)
        
        # Create cache entry but don't create file
        cache_key = "missing_file_key"
        entry = CacheEntry(
            key=cache_key,
            response="missing file response",
            prompt="missing file prompt"
        )
        
        cache_manager._cache_index[cache_key] = entry
        
        # Try to get cache with missing file
        cached_response = cache_manager.get(cache_key)
        
        assert cached_response is None
        assert cache_key not in cache_manager._cache_index
    
    def test_get_cache_read_error(self, temp_cache_dir):
        """Test getting cache when file read fails."""
        config = CacheConfig(cache_dir=temp_cache_dir)
        cache_manager = CacheManager(config)
        
        # Create cache entry and file
        cache_key = "read_error_key"
        entry = CacheEntry(
            key=cache_key,
            response="read error response",
            prompt="read error prompt"
        )
        
        cache_manager._cache_index[cache_key] = entry
        
        # Create corrupted cache file
        cache_file = cache_manager.get_cache_file_path(cache_key)
        with open(cache_file, 'w') as f:
            f.write("invalid json content")
        
        # Try to get cache with read error
        cached_response = cache_manager.get(cache_key)
        
        assert cached_response is None
        assert cache_key not in cache_manager._cache_index
        assert cache_manager.cache_errors == 1
    
    def test_set_cache_error(self, temp_cache_dir):
        """Test setting cache when error occurs."""
        config = CacheConfig(cache_dir=temp_cache_dir)
        cache_manager = CacheManager(config)
        
        # Mock file write to raise error
        with patch('builtins.open') as mock_open:
            mock_open.side_effect = OSError("Write error")
            
            result = cache_manager.set("error_key", "prompt", "response")
            
            assert result == False
            assert cache_manager.cache_errors == 1
    
    def test_cache_cleanup(self, temp_cache_dir):
        """Test cache cleanup functionality."""
        config = CacheConfig(cache_dir=temp_cache_dir, ttl_hours=1)
        cache_manager = CacheManager(config)
        
        # Create some cache entries
        cache_manager.set("key1", "prompt1", "response1")
        cache_manager.set("key2", "prompt2", "response2")
        
        # Create expired entry
        expired_entry = CacheEntry(
            key="expired_key",
            response="expired response",
            prompt="expired prompt",
            timestamp=0.0  # Very old
        )
        cache_manager._cache_index["expired_key"] = expired_entry
        
        # Force cleanup
        cache_manager.cleanup(force=True)
        
        # Expired entry should be removed
        assert "expired_key" not in cache_manager._cache_index
        assert len(cache_manager._cache_index) == 2
    
    def test_cache_size_limit_enforcement(self, temp_cache_dir):
        """Test cache size limit enforcement."""
        config = CacheConfig(cache_dir=temp_cache_dir, max_size_mb=1)  # Very small limit
        cache_manager = CacheManager(config)
        
        # Create large cache entries
        large_response = "x" * (1024 * 1024)  # 1MB
        
        cache_manager.set("large_key1", "prompt1", large_response)
        cache_manager.set("large_key2", "prompt2", large_response)
        
        # Force cleanup to enforce size limits
        cache_manager.cleanup(force=True)
        
        # Should have removed some entries to stay under limit
        assert len(cache_manager._cache_index) < 2
    
    def test_get_stats(self, temp_cache_dir):
        """Test getting cache statistics."""
        config = CacheConfig(cache_dir=temp_cache_dir)
        cache_manager = CacheManager(config)
        
        # Add some cache entries
        cache_manager.set("key1", "prompt1", "response1")
        cache_manager.set("key2", "prompt2", "response2")
        
        # Get some cache hits and misses
        cache_manager.get("key1")  # Hit
        cache_manager.get("key2")  # Hit
        cache_manager.get("non_existent")  # Miss
        
        stats = cache_manager.get_stats()
        
        assert stats["enabled"] == True
        assert stats["cache_dir"] == temp_cache_dir
        assert stats["entry_count"] == 2
        assert stats["cache_hits"] == 2
        assert stats["cache_misses"] == 1
        assert stats["cache_writes"] == 2
        assert stats["cache_errors"] == 0
        assert stats["total_requests"] == 3
        assert "hit_rate_percent" in stats
    
    def test_get_stats_disabled(self):
        """Test getting stats when caching is disabled."""
        config = CacheConfig(enabled=False)
        cache_manager = CacheManager(config)
        
        stats = cache_manager.get_stats()
        
        assert stats["enabled"] == False
    
    def test_clear_cache(self, temp_cache_dir):
        """Test clearing all cache entries."""
        config = CacheConfig(cache_dir=temp_cache_dir)
        cache_manager = CacheManager(config)
        
        # Add some cache entries
        cache_manager.set("key1", "prompt1", "response1")
        cache_manager.set("key2", "prompt2", "response2")
        
        # Clear cache
        cache_manager.clear()
        
        # All entries should be removed
        assert len(cache_manager._cache_index) == 0
        assert cache_manager.cache_hits == 0
        assert cache_manager.cache_misses == 0
        assert cache_manager.cache_writes == 0
        assert cache_manager.cache_errors == 0
        
        # Cache files should be removed
        cache_files = list(Path(temp_cache_dir).glob("*.json"))
        assert len(cache_files) == 0
    
    def test_get_cache_info(self, temp_cache_dir):
        """Test getting information about specific cache entry."""
        config = CacheConfig(cache_dir=temp_cache_dir)
        cache_manager = CacheManager(config)
        
        # Add cache entry
        cache_manager.set("info_key", "prompt", "response")
        
        # Get cache info
        info = cache_manager.get_cache_info("info_key")
        
        assert info is not None
        assert info["key"] == "info_key"
        assert info["response"] == "response"
        assert info["prompt"] == "prompt"
        assert info["exists"] == True
        assert info["expired"] == False
        assert "file_size" in info
        assert "last_modified" in info
    
    def test_get_cache_info_not_found(self):
        """Test getting cache info for non-existent entry."""
        cache_manager = CacheManager()
        
        info = cache_manager.get_cache_info("non_existent_key")
        
        assert info is None
    
    def test_load_cache_index(self, temp_cache_dir):
        """Test loading cache index from disk."""
        config = CacheConfig(cache_dir=temp_cache_dir)
        cache_manager = CacheManager(config)
        
        # Create cache index file
        index_file = Path(temp_cache_dir) / "cache_index.json"
        index_data = {
            "test_key": {
                "key": "test_key",
                "response": "test response",
                "prompt": "test prompt",
                "timestamp": 1234567890.0,
                "model": "gpt-5-mini",
                "temperature": 1.0,
                "max_tokens": 4000,
                "token_count": 0,
                "file_size": 0
            }
        }
        
        with open(index_file, 'w') as f:
            json.dump(index_data, f)
        
        # Create new cache manager to load index
        new_cache_manager = CacheManager(config)
        
        # Should have loaded the index
        assert "test_key" in new_cache_manager._cache_index
    
    def test_load_cache_index_error(self, temp_cache_dir):
        """Test loading cache index when error occurs."""
        config = CacheConfig(cache_dir=temp_cache_dir)
        
        # Create corrupted index file
        index_file = Path(temp_cache_dir) / "cache_index.json"
        with open(index_file, 'w') as f:
            f.write("invalid json content")
        
        # Should handle error gracefully
        cache_manager = CacheManager(config)
        
        # Should have empty index
        assert len(cache_manager._cache_index) == 0
    
    def test_save_cache_index(self, temp_cache_dir):
        """Test saving cache index to disk."""
        config = CacheConfig(cache_dir=temp_cache_dir)
        cache_manager = CacheManager(config)
        
        # Add cache entry
        cache_manager.set("save_key", "prompt", "response")
        
        # Check if index file was created
        index_file = Path(temp_cache_dir) / "cache_index.json"
        assert index_file.exists()
        
        # Check index content
        with open(index_file, 'r') as f:
            index_data = json.load(f)
        
        assert "save_key" in index_data
        assert index_data["save_key"]["key"] == "save_key"


class TestConvenienceFunctions:
    """Test convenience functions."""
    
    def test_create_cache_manager(self):
        """Test create_cache_manager function."""
        config = CacheConfig(cache_dir="./test_cache")
        cache_manager = create_cache_manager(config)
        
        assert isinstance(cache_manager, CacheManager)
        assert cache_manager.config.cache_dir == "./test_cache"
    
    def test_get_default_cache_manager(self):
        """Test get_default_cache_manager function."""
        cache_manager = get_default_cache_manager()
        
        assert isinstance(cache_manager, CacheManager)
        assert cache_manager.config.cache_dir == "./cache"
        assert cache_manager.config.max_size_mb == 100
    
    def test_generate_cache_key(self):
        """Test generate_cache_key function."""
        key = generate_cache_key("test prompt", "test system")
        
        assert isinstance(key, str)
        assert len(key) == 64  # SHA256 hash length


class TestEdgeCases:
    """Test edge cases and boundary conditions."""
    
    def test_cache_manager_empty_cache_dir(self):
        """Test cache manager with empty cache directory."""
        config = CacheConfig(cache_dir="")
        cache_manager = CacheManager(config)
        
        # Should handle empty directory gracefully
        assert cache_manager.config.enabled == False
    
    def test_cache_manager_very_large_cache(self):
        """Test cache manager with very large cache entries."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config = CacheConfig(cache_dir=temp_dir, max_size_mb=1)
            cache_manager = CacheManager(config)
            
            # Create very large response
            large_response = "x" * (2 * 1024 * 1024)  # 2MB
            
            # Should handle large entries gracefully
            result = cache_manager.set("large_key", "prompt", large_response)
            assert result == True
    
    def test_cache_entry_very_long_strings(self):
        """Test cache entry with very long strings."""
        long_string = "x" * 10000
        
        entry = CacheEntry(
            key=long_string,
            response=long_string,
            prompt=long_string
        )
        
        # Should handle long strings gracefully
        entry_dict = entry.to_dict()
        assert entry_dict["key"] == long_string
        assert entry_dict["response"] == long_string
        assert entry_dict["prompt"] == long_string
    
    def test_cache_manager_concurrent_access(self):
        """Test cache manager with concurrent access simulation."""
        cache_manager = CacheManager()
        
        # Simulate concurrent access
        cache_manager.set("concurrent_key", "prompt", "response")
        
        # Multiple simultaneous gets should work
        response1 = cache_manager.get("concurrent_key")
        response2 = cache_manager.get("concurrent_key")
        
        assert response1 == response2
        assert response1 == "response"


class TestErrorHandling:
    """Test error handling scenarios."""
    
    def test_cache_manager_permission_error(self):
        """Test cache manager with permission errors."""
        # Mock path operations to raise permission error
        with patch('pathlib.Path.mkdir') as mock_mkdir:
            mock_mkdir.side_effect = PermissionError("Permission denied")
            
            config = CacheConfig(cache_dir="/root/forbidden")
            cache_manager = CacheManager(config)
            
            # Should disable caching on permission error
            assert cache_manager.config.enabled == False
    
    def test_cache_manager_disk_full(self):
        """Test cache manager when disk is full."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config = CacheConfig(cache_dir=temp_dir)
            cache_manager = CacheManager(config)
            
            # Mock file write to simulate disk full
            with patch('builtins.open') as mock_open:
                mock_open.side_effect = OSError("No space left on device")
                
                result = cache_manager.set("disk_full_key", "prompt", "response")
                
                assert result == False
                assert cache_manager.cache_errors == 1
    
    def test_cache_manager_corrupted_files(self):
        """Test cache manager with corrupted cache files."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config = CacheConfig(cache_dir=temp_dir)
            cache_manager = CacheManager(config)
            
            # Create corrupted cache file
            corrupted_file = Path(temp_dir) / "corrupted.json"
            with open(corrupted_file, 'w') as f:
                f.write("invalid json content")
            
            # Should handle corrupted files gracefully
            cache_manager.cleanup(force=True)
            
            # Corrupted file should be handled without crashing
            assert True  # If we get here, no exception was raised
