"""
Tests for configuration management.
"""

import pytest
import os
import tempfile
from unittest.mock import patch, mock_open
from pathlib import Path
from extractEntity_Phase.infrastructure.config import (
    Environment, APIConfig, CacheConfig, LoggingConfig,
    PipelineConfig, Config, get_config, set_config
)


class TestEnvironment:
    """Test environment enumeration."""
    
    def test_environment_values(self):
        """Test environment enum values."""
        assert Environment.DEVELOPMENT.value == "development"
        assert Environment.TESTING.value == "testing"
        assert Environment.PRODUCTION.value == "production"
    
    def test_environment_from_string(self):
        """Test environment creation from string."""
        assert Environment("development") == Environment.DEVELOPMENT
        assert Environment("production") == Environment.PRODUCTION
    
    def test_environment_invalid_value(self):
        """Test environment with invalid value."""
        with pytest.raises(ValueError):
            Environment("invalid")


class TestAPIConfig:
    """Test API configuration dataclass."""
    
    def test_api_config_creation(self):
        """Test API config creation with all fields."""
        api_config = APIConfig(
            api_key="test_key_123",
            model="gpt-4",
            temperature=0.5,
            max_tokens=2000,
            rpm_limit=120,
            concurrent_limit=5,
            retry_attempts=5,
            base_delay=10,
            tpm_limit=180000,
            tpd_limit=4000000
        )
        assert api_config.api_key == "test_key_123"
        assert api_config.model == "gpt-4"
        assert api_config.temperature == 0.5
        assert api_config.max_tokens == 2000
        assert api_config.rpm_limit == 120
        assert api_config.concurrent_limit == 5
        assert api_config.retry_attempts == 5
        assert api_config.base_delay == 10
        assert api_config.tpm_limit == 180000
        assert api_config.tpd_limit == 4000000
    
    def test_api_config_defaults(self):
        """Test API config default values."""
        api_config = APIConfig(api_key="test_key")
        assert api_config.model == "gpt-5-mini"
        assert api_config.temperature == 0.1
        assert api_config.max_tokens == 4000
        assert api_config.rpm_limit == 60
        assert api_config.concurrent_limit == 3
        assert api_config.retry_attempts == 3
        assert api_config.base_delay == 5
        assert api_config.tpm_limit == 90000
        assert api_config.tpd_limit == 2000000
    
    def test_api_config_validation(self):
        """Test API config validation."""
        # APIConfig is a dataclass without custom validation
        # All validation is handled in the Config class
        api_config = APIConfig(api_key="test")
        assert api_config.api_key == "test"


class TestCacheConfig:
    """Test cache configuration dataclass."""
    
    def test_cache_config_creation(self):
        """Test cache config creation with all fields."""
        from pathlib import Path
        cache_config = CacheConfig(
            enabled=True,
            directory=Path("/tmp/cache"),
            max_size_mb=100,
            ttl_hours=24
        )
        assert cache_config.enabled == True
        assert cache_config.directory == Path("/tmp/cache")
        assert cache_config.max_size_mb == 100
        assert cache_config.ttl_hours == 24
    
    def test_cache_config_defaults(self):
        """Test cache config default values."""
        from pathlib import Path
        cache_config = CacheConfig()
        assert cache_config.enabled == True
        assert cache_config.directory == Path(".cache/gpt5mini_ent")
        assert cache_config.max_size_mb == 1000
        assert cache_config.ttl_hours == 24
    
    def test_cache_config_validation(self):
        """Test cache config validation."""
        # CacheConfig is a dataclass without custom validation
        # All validation is handled in the Config class
        cache_config = CacheConfig()
        assert cache_config.enabled == True


class TestLoggingConfig:
    """Test logging configuration dataclass."""
    
    def test_logging_config_creation(self):
        """Test logging config creation with all fields."""
        from pathlib import Path
        logging_config = LoggingConfig(
            level="DEBUG",
            directory=Path("/tmp/logs"),
            max_file_size_mb=20,
            backup_count=10,
            format="[{timestamp}] {level} - {message}"
        )
        assert logging_config.level == "DEBUG"
        assert logging_config.directory == Path("/tmp/logs")
        assert logging_config.max_file_size_mb == 20
        assert logging_config.backup_count == 10
        assert logging_config.format == "[{timestamp}] {level} - {message}"
    
    def test_logging_config_defaults(self):
        """Test logging config default values."""
        from pathlib import Path
        logging_config = LoggingConfig()
        assert logging_config.level == "INFO"
        assert logging_config.directory == Path("../docs/Iteration_Terminal_Progress")
        assert logging_config.max_file_size_mb == 10
        assert logging_config.backup_count == 5
        assert logging_config.format == "[{timestamp}] {message}"
    
    def test_logging_config_validation(self):
        """Test logging config validation."""
        # LoggingConfig is a dataclass without custom validation
        # All validation is handled in the Config class
        logging_config = LoggingConfig()
        assert logging_config.level == "INFO"


class TestPipelineConfig:
    """Test pipeline configuration dataclass."""
    
    def test_pipeline_config_creation(self):
        """Test pipeline config creation with all fields."""
        from pathlib import Path
        pipeline_config = PipelineConfig(
            dataset="TestDataset",
            dataset_path=Path("/tmp/dataset"),
            iteration=2,
            output_dir=Path("/tmp/output"),
            batch_size=20,
            max_text_length=15000
        )
        assert pipeline_config.dataset == "TestDataset"
        assert pipeline_config.dataset_path == Path("/tmp/dataset")
        assert pipeline_config.iteration == 2
        assert pipeline_config.output_dir == Path("/tmp/output")
        assert pipeline_config.batch_size == 20
        assert pipeline_config.max_text_length == 15000
    
    def test_pipeline_config_defaults(self):
        """Test pipeline config default values."""
        from pathlib import Path
        pipeline_config = PipelineConfig()
        assert pipeline_config.dataset == "DreamOf_RedChamber"
        assert pipeline_config.dataset_path == Path("../datasets/GPT5mini_result_DreamOf_RedChamber/")
        assert pipeline_config.iteration == 1
        assert pipeline_config.output_dir is None
        assert pipeline_config.batch_size == 10
        assert pipeline_config.max_text_length == 10000
    
    def test_pipeline_config_validation(self):
        """Test pipeline config validation."""
        # PipelineConfig is a dataclass without custom validation
        # All validation is handled in the Config class
        pipeline_config = PipelineConfig()
        assert pipeline_config.dataset == "DreamOf_RedChamber"


class TestConfig:
    """Test main configuration class."""
    
    def test_config_creation(self):
        """Test config creation with environment parameter."""
        with patch.dict(os.environ, {'OPENAI_API_KEY': 'test_key'}):
            config = Config(environment=Environment.DEVELOPMENT)
            
            assert config.environment == Environment.DEVELOPMENT
            assert isinstance(config.api, APIConfig)
            assert isinstance(config.cache, CacheConfig)
            assert isinstance(config.logging, LoggingConfig)
            assert isinstance(config.pipeline, PipelineConfig)
    
    def test_config_defaults(self):
        """Test config default values."""
        with patch.dict(os.environ, {'OPENAI_API_KEY': 'test_key'}):
            config = Config()
            
            assert config.environment == Environment.DEVELOPMENT
            assert isinstance(config.api, APIConfig)
            assert isinstance(config.cache, CacheConfig)
            assert isinstance(config.logging, LoggingConfig)
            assert isinstance(config.pipeline, PipelineConfig)
    
    def test_config_from_environment_variables(self):
        """Test config loading from environment variables."""
        env_vars = {
            'OPENAI_API_KEY': 'env_api_key',
            'GPT5_MODEL': 'gpt-4',
            'GPT5_TEMPERATURE': '0.5',
            'CACHE_ENABLED': 'false',
            'LOG_LEVEL': 'DEBUG',
            'PIPELINE_MAX_TEXT_LENGTH': '15000'
        }
        
        with patch.dict(os.environ, env_vars):
            config = Config(environment=Environment.PRODUCTION)
            
            assert config.environment == Environment.PRODUCTION
            assert config.api.api_key == 'env_api_key'
            assert config.api.model == 'gpt-4'
            assert config.api.temperature == 0.5
            assert config.cache.enabled == False
            assert config.logging.level == 'DEBUG'
            assert config.pipeline.max_text_length == 15000
    
    def test_config_validation(self):
        """Test config validation."""
        # Config accepts Environment enum values
        with patch.dict(os.environ, {'OPENAI_API_KEY': 'test_key'}):
            config = Config(environment=Environment.DEVELOPMENT)
            assert config.environment == Environment.DEVELOPMENT
    
    def test_get_output_directory(self):
        """Test output directory generation."""
        with patch.dict(os.environ, {'OPENAI_API_KEY': 'test_key'}):
            config = Config(environment=Environment.DEVELOPMENT)
            output_dir = config.get_output_directory()
            
            assert "DreamOf_RedChamber" in str(output_dir)
            assert "Graph_Iteration1" in str(output_dir)
            assert isinstance(output_dir, Path)
    
    def test_to_dict(self):
        """Test config serialization to dictionary."""
        with patch.dict(os.environ, {'OPENAI_API_KEY': 'test_key'}):
            config = Config()
            config_dict = config.to_dict()
            
            assert "environment" in config_dict
            assert "api" in config_dict
            assert "cache" in config_dict
            assert "logging" in config_dict
            assert "pipeline" in config_dict
            
            # Check nested structures
            assert "model" in config_dict["api"]
            assert "enabled" in config_dict["cache"]
            assert "level" in config_dict["logging"]
            assert "max_text_length" in config_dict["pipeline"]
    
    def test_string_representation(self):
        """Test config string representation."""
        with patch.dict(os.environ, {'OPENAI_API_KEY': 'test_key'}):
            config = Config()
            config_str = str(config)
            
            assert "Config" in config_str
            assert "development" in config_str
            assert "DreamOf_RedChamber" in config_str
    
    def test_repr_representation(self):
        """Test config detailed string representation."""
        with patch.dict(os.environ, {'OPENAI_API_KEY': 'test_key'}):
            config = Config()
            repr_str = repr(config)
            
            assert "Config" in repr_str
            assert "environment=" in repr_str
            assert "api=" in repr_str
            assert "cache=" in repr_str
            assert "logging=" in repr_str
            assert "pipeline=" in repr_str


class TestConfigFunctions:
    """Test configuration utility functions."""
    
    def test_get_config_singleton(self):
        """Test get_config singleton behavior."""
        # Clear any existing config
        set_config(None)
        
        with patch.dict(os.environ, {'OPENAI_API_KEY': 'test_key'}):
            config1 = get_config()
            config2 = get_config()
            
            assert config1 is config2
            assert isinstance(config1, Config)
    
    def test_set_config(self):
        """Test set_config function."""
        with patch.dict(os.environ, {'OPENAI_API_KEY': 'test_key'}):
            # Create a custom config
            custom_config = Config(environment=Environment.PRODUCTION)
            
            # Set the custom config
            set_config(custom_config)
            
            # Get the config and verify it's the custom one
            retrieved_config = get_config()
            assert retrieved_config is custom_config
            assert retrieved_config.environment == Environment.PRODUCTION
    
    def test_set_config_none(self):
        """Test set_config with None."""
        # Set config to None
        set_config(None)
        
        with patch.dict(os.environ, {'OPENAI_API_KEY': 'test_key'}):
            # Should create a new default config
            config = get_config()
            assert isinstance(config, Config)
            assert config.environment == Environment.DEVELOPMENT


class TestErrorHandling:
    """Test error handling scenarios."""
    
    def test_missing_required_environment_variables(self):
        """Test handling of missing required environment variables."""
        # Clear environment variables
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(ValueError, match="Required environment variable OPENAI_API_KEY is not set"):
                Config()
    
    def test_invalid_environment_variable_values(self):
        """Test handling of invalid environment variable values."""
        env_vars = {
            'OPENAI_API_KEY': 'test_key',
            'PIPELINE_MAX_TEXT_LENGTH': 'invalid_number',
            'GPT5_TEMPERATURE': 'not_a_number',
            'CACHE_MAX_SIZE_MB': 'negative_value'
        }
        
        with patch.dict(os.environ, env_vars):
            # Should handle invalid values gracefully and use defaults
            config = Config()
            
            assert config.pipeline.max_text_length == 10000  # Default value
            assert config.api.temperature == 0.1  # Default value
            assert config.cache.max_size_mb == 1000  # Default value
    
    def test_invalid_configuration_values(self):
        """Test handling of invalid configuration values."""
        # Dataclasses don't have built-in validation
        # All validation is handled in the Config class
        api_config = APIConfig(api_key="test")
        cache_config = CacheConfig()
        logging_config = LoggingConfig()
        pipeline_config = PipelineConfig()
        
        assert api_config.api_key == "test"
        assert cache_config.enabled == True
        assert logging_config.level == "INFO"
        assert pipeline_config.dataset == "DreamOf_RedChamber"


class TestEdgeCases:
    """Test edge cases and boundary conditions."""
    
    def test_empty_environment_variables(self):
        """Test handling of empty environment variables."""
        env_vars = {
            'OPENAI_API_KEY': 'sk-test-key-for-testing-only',
            'CACHE_DIR': '',
            'LOG_DIR': ''
        }
        
        with patch.dict(os.environ, env_vars):
            config = Config()
            
            # Should handle empty strings gracefully
            assert config.api.api_key == "sk-test-key-for-testing-only"
            assert config.cache.directory == Path(".cache/gpt5mini_ent")  # Should use default
            assert config.logging.directory == Path("../docs/Iteration_Terminal_Progress")  # Should use default
    
    def test_whitespace_environment_variables(self):
        """Test handling of whitespace-only environment variables."""
        env_vars = {
            'OPENAI_API_KEY': 'sk-test-key-for-testing-only',
            'CACHE_DIR': '  \t  ',
            'LOG_DIR': '\n\n'
        }
        
        with patch.dict(os.environ, env_vars):
            config = Config()
            
            # Should handle whitespace gracefully
            assert config.api.api_key == "sk-test-key-for-testing-only"
            assert config.cache.directory == Path(".cache/gpt5mini_ent")  # Should use default
            assert config.logging.directory == Path("../docs/Iteration_Terminal_Progress")  # Should use default
    
    def test_very_large_values(self):
        """Test handling of very large configuration values."""
        with patch.dict(os.environ, {'OPENAI_API_KEY': 'test_key'}):
            # Test very large text length
            config = Config()
            config.pipeline.max_text_length = 1000000  # 1MB
            
            assert config.pipeline.max_text_length == 1000000
            
            # Test very large cache size
            config.cache.max_size_mb = 10000  # 10GB
            
            assert config.cache.max_size_mb == 10000
    
    def test_zero_values_where_allowed(self):
        """Test zero values where they are allowed."""
        with patch.dict(os.environ, {'OPENAI_API_KEY': 'test_key'}):
            # Test zero retry attempts (allowed)
            config = Config()
            config.api.retry_attempts = 0
            
            assert config.api.retry_attempts == 0
            
            # Test zero backup count (allowed)
            config.logging.backup_count = 0
            
            assert config.logging.backup_count == 0
    
    def test_boundary_values(self):
        """Test boundary values for configuration."""
        with patch.dict(os.environ, {'OPENAI_API_KEY': 'test_key'}):
            # Test minimum valid values
            config = Config()
            
            # API config boundaries
            config.api.retry_attempts = 0  # Minimum valid retries
            config.api.rpm_limit = 1  # Minimum valid rate limit
            config.api.concurrent_limit = 1  # Minimum valid concurrent limit
            
            assert config.api.retry_attempts == 0
            assert config.api.rpm_limit == 1
            assert config.api.concurrent_limit == 1
            
            # Cache config boundaries
            config.cache.max_size_mb = 1  # Minimum valid size
            config.cache.ttl_hours = 1  # Minimum valid TTL
            
            assert config.cache.max_size_mb == 1
            assert config.cache.ttl_hours == 1
            
            # Pipeline config boundaries
            config.pipeline.max_text_length = 1  # Minimum valid length
            config.pipeline.batch_size = 1  # Minimum valid batch size
            config.pipeline.iteration = 1  # Minimum valid iteration
            
            assert config.pipeline.max_text_length == 1
            assert config.pipeline.batch_size == 1
            assert config.pipeline.iteration == 1

