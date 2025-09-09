"""
Configuration Management Module

This module provides centralized configuration management for the ECTD pipeline,
including environment variable support, validation, and default values.

The module follows the dependency injection pattern and provides a clean interface
for accessing configuration values throughout the application.
"""

import os
import logging
import time
from pathlib import Path
from typing import Optional, Dict, Any, List
from dataclasses import dataclass, field
from enum import Enum


class Environment(Enum):
    """Environment enumeration for different deployment contexts."""
    DEVELOPMENT = "development"
    TESTING = "testing"
    PRODUCTION = "production"


@dataclass
class APIConfig:
    """API configuration settings."""
    api_key: str
    model: str = "gpt-5-mini"
    temperature: float = 0.1
    max_tokens: int = 4000
    rpm_limit: int = 60
    concurrent_limit: int = 3
    retry_attempts: int = 3
    base_delay: int = 5
    tpm_limit: int = 90000
    tpd_limit: int = 2000000


@dataclass
class CacheConfig:
    """Cache configuration settings."""
    enabled: bool = True
    directory: Path = field(default_factory=lambda: Path(".cache/gpt5mini_ent"))
    max_size_mb: int = 1000
    ttl_hours: int = 24


@dataclass
class LoggingConfig:
    """Logging configuration settings."""
    level: str = "INFO"
    directory: Path = field(default_factory=lambda: Path("../docs/Iteration_Terminal_Progress"))
    max_file_size_mb: int = 10
    backup_count: int = 5
    format: str = "[{timestamp}] {message}"
    log_file: str = "default.log"  # Add this field


@dataclass
class PipelineConfig:
    """Pipeline configuration settings."""
    dataset: str = "DreamOf_RedChamber"
    dataset_path: Path = field(default_factory=lambda: Path("../datasets/GPT5mini_result_DreamOf_RedChamber/"))
    iteration: int = 1
    output_dir: Optional[Path] = None
    batch_size: int = 10
    max_text_length: int = 10000


class Config:
    """
    Centralized configuration management for the ECTD pipeline.
    
    This class provides a single point of access to all configuration values,
    with support for environment variable overrides and validation.
    
    Attributes:
        environment: Current deployment environment
        api: API configuration settings
        cache: Cache configuration settings
        logging: Logging configuration settings
        pipeline: Pipeline configuration settings
    """
    
    def __init__(self, environment: Environment = Environment.DEVELOPMENT):
        """
        Initialize configuration with environment-specific defaults.
        
        Args:
            environment: Deployment environment for configuration
        """
        self.environment = environment
        self._logger = logging.getLogger(__name__)
        
        # Load configuration from environment variables
        self.api = self._load_api_config()
        self.cache = self._load_cache_config()
        self.logging = self._load_logging_config()
        self.pipeline = self._load_pipeline_config()
        
        # Validate configuration
        self._validate_config()
        
        self._logger.info(f"Configuration loaded for environment: {environment.value}")
    
    def _load_api_config(self) -> APIConfig:
        """Load API configuration from environment variables."""
        api_key = self._get_env_var("OPENAI_API_KEY", required=True)
        
        return APIConfig(
            api_key=api_key,
            model=os.getenv("GPT5_MODEL", "gpt-5-mini"),
            temperature=self._safe_float(os.getenv("GPT5_TEMPERATURE", "0.1"), 0.1),
            max_tokens=int(os.getenv("GPT5_MAX_TOKENS", "4000")),
            rpm_limit=int(os.getenv("OPENAI_RPM_LIMIT", "60")),
            concurrent_limit=int(os.getenv("OPENAI_CONCURRENT_LIMIT", "3")),
            retry_attempts=int(os.getenv("OPENAI_RETRY_ATTEMPTS", "3")),
            base_delay=int(os.getenv("OPENAI_BASE_DELAY", "5")),
            tpm_limit=int(os.getenv("OPENAI_TPM_LIMIT", "90000")),
            tpd_limit=int(os.getenv("OPENAI_TPD_LIMIT", "2000000"))
        )
    
    def _load_cache_config(self) -> CacheConfig:
        """Load cache configuration from environment variables."""
        cache_dir = os.getenv("CACHE_DIR", ".cache/gpt5mini_ent")
        if not cache_dir or not cache_dir.strip():
            cache_dir = ".cache/gpt5mini_ent"
        
        return CacheConfig(
            enabled=os.getenv("CACHE_ENABLED", "true").lower() == "true",
            directory=Path(cache_dir),
            max_size_mb=self._safe_int(os.getenv("CACHE_MAX_SIZE_MB", "1000"), 1000),
            ttl_hours=int(os.getenv("CACHE_TTL_HOURS", "24"))
        )
    
    def _load_logging_config(self) -> LoggingConfig:
        """Load logging configuration from environment variables."""
        log_dir = os.getenv("LOG_DIR", "../docs/Iteration_Terminal_Progress")
        if not log_dir or not log_dir.strip():
            log_dir = "../docs/Iteration_Terminal_Progress"
        
        return LoggingConfig(
            level=os.getenv("LOG_LEVEL", "INFO"),
            directory=Path(log_dir),
            max_file_size_mb=int(os.getenv("LOG_MAX_FILE_SIZE_MB", "10")),
            backup_count=int(os.getenv("LOG_BACKUP_COUNT", "5")),
            format=os.getenv("LOG_FORMAT", "[{timestamp}] {message}")
        )
    
    def _load_pipeline_config(self) -> PipelineConfig:
        """Load pipeline configuration from environment variables."""
        dataset = os.getenv("PIPELINE_DATASET", "DreamOf_RedChamber")
        dataset_path = os.getenv("PIPELINE_DATASET_PATH", f"../datasets/GPT5mini_result_{dataset}/")
        iteration = int(os.getenv("PIPELINE_ITERATION", "1"))
        output_dir = os.getenv("PIPELINE_OUTPUT_DIR")
        
        return PipelineConfig(
            dataset=dataset,
            dataset_path=Path(dataset_path),
            iteration=iteration,
            output_dir=Path(output_dir) if output_dir else None,
            batch_size=int(os.getenv("PIPELINE_BATCH_SIZE", "10")),
            max_text_length=self._safe_int(os.getenv("PIPELINE_MAX_TEXT_LENGTH", "10000"), 10000)
        )
    
    def _get_env_var(self, name: str, required: bool = False, default: Optional[str] = None) -> str:
        """
        Get environment variable with validation.
        
        Args:
            name: Environment variable name
            required: Whether the variable is required
            default: Default value if not required
            
        Returns:
            Environment variable value
            
        Raises:
            ValueError: If required variable is missing
        """
        value = os.getenv(name, default)
        
        if required and not value:
            raise ValueError(f"Required environment variable {name} is not set")
        
        return value or ""
    
    def _safe_int(self, value: str, default: int = 0) -> int:
        """Safely convert string to int with default fallback."""
        if not value or not value.strip():  # Handle empty or whitespace-only strings
            return default
        try:
            return int(value)
        except (ValueError, TypeError):
            return default

    def _safe_float(self, value: str, default: float = 0.0) -> float:
        """Safely convert string to float with default fallback."""
        try:
            return float(value)
        except (ValueError, TypeError):
            return default
    
    def _validate_config(self) -> None:
        """Validate configuration values and log warnings for invalid settings."""
        # Validate API configuration
        if self.api.temperature < 0.0 or self.api.temperature > 2.0:
            self._logger.warning(f"Invalid temperature value: {self.api.temperature}. Using default: 0.1")
            self.api.temperature = 0.1
        
        if self.api.max_tokens < 1 or self.api.max_tokens > 8000:
            self._logger.warning(f"Invalid max_tokens value: {self.api.max_tokens}. Using default: 4000")
            self.api.max_tokens = 4000
        
        # Validate pipeline configuration
        if self.pipeline.iteration < 1:
            self._logger.warning(f"Invalid iteration value: {self.pipeline.iteration}. Using default: 1")
            self.pipeline.iteration = 1
        
        # Validate cache configuration
        if self.cache.max_size_mb < 1:
            self._logger.warning(f"Invalid cache max size: {self.cache.max_size_mb}MB. Using default: 1000MB")
            self.cache.max_size_mb = 1000
    
    def get_output_directory(self) -> Path:
        """
        Get the output directory for the current iteration.
        
        Returns:
            Path to the output directory
        """
        if self.pipeline.output_dir:
            return self.pipeline.output_dir
        
        return self.pipeline.dataset_path / f"Graph_Iteration{self.pipeline.iteration}"
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert configuration to dictionary for serialization.
        
        Returns:
            Dictionary representation of configuration
        """
        return {
            "environment": self.environment.value,
            "api": {
                "model": self.api.model,
                "temperature": self.api.temperature,
                "max_tokens": self.api.max_tokens,
                "rpm_limit": self.api.rpm_limit,
                "concurrent_limit": self.api.concurrent_limit,
                "retry_attempts": self.api.retry_attempts,
                "base_delay": self.api.base_delay,
                "tpm_limit": self.api.tpm_limit,
                "tpd_limit": self.api.tpd_limit
            },
            "cache": {
                "enabled": self.cache.enabled,
                "directory": str(self.cache.directory),
                "max_size_mb": self.cache.max_size_mb,
                "ttl_hours": self.cache.ttl_hours
            },
            "logging": {
                "level": self.logging.level,
                "directory": str(self.logging.directory),
                "max_file_size_mb": self.logging.max_file_size_mb,
                "backup_count": self.logging.backup_count,
                "format": self.logging.format
            },
            "pipeline": {
                "dataset": self.pipeline.dataset,
                "dataset_path": str(self.pipeline.dataset_path),
                "iteration": self.pipeline.iteration,
                "output_dir": str(self.get_output_directory()),
                "batch_size": self.pipeline.batch_size,
                "max_text_length": self.pipeline.max_text_length
            }
        }
    
    def __str__(self) -> str:
        """String representation of configuration."""
        return f"Config(environment={self.environment.value}, dataset={self.pipeline.dataset}, iteration={self.pipeline.iteration})"
    
    def __repr__(self) -> str:
        """Detailed string representation of configuration."""
        return f"Config(environment={self.environment.value}, api={self.api}, cache={self.cache}, logging={self.logging}, pipeline={self.pipeline})"


# Global configuration instance
_config: Optional[Config] = None


def get_config() -> Config:
    """
    Get or create the global configuration instance.
    
    Returns:
        Global configuration instance
    """
    global _config
    if _config is None:
        _config = Config()
    return _config


def set_config(config: Config) -> None:
    """
    Set the global configuration instance.
    
    Args:
        config: Configuration instance to set globally
    """
    global _config
    _config = config


# Backward compatibility with openai_config.py
_config_instance = None
_token_usage_minute: List[int] = []
_token_usage_day: List[int] = []
_last_reset_minute = time.time()
_last_reset_day = time.time()


def get_api_key() -> str:
    """Get API key from environment."""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable is not set")
    return api_key


def track_token_usage(tokens: int) -> bool:
    """Track token usage for TPM/TPD limits."""
    global _token_usage_minute, _token_usage_day, _last_reset_minute, _last_reset_day
    current_time = time.time()
    
    # Reset minute tracking if needed
    if current_time - _last_reset_minute >= 60:
        _token_usage_minute.clear()
        _last_reset_minute = current_time
    
    # Reset day tracking if needed  
    if current_time - _last_reset_day >= 86400:
        _token_usage_day.clear()
        _last_reset_day = current_time
    
    # Add token usage
    _token_usage_minute.append(tokens)
    _token_usage_day.append(tokens)
    return True


def get_token_usage_stats() -> Dict[str, int]:
    """Get current token usage statistics."""
    config = get_config()
    
    minute_tokens = sum(_token_usage_minute)
    day_tokens = sum(_token_usage_day)
    
    return {
        'minute_tokens': minute_tokens,
        'day_tokens': day_tokens, 
        'minute_remaining': max(0, config.api.tpm_limit - minute_tokens),
        'day_remaining': max(0, config.api.tpd_limit - day_tokens),
        'minute_percentage': (minute_tokens / config.api.tpm_limit) * 100,
        'day_percentage': (day_tokens / config.api.tpd_limit) * 100
    }


def calculate_rate_limit_delay() -> int:
    """Calculate delay for rate limiting."""
    config = get_config()
    return max(config.api.base_delay, 60 // config.api.rpm_limit)


def get_api_config_summary() -> Dict[str, Any]:
    """Get API configuration summary."""
    config = get_config()
    return {
        'rpm_limit': config.api.rpm_limit,
        'concurrent_limit': config.api.concurrent_limit,
        'retry_attempts': config.api.retry_attempts,
        'base_delay': config.api.base_delay,
        'calculated_delay': calculate_rate_limit_delay(),
        'temperature': config.api.temperature,
        'max_tokens': config.api.max_tokens,
        'model': config.api.model,
        'tpm_limit': config.api.tpm_limit,
        'tpd_limit': config.api.tpd_limit
    }


# Constants for backward compatibility
OPENAI_RPM_LIMIT = 60
OPENAI_CONCURRENT_LIMIT = 3  
OPENAI_RETRY_ATTEMPTS = 3
OPENAI_BASE_DELAY = 5
OPENAI_TEMPERATURE = 0.1
OPENAI_MAX_TOKENS = 4000
OPENAI_TPM_LIMIT = 90000
OPENAI_TPD_LIMIT = 2000000
GPT5_MINI_MODEL = "gpt-5-mini"
