"""
Pytest configuration and common fixtures for ECTD pipeline tests.
"""

import pytest
import tempfile
import os
from pathlib import Path
from unittest.mock import Mock, patch


@pytest.fixture(autouse=True)
def setup_test_environment():
    """Set up test environment variables before each test."""
    # Set required environment variables for testing
    os.environ["OPENAI_API_KEY"] = "sk-test-key-for-testing-only"
    os.environ["GPT5_MODEL"] = "gpt-5-mini"
    os.environ["GPT5_TEMPERATURE"] = "0.1"
    os.environ["GPT5_MAX_TOKENS"] = "4000"
    os.environ["CACHE_ENABLED"] = "true"
    os.environ["LOG_LEVEL"] = "INFO"
    
    yield
    
    # Clean up environment variables after test
    test_env_vars = [
        "OPENAI_API_KEY", "GPT5_MODEL", "GPT5_TEMPERATURE", 
        "GPT5_MAX_TOKENS", "CACHE_ENABLED", "LOG_LEVEL"
    ]
    for var in test_env_vars:
        if var in os.environ:
            del os.environ[var]


@pytest.fixture
def temp_dir():
    """Create temporary directory for tests."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        yield Path(tmp_dir)


@pytest.fixture
def sample_chinese_text():
    """Sample Chinese text for testing."""
    return "紅樓夢是一部中國古典小說，作者是曹雪芹。故事發生在大觀園中，主要講述賈寶玉和林黛玉的愛情故事。"


@pytest.fixture
def sample_entities():
    """Sample entities for testing."""
    return [
        {"text": "紅樓夢", "type": "book", "confidence": 0.95},
        {"text": "曹雪芹", "type": "person", "confidence": 0.98},
        {"text": "大觀園", "type": "location", "confidence": 0.92},
        {"text": "賈寶玉", "type": "person", "confidence": 0.97},
        {"text": "林黛玉", "type": "person", "confidence": 0.96}
    ]


@pytest.fixture
def mock_config():
    """Mock configuration for testing."""
    config = Mock()
    config.api.api_key = "test_api_key"
    config.api.model = "gpt-5-mini"
    config.api.temperature = 0.1
    config.api.max_tokens = 4000
    config.cache.enabled = True
    config.cache.directory = Path(".cache/test")
    config.logging.level = "INFO"
    config.logging.directory = Path("logs/test")
    config.pipeline.dataset = "test_dataset"
    config.pipeline.iteration = 1
    return config


@pytest.fixture
def mock_logger():
    """Mock logger for testing."""
    logger = Mock()
    logger.info = Mock()
    logger.warning = Mock()
    logger.error = Mock()
    logger.debug = Mock()
    logger.success = Mock()
    return logger


@pytest.fixture
def mock_litellm():
    """Mock LiteLLM availability."""
    with patch('extractEntity_Phase.api.gpt5mini_client.LITELLM_AVAILABLE', True):
        yield