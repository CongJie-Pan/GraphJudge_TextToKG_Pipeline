"""
Unit tests for configuration management in the GraphJudge Streamlit Pipeline.

This module tests the simplified configuration system in core/config.py,
following the TDD principles outlined in docs/Testing_Demands.md.

Test coverage includes:
- Environment variable loading
- API configuration validation
- Model parameter management
- Error handling for missing credentials
- Cross-platform path handling
"""

import pytest
import os
import tempfile
from pathlib import Path
from unittest.mock import patch, mock_open

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from core.config import (
    load_env_file, get_api_config, get_api_key, get_model_config,
    _load_env_from_path, GPT5_MINI_MODEL, PERPLEXITY_MODEL,
    OPENAI_TEMPERATURE, OPENAI_MAX_TOKENS, DEFAULT_TIMEOUT, MAX_RETRIES
)


class TestConstants:
    """Test configuration constants."""
    
    def test_model_constants(self):
        """Test model name constants."""
        assert GPT5_MINI_MODEL == "gpt-5-mini"
        assert PERPLEXITY_MODEL == "perplexity/sonar-reasoning"
    
    def test_parameter_constants(self):
        """Test parameter constants."""
        assert OPENAI_TEMPERATURE == 0.0  # Updated for deterministic structured output
        assert OPENAI_MAX_TOKENS == 12000  # Increased for reasoning models with complex output
        assert DEFAULT_TIMEOUT == 180  # Increased for GPT-5-mini reasoning timeout issues
        assert MAX_RETRIES == 3


class TestEnvFileLoading:
    """Test .env file loading functionality."""
    
    def test_load_env_from_path_valid_file(self):
        """Test loading environment variables from valid .env file."""
        env_content = """
# Test configuration
OPENAI_API_KEY=test-key-123
AZURE_OPENAI_ENDPOINT=https://test.openai.azure.com/
# Comment line
EMPTY_LINE_ABOVE=yes

QUOTED_VALUE="quoted-string"
SINGLE_QUOTED='single-quoted'
"""
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.env', delete=False) as f:
            f.write(env_content)
            temp_path = Path(f.name)
        
        try:
            # Clear environment first
            env_vars_to_clear = ['OPENAI_API_KEY', 'AZURE_OPENAI_ENDPOINT', 'QUOTED_VALUE', 'SINGLE_QUOTED']
            for var in env_vars_to_clear:
                if var in os.environ:
                    del os.environ[var]
            
            _load_env_from_path(temp_path)
            
            assert os.environ['OPENAI_API_KEY'] == 'test-key-123'
            assert os.environ['AZURE_OPENAI_ENDPOINT'] == 'https://test.openai.azure.com/'
            assert os.environ['QUOTED_VALUE'] == 'quoted-string'
            assert os.environ['SINGLE_QUOTED'] == 'single-quoted'
            
        finally:
            # Cleanup
            temp_path.unlink()
            for var in env_vars_to_clear:
                if var in os.environ:
                    del os.environ[var]
    
    def test_load_env_from_path_nonexistent_file(self):
        """Test loading from non-existent file (should not raise error)."""
        fake_path = Path("/nonexistent/path/.env")
        
        # Should not raise exception
        _load_env_from_path(fake_path)
    
    def test_load_env_from_path_malformed_lines(self):
        """Test handling of malformed lines in .env file."""
        env_content = """
VALID_KEY=valid_value
INVALID_LINE_NO_EQUALS
=VALUE_WITHOUT_KEY
KEY_WITHOUT_VALUE=
SPACES_IN_KEY = value_with_spaces
"""
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.env', delete=False) as f:
            f.write(env_content)
            temp_path = Path(f.name)
        
        try:
            # Clear environment first
            if 'VALID_KEY' in os.environ:
                del os.environ['VALID_KEY']
            
            _load_env_from_path(temp_path)
            
            # Only valid line should be loaded
            assert os.environ.get('VALID_KEY') == 'valid_value'
            
        finally:
            temp_path.unlink()
            if 'VALID_KEY' in os.environ:
                del os.environ['VALID_KEY']
    
    @patch('core.config.Path')
    def test_load_env_file_searches_multiple_paths(self, mock_path):
        """Test that load_env_file searches multiple possible paths."""
        # Mock path structure
        mock_current_file = mock_path(__file__).parent
        mock_env_paths = [
            mock_current_file / '.env',
            mock_current_file.parent / '.env', 
            mock_current_file.parent.parent / '.env',
            mock_current_file.parent.parent.parent / 'chat' / '.env'
        ]
        
        # Make all paths return False for exists() initially
        for mock_env_path in mock_env_paths:
            mock_env_path.exists.return_value = False
        
        # Make the third path exist and test that it's loaded
        mock_env_paths[2].exists.return_value = True
        
        with patch('core.config._load_env_from_path') as mock_load:
            load_env_file()
            mock_load.assert_called_once_with(mock_env_paths[2])


class TestApiConfiguration:
    """Test API configuration management."""
    
    def setUp(self):
        """Clean environment before each test."""
        env_vars = ['AZURE_OPENAI_KEY', 'AZURE_OPENAI_ENDPOINT', 'OPENAI_API_KEY', 'OPENAI_API_BASE']
        for var in env_vars:
            if var in os.environ:
                del os.environ[var]
    
    def test_get_api_config_azure_priority(self):
        """Test that Azure OpenAI takes priority over standard OpenAI."""
        with patch.dict(os.environ, {
            'AZURE_OPENAI_KEY': 'azure-key-123',
            'AZURE_OPENAI_ENDPOINT': 'https://azure.openai.azure.com/',
            'OPENAI_API_KEY': 'openai-key-456'  # Should be ignored
        }):
            api_key, api_base = get_api_config()
            
            assert api_key == 'azure-key-123'
            assert api_base == 'https://azure.openai.azure.com/'
    
    def test_get_api_config_openai_fallback(self):
        """Test fallback to standard OpenAI when Azure not available."""
        with patch.dict(os.environ, {
            'OPENAI_API_KEY': 'openai-key-789'
        }, clear=True):
            api_key, api_base = get_api_config(load_env=False)
            
            assert api_key == 'openai-key-789'
            assert api_base is None  # Default endpoint
    
    def test_get_api_config_openai_with_base(self):
        """Test OpenAI configuration with custom base URL."""
        with patch.dict(os.environ, {
            'OPENAI_API_KEY': 'openai-key-custom',
            'OPENAI_API_BASE': 'https://custom.openai.endpoint/'
        }, clear=True):
            api_key, api_base = get_api_config(load_env=False)
            
            assert api_key == 'openai-key-custom'
            assert api_base == 'https://custom.openai.endpoint/'
    
    def test_get_api_config_no_credentials(self):
        """Test error when no API credentials are provided."""
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(ValueError, match="No valid API configuration found"):
                get_api_config(load_env=False)
    
    def test_get_api_config_azure_incomplete(self):
        """Test Azure config with missing endpoint."""
        with patch.dict(os.environ, {
            'AZURE_OPENAI_KEY': 'azure-key-only'
            # Missing AZURE_OPENAI_ENDPOINT
        }, clear=True):
            with pytest.raises(ValueError, match="No valid API configuration found"):
                get_api_config(load_env=False)
    
    def test_get_api_config_with_load_env_file(self):
        """Test that get_api_config loads .env file."""
        with patch('core.config.load_env_file') as mock_load:
            with patch.dict(os.environ, {'OPENAI_API_KEY': 'test-key'}):
                get_api_config()
                mock_load.assert_called_once()


class TestApiKeyFunction:
    """Test get_api_key convenience function."""
    
    def test_get_api_key_success(self):
        """Test successful API key retrieval."""
        with patch('core.config.get_api_config') as mock_config:
            mock_config.return_value = ('test-key', 'https://api.test.com')
            
            result = get_api_key()
            
            assert result == 'test-key'
            mock_config.assert_called_once()
    
    def test_get_api_key_none_returned(self):
        """Test error when get_api_config returns None key."""
        with patch('core.config.get_api_config') as mock_config:
            mock_config.return_value = (None, 'https://api.test.com')
            
            with pytest.raises(ValueError, match="No API key found in configuration"):
                get_api_key()
    
    def test_get_api_key_empty_string(self):
        """Test error when get_api_config returns empty string."""
        with patch('core.config.get_api_config') as mock_config:
            mock_config.return_value = ('', 'https://api.test.com')
            
            with pytest.raises(ValueError, match="No API key found in configuration"):
                get_api_key()


class TestModelConfiguration:
    """Test model configuration management."""
    
    def test_get_model_config_structure(self):
        """Test model configuration structure and values."""
        config = get_model_config()
        
        # Check all required keys are present
        required_keys = [
            "entity_model", "triple_model", "judgment_model",
            "temperature", "max_tokens", "timeout", "max_retries",
            "progressive_timeouts", "reasoning_efforts"
        ]
        
        for key in required_keys:
            assert key in config, f"Missing required key: {key}"
        
        # Check specific values
        assert config["entity_model"] == GPT5_MINI_MODEL
        assert config["triple_model"] == GPT5_MINI_MODEL
        assert config["judgment_model"] == PERPLEXITY_MODEL
        assert config["temperature"] == OPENAI_TEMPERATURE
        assert config["max_tokens"] == OPENAI_MAX_TOKENS
        assert config["timeout"] == DEFAULT_TIMEOUT
        assert config["max_retries"] == MAX_RETRIES

        # Check new progressive timeout configuration
        assert config["progressive_timeouts"] == [120, 180, 240]
        assert config["reasoning_efforts"] == ["minimal", "medium", None]
    
    def test_get_model_config_types(self):
        """Test that model configuration values have correct types."""
        config = get_model_config()
        
        # String values
        assert isinstance(config["entity_model"], str)
        assert isinstance(config["triple_model"], str)
        assert isinstance(config["judgment_model"], str)
        
        # Numeric values
        assert isinstance(config["temperature"], (int, float))
        assert isinstance(config["max_tokens"], int)
        assert isinstance(config["timeout"], int)
        assert isinstance(config["max_retries"], int)
        
        # Value ranges
        assert 0.0 <= config["temperature"] <= 2.0
        assert config["max_tokens"] > 0
        assert config["timeout"] > 0
        assert config["max_retries"] >= 0


class TestEnvironmentVariableHandling:
    """Test environment variable handling edge cases."""
    
    def test_whitespace_handling(self):
        """Test handling of whitespace in environment variables."""
        with patch.dict(os.environ, {
            'OPENAI_API_KEY': '  test-key-with-spaces  '
        }, clear=True):
            api_key, _ = get_api_config(load_env=False)
            # Config should handle whitespace properly
            assert api_key == '  test-key-with-spaces  '  # Current implementation preserves spaces
    
    def test_empty_string_handling(self):
        """Test handling of empty string environment variables."""
        with patch.dict(os.environ, {
            'OPENAI_API_KEY': '',
            'AZURE_OPENAI_KEY': '   ',  # Whitespace only
            'AZURE_OPENAI_ENDPOINT': ''
        }, clear=True):
            with pytest.raises(ValueError, match="No valid API configuration found"):
                get_api_config(load_env=False)
    
    def test_case_sensitivity(self):
        """Test that environment variable names are case sensitive."""
        import platform
        
        if platform.system() == 'Windows':
            # Windows environment variables are case-insensitive, so this test
            # will actually work and find the key
            with patch.dict(os.environ, {
                'openai_api_key': 'lowercase-key'  # This becomes OPENAI_API_KEY on Windows
            }, clear=True):
                api_key, _ = get_api_config(load_env=False)
                assert api_key == 'lowercase-key'  # Should find the key on Windows
        else:
            # On Unix systems, environment variables are case-sensitive
            with patch.dict(os.environ, {
                'openai_api_key': 'lowercase-key'  # Wrong case
            }, clear=True):
                with pytest.raises(ValueError, match="No valid API configuration found"):
                    get_api_config(load_env=False)


# Integration tests
class TestConfigurationIntegration:
    """Integration tests for configuration components."""
    
    def test_full_configuration_workflow(self):
        """Test complete configuration loading workflow."""
        # Test the full configuration workflow with clean environment
        with patch.dict(os.environ, {
            'OPENAI_API_KEY': 'integration-test-key',
            'OPENAI_API_BASE': 'https://integration.test.com/v1'
        }, clear=True):
            # Test all configuration functions together
            api_key = get_api_key(load_env=False)
            api_key_full, api_base_full = get_api_config(load_env=False)
            model_config = get_model_config()
            
            # Verify results
            assert api_key == 'integration-test-key'
            assert api_key_full == 'integration-test-key'
            assert api_base_full == 'https://integration.test.com/v1'
            assert isinstance(model_config, dict)
            assert len(model_config) == 9  # Expected number of config keys (updated)


class TestErrorMessages:
    """Test error message quality and user-friendliness."""
    
    def test_api_config_error_message_content(self):
        """Test that error messages are helpful for users."""
        with patch.dict(os.environ, {}, clear=True):
            try:
                get_api_config(load_env=False)
                pytest.fail("Should have raised ValueError")
            except ValueError as e:
                error_msg = str(e)
                
                # Check that error message contains useful information
                assert "AZURE_OPENAI_KEY" in error_msg
                assert "AZURE_OPENAI_ENDPOINT" in error_msg
                assert "OPENAI_API_KEY" in error_msg
                assert "Please set either" in error_msg
    
    def test_api_key_error_message_content(self):
        """Test get_api_key error message."""
        with patch('core.config.get_api_config') as mock_config:
            mock_config.return_value = (None, None)
            
            try:
                get_api_key()
                pytest.fail("Should have raised ValueError")
            except ValueError as e:
                error_msg = str(e)
                assert "No API key found in configuration" in error_msg


# Performance and edge case tests
class TestPerformanceAndEdgeCases:
    """Test performance and edge cases."""
    
    def test_repeated_config_calls(self):
        """Test that repeated configuration calls work correctly."""
        with patch.dict(os.environ, {'OPENAI_API_KEY': 'test-key'}):
            # Call multiple times
            results = [get_api_config() for _ in range(5)]
            
            # All results should be identical
            for result in results:
                assert result == results[0]
    
    def test_large_env_file(self):
        """Test handling of large .env files."""
        # Create a large .env file with many variables
        large_env_content = "\n".join([
            f"VAR_{i}=value_{i}" for i in range(1000)
        ]) + "\nOPENAI_API_KEY=test-key\n"
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.env', delete=False) as f:
            f.write(large_env_content)
            temp_path = Path(f.name)
        
        try:
            # Clear environment
            if 'OPENAI_API_KEY' in os.environ:
                del os.environ['OPENAI_API_KEY']
            
            _load_env_from_path(temp_path)
            
            # Should still find the API key
            assert os.environ['OPENAI_API_KEY'] == 'test-key'
            
        finally:
            temp_path.unlink()
            if 'OPENAI_API_KEY' in os.environ:
                del os.environ['OPENAI_API_KEY']