"""
Unit Tests for API Configuration Module

This test suite validates the functionality of the config.py module,
ensuring that environment variable loading, validation, and error handling
work correctly across different scenarios for multiple AI API providers.

Test Coverage:
- OpenAI API key and base URL loading from environment variables
- Moonshot AI API configuration and validation
- Validation functions for proper configuration
- Error handling for missing or invalid credentials
- URL format standardization
- Individual getter functions for different API providers

Run with: pytest test_config.py
"""

import pytest
import os
from unittest.mock import patch
import sys
import tempfile

# Add the parent directory to the path to import the config module
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the functions we want to test
from config import (
    get_api_config,
    validate_api_config,
    get_api_key,
    get_api_base,
    get_moonshot_api_config,
    get_moonshot_api_base,
    validate_moonshot_api_config
)


class BaseConfigTest:
    """Base class for configuration tests with proper environment isolation."""
    
    def setup_method(self):
        """Set up method called before each test method."""
        # Store original environment
        self.original_env = os.environ.copy()
    
    def teardown_method(self):
        """Tear down method called after each test method."""
        # Restore original environment
        os.environ.clear()
        os.environ.update(self.original_env)


class TestGetApiConfig(BaseConfigTest):
    """Test cases for the get_api_config function."""
    
    def test_get_api_config_with_valid_env_vars(self):
        """Test successful API config loading with valid environment variables."""
        test_api_key = "sk-test123456789abcdef"
        test_api_base = "https://api.openai.com/v1"
        
        with patch.dict(os.environ, {
            'OPENAI_API_KEY': test_api_key,
            'OPENAI_API_BASE': test_api_base
        }, clear=True):
            with patch('config.load_env_file'):  # Mock load_env_file to prevent external .env loading
                api_key, api_base = get_api_config()
                
                assert api_key == test_api_key
                assert api_base == test_api_base
    
    def test_get_api_config_with_default_base_url(self):
        """Test API config loading with default base URL when not specified."""
        test_api_key = "sk-test123456789abcdef"
        
        with patch.dict(os.environ, {
            'OPENAI_API_KEY': test_api_key
        }, clear=True):
            with patch('config.load_env_file'):  # Mock load_env_file to prevent external .env loading
                api_key, api_base = get_api_config()
                
                assert api_key == test_api_key
                assert api_base == "https://api.openai.com/v1"
    
    def test_get_api_config_missing_api_key(self):
        """Test error handling when API key is missing."""
        with patch.dict(os.environ, {}, clear=True):
            with patch('config.load_env_file'):  # Mock load_env_file to prevent external .env loading
                with pytest.raises(ValueError) as exc_info:
                    get_api_config()
                
                assert "API key not found" in str(exc_info.value)
    
    def test_get_api_config_empty_api_key(self):
        """Test error handling when API key is empty string."""
        with patch.dict(os.environ, {
            'OPENAI_API_KEY': ''
        }, clear=True):
            with patch('config.load_env_file'):  # Mock load_env_file to prevent external .env loading
                with pytest.raises(ValueError) as exc_info:
                    get_api_config()
                
                assert "API key not found" in str(exc_info.value)
    
    def test_get_api_config_whitespace_api_key(self):
        """Test error handling when API key contains only whitespace."""
        with patch.dict(os.environ, {
            'OPENAI_API_KEY': '   '
        }, clear=True):
            with patch('config.load_env_file'):  # Mock load_env_file to prevent external .env loading
                with pytest.raises(ValueError) as exc_info:
                    get_api_config()
                
                assert "API key not found" in str(exc_info.value)
    
    def test_api_base_url_formatting(self):
        """Test that API base URL is properly formatted."""
        test_api_key = "sk-test123456789abcdef"
        
        # Test URL without /v1 suffix
        with patch.dict(os.environ, {
            'OPENAI_API_KEY': test_api_key,
            'OPENAI_API_BASE': 'https://api.openai.com'
        }, clear=True):
            with patch('config.load_env_file'):
                _, api_base = get_api_config()
                assert api_base == "https://api.openai.com/v1"
        
        # Test URL without trailing slash
        with patch.dict(os.environ, {
            'OPENAI_API_KEY': test_api_key,
            'OPENAI_API_BASE': 'https://api.openai.com/'
        }, clear=True):
            with patch('config.load_env_file'):
                _, api_base = get_api_config()
                assert api_base == "https://api.openai.com/v1"
        
        # Test URL with existing /v1
        with patch.dict(os.environ, {
            'OPENAI_API_KEY': test_api_key,
            'OPENAI_API_BASE': 'https://api.openai.com/v1'
        }, clear=True):
            with patch('config.load_env_file'):
                _, api_base = get_api_config()
                assert api_base == "https://api.openai.com/v1"


class TestValidateApiConfig(BaseConfigTest):
    """Test cases for the validate_api_config function."""
    
    def test_validate_api_config_valid(self):
        """Test validation with valid configuration."""
        test_api_key = "sk-test123456789abcdef"
        test_api_base = "https://api.openai.com/v1"
        
        with patch.dict(os.environ, {
            'OPENAI_API_KEY': test_api_key,
            'OPENAI_API_BASE': test_api_base
        }, clear=True):
            with patch('config.load_env_file'):
                assert validate_api_config() is True
    
    def test_validate_api_config_missing_key(self):
        """Test validation with missing API key."""
        with patch.dict(os.environ, {}, clear=True):
            with patch('config.load_env_file'):
                assert validate_api_config() is False
    
    def test_validate_api_config_short_key(self):
        """Test validation with API key that's too short."""
        with patch.dict(os.environ, {
            'OPENAI_API_KEY': 'short'
        }, clear=True):
            with patch('config.load_env_file'):
                assert validate_api_config() is False
    
    def test_validate_api_config_invalid_base_url(self):
        """Test validation with invalid base URL."""
        test_api_key = "sk-test123456789abcdef"
        
        with patch.dict(os.environ, {
            'OPENAI_API_KEY': test_api_key,
            'OPENAI_API_BASE': 'invalid-url'
        }, clear=True):
            with patch('config.load_env_file'):
                assert validate_api_config() is False


class TestIndividualGetters(BaseConfigTest):
    """Test cases for individual getter functions."""
    
    def test_get_api_key(self):
        """Test get_api_key function."""
        test_api_key = "sk-test123456789abcdef"
        
        with patch.dict(os.environ, {
            'OPENAI_API_KEY': test_api_key
        }, clear=True):
            with patch('config.load_env_file'):
                assert get_api_key() == test_api_key
    
    def test_get_api_base(self):
        """Test get_api_base function."""
        test_api_key = "sk-test123456789abcdef"
        test_api_base = "https://custom.api.com/v1"
        
        with patch.dict(os.environ, {
            'OPENAI_API_KEY': test_api_key,
            'OPENAI_API_BASE': test_api_base
        }, clear=True):
            with patch('config.load_env_file'):
                assert get_api_base() == test_api_base
    
    def test_get_api_key_missing(self):
        """Test get_api_key with missing environment variable."""
        with patch.dict(os.environ, {}, clear=True):
            with patch('config.load_env_file'):
                with pytest.raises(ValueError):
                    get_api_key()
    
    def test_get_api_base_default(self):
        """Test get_api_base with default value."""
        test_api_key = "sk-test123456789abcdef"
        
        with patch.dict(os.environ, {
            'OPENAI_API_KEY': test_api_key
        }, clear=True):
            with patch('config.load_env_file'):
                assert get_api_base() == "https://api.openai.com/v1"


class TestMoonshotApiConfig(BaseConfigTest):
    """Test cases for the Moonshot AI API configuration functions."""
    
    def test_get_moonshot_api_config_valid(self):
        """Test successful Moonshot API config loading with valid environment variables."""
        test_api_key = "mk-test123456789abcdef"
        
        with patch.dict(os.environ, {
            'MOONSHOT_API_KEY': test_api_key
        }, clear=True):
            with patch('config.load_env_file'):
                api_key = get_moonshot_api_config()
                assert api_key == test_api_key
    
    def test_get_moonshot_api_config_missing_key(self):
        """Test error handling when Moonshot API key is missing."""
        with patch.dict(os.environ, {}, clear=True):
            with patch('config.load_env_file'):
                with pytest.raises(ValueError) as exc_info:
                    get_moonshot_api_config()
                
                assert "Moonshot API key not found" in str(exc_info.value)
    
    def test_get_moonshot_api_config_empty_key(self):
        """Test error handling when Moonshot API key is empty string."""
        with patch.dict(os.environ, {
            'MOONSHOT_API_KEY': ''
        }, clear=True):
            with patch('config.load_env_file'):
                with pytest.raises(ValueError) as exc_info:
                    get_moonshot_api_config()
                
                assert "Moonshot API key not found" in str(exc_info.value)
    
    def test_get_moonshot_api_config_whitespace_key(self):
        """Test error handling when Moonshot API key contains only whitespace."""
        with patch.dict(os.environ, {
            'MOONSHOT_API_KEY': '   '
        }, clear=True):
            with patch('config.load_env_file'):
                with pytest.raises(ValueError) as exc_info:
                    get_moonshot_api_config()
                
                assert "Moonshot API key not found" in str(exc_info.value)
    
    def test_get_moonshot_api_base_default(self):
        """Test Moonshot API base URL with default value."""
        with patch.dict(os.environ, {}, clear=True):
            with patch('config.load_env_file'):
                api_base = get_moonshot_api_base()
                assert api_base == "https://api.moonshot.ai/v1"
    
    def test_get_moonshot_api_base_custom(self):
        """Test Moonshot API base URL with custom value."""
        test_api_base = "https://custom.moonshot.com/v1"
        
        with patch.dict(os.environ, {
            'MOONSHOT_API_BASE': test_api_base
        }, clear=True):
            with patch('config.load_env_file'):
                api_base = get_moonshot_api_base()
                assert api_base == test_api_base
    
    def test_moonshot_api_base_url_formatting(self):
        """Test that Moonshot API base URL is properly formatted."""
        # Test URL without /v1 suffix
        with patch.dict(os.environ, {
            'MOONSHOT_API_BASE': 'https://api.moonshot.ai'
        }, clear=True):
            with patch('config.load_env_file'):
                api_base = get_moonshot_api_base()
                assert api_base == "https://api.moonshot.ai/v1"
        
        # Test URL without trailing slash
        with patch.dict(os.environ, {
            'MOONSHOT_API_BASE': 'https://api.moonshot.ai/'
        }, clear=True):
            with patch('config.load_env_file'):
                api_base = get_moonshot_api_base()
                assert api_base == "https://api.moonshot.ai/v1"
        
        # Test URL with existing /v1
        with patch.dict(os.environ, {
            'MOONSHOT_API_BASE': 'https://api.moonshot.ai/v1'
        }, clear=True):
            with patch('config.load_env_file'):
                api_base = get_moonshot_api_base()
                assert api_base == "https://api.moonshot.ai/v1"


class TestValidateMoonshotApiConfig(BaseConfigTest):
    """Test cases for the validate_moonshot_api_config function."""
    
    def test_validate_moonshot_api_config_valid(self):
        """Test validation with valid Moonshot configuration."""
        test_api_key = "mk-test123456789abcdef"
        test_api_base = "https://api.moonshot.ai/v1"
        
        with patch.dict(os.environ, {
            'MOONSHOT_API_KEY': test_api_key,
            'MOONSHOT_API_BASE': test_api_base
        }, clear=True):
            with patch('config.load_env_file'):
                assert validate_moonshot_api_config() is True
    
    def test_validate_moonshot_api_config_missing_key(self):
        """Test validation with missing Moonshot API key."""
        with patch.dict(os.environ, {}, clear=True):
            with patch('config.load_env_file'):
                assert validate_moonshot_api_config() is False
    
    def test_validate_moonshot_api_config_short_key(self):
        """Test validation with Moonshot API key that's too short."""
        with patch.dict(os.environ, {
            'MOONSHOT_API_KEY': 'short'
        }, clear=True):
            with patch('config.load_env_file'):
                assert validate_moonshot_api_config() is False
    
    def test_validate_moonshot_api_config_invalid_base_url(self):
        """Test validation with invalid Moonshot base URL."""
        test_api_key = "mk-test123456789abcdef"
        
        with patch.dict(os.environ, {
            'MOONSHOT_API_KEY': test_api_key,
            'MOONSHOT_API_BASE': 'invalid-url'
        }, clear=True):
            with patch('config.load_env_file'):
                assert validate_moonshot_api_config() is False


class TestMoonshotIntegration(BaseConfigTest):
    """Integration tests for Moonshot API configuration."""
    
    def test_moonshot_config_with_openai_coexistence(self):
        """Test that Moonshot and OpenAI configurations can coexist."""
        openai_key = "sk-openai123456789abcdef"
        moonshot_key = "mk-moonshot123456789abcdef"
        
        with patch.dict(os.environ, {
            'OPENAI_API_KEY': openai_key,
            'MOONSHOT_API_KEY': moonshot_key
        }, clear=True):
            with patch('config.load_env_file'):
                # Both configurations should work independently
                assert get_api_key() == openai_key
                assert get_moonshot_api_config() == moonshot_key
                
                # Both validations should pass
                assert validate_api_config() is True
                assert validate_moonshot_api_config() is True
    
    def test_moonshot_environment_isolation(self):
        """Test that Moonshot configuration is properly isolated."""
        # Test with different Moonshot keys
        with patch.dict(os.environ, {
            'MOONSHOT_API_KEY': 'mk-test-key-1'
        }, clear=True):
            with patch('config.load_env_file'):
                assert get_moonshot_api_config() == 'mk-test-key-1'
        
        with patch.dict(os.environ, {
            'MOONSHOT_API_KEY': 'mk-test-key-2'
        }, clear=True):
            with patch('config.load_env_file'):
                assert get_moonshot_api_config() == 'mk-test-key-2'


class TestIntegration(BaseConfigTest):
    """Integration tests for the configuration module."""
    
    def test_config_module_standalone_execution(self):
        """Test that the config module can be executed standalone for validation."""
        test_api_key = "sk-test123456789abcdef"
        
        with patch.dict(os.environ, {
            'OPENAI_API_KEY': test_api_key
        }, clear=True):
            with patch('config.load_env_file'):
                # This test ensures the module's __main__ block works correctly
                # We can't easily test the actual execution, but we can test the functions it uses
                try:
                    api_key, api_base = get_api_config()
                    assert len(api_key) > 0
                    assert api_base.startswith('https://')
                except Exception as e:
                    pytest.fail(f"Config module execution failed: {e}")
    
    def test_environment_variable_isolation(self):
        """Test that tests don't interfere with each other's environment variables."""
        # Test with different values
        with patch.dict(os.environ, {
            'OPENAI_API_KEY': 'test-key-1'
        }, clear=True):
            with patch('config.load_env_file'):
                assert get_api_key() == 'test-key-1'
        
        with patch.dict(os.environ, {
            'OPENAI_API_KEY': 'test-key-2'
        }, clear=True):
            with patch('config.load_env_file'):
                assert get_api_key() == 'test-key-2'


class TestPipelineEnvironmentVariableCompatibility(BaseConfigTest):
    """Test compatibility between API config and pipeline environment variables."""

    def test_pipeline_env_vars_do_not_interfere_with_api_config(self):
        """Test that PIPELINE_* environment variables do not interfere with API configuration."""
        # Set up pipeline environment variables
        pipeline_env_vars = {
            'PIPELINE_ITERATION': '5',
            'PIPELINE_DATASET_PATH': '../datasets/test/',
            'PIPELINE_INPUT_FILE': '/tmp/test/input.json',
            'PIPELINE_OUTPUT_FILE': '/tmp/test/output.csv',
            'PIPELINE_OUTPUT_DIR': '/tmp/test/output/'
        }
        
        # Set up API environment variables
        api_env_vars = {
            'OPENAI_API_KEY': 'test-openai-key',
            'OPENAI_API_BASE': 'https://api.openai.com/v1',
            'MOONSHOT_API_KEY': 'test-moonshot-key'
        }
        
        # Combine all environment variables
        combined_env = {**pipeline_env_vars, **api_env_vars}
        
        with patch.dict(os.environ, combined_env, clear=True):
            with patch('config.load_env_file'):  # Mock load_env_file to prevent external .env loading
                # Test that API configuration still works correctly
                assert get_api_key() == 'test-openai-key'
                assert get_api_base() == 'https://api.openai.com/v1'
                assert get_moonshot_api_config() == 'test-moonshot-key'
                
                # Test that pipeline variables are accessible but don't affect API config
                assert os.environ.get('PIPELINE_ITERATION') == '5'
                assert os.environ.get('PIPELINE_DATASET_PATH') == '../datasets/test/'
                
                # Verify API validation still works
                api_config = get_api_config()
                assert validate_api_config() is True

    def test_pipeline_env_vars_namespace_isolation(self):
        """Test that PIPELINE_* variables are properly isolated from API config namespace."""
        # Set up conflicting variable names (hypothetical scenario)
        env_vars = {
            'PIPELINE_API_KEY': 'pipeline-key',  # Should not interfere with OPENAI_API_KEY
            'PIPELINE_BASE_URL': 'pipeline-url',  # Should not interfere with API base
            'OPENAI_API_KEY': 'real-api-key'
        }
        
        with patch.dict(os.environ, env_vars, clear=True):
            with patch('config.load_env_file'):  # Mock load_env_file to prevent external .env loading
                # Verify that API config uses correct variables
                assert get_api_key() == 'real-api-key'
                
                # Verify pipeline variables are separate
                assert os.environ.get('PIPELINE_API_KEY') == 'pipeline-key'
                assert os.environ.get('PIPELINE_BASE_URL') == 'pipeline-url'

    def test_environment_variable_precedence(self):
        """Test that environment variable precedence works correctly with pipeline vars."""
        env_vars = {
            'OPENAI_API_KEY': 'env-api-key',
            'PIPELINE_ITERATION': '10',
            'PIPELINE_OUTPUT_DIR': '/custom/output/'
        }
        
        with patch.dict(os.environ, env_vars, clear=True):
            with patch('config.load_env_file'):  # Mock load_env_file to prevent external .env loading
                # API config should use standard precedence rules
                assert get_api_key() == 'env-api-key'
                
                # Pipeline variables should be directly accessible
                assert os.environ.get('PIPELINE_ITERATION') == '10'
                assert os.environ.get('PIPELINE_OUTPUT_DIR') == '/custom/output/'
                
                # Verify no cross-contamination
                api_config = get_api_config()
                assert 'PIPELINE_' not in str(api_config)


if __name__ == "__main__":
    # When run directly, execute all tests
    pytest.main([__file__, "-v"])