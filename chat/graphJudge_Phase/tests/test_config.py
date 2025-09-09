"""
Unit tests for the config module.

Tests configuration constants, environment variable handling,
and settings validation for the GraphJudge system.
"""

import pytest
import os
from unittest.mock import patch, mock_open

from ..config import (
    PERPLEXITY_MODEL,
    PERPLEXITY_MODELS,
    GOLD_BOOTSTRAP_CONFIG,
    LOG_DIR,
    iteration,
    input_file,
    output_file
)


class TestConfigConstants:
    """Test configuration constants and default values."""
    
    def test_perplexity_model_constants(self):
        """Test Perplexity model configuration constants."""
        assert PERPLEXITY_MODEL == "perplexity/sonar-reasoning"
        assert isinstance(PERPLEXITY_MODELS, dict)
        assert "sonar-reasoning" in PERPLEXITY_MODELS
        assert "sonar-pro" in PERPLEXITY_MODELS
        assert PERPLEXITY_MODELS["sonar-reasoning"] == "perplexity/sonar-reasoning"
    
    def test_gold_bootstrap_config(self):
        """Test gold label bootstrapping configuration."""
        assert isinstance(GOLD_BOOTSTRAP_CONFIG, dict)
        
        # Check required keys
        required_keys = ['fuzzy_threshold', 'sample_rate', 'llm_batch_size', 
                        'max_source_lines', 'random_seed']
        for key in required_keys:
            assert key in GOLD_BOOTSTRAP_CONFIG
        
        # Check value types and ranges
        assert 0.0 <= GOLD_BOOTSTRAP_CONFIG['fuzzy_threshold'] <= 1.0
        assert 0.0 <= GOLD_BOOTSTRAP_CONFIG['sample_rate'] <= 1.0
        assert GOLD_BOOTSTRAP_CONFIG['llm_batch_size'] > 0
        assert GOLD_BOOTSTRAP_CONFIG['max_source_lines'] > 0
        assert isinstance(GOLD_BOOTSTRAP_CONFIG['random_seed'], int)
    
    def test_log_directory_configuration(self):
        """Test log directory configuration."""
        assert isinstance(LOG_DIR, str)
        assert len(LOG_DIR) > 0
        # Should contain expected path components
        assert "logs" in LOG_DIR
        assert "iteration2" in LOG_DIR


class TestEnvironmentVariables:
    """Test environment variable handling."""
    
    def test_default_iteration_value(self):
        """Test default iteration value when environment variable is not set."""
        import importlib
        with patch.dict(os.environ, {}, clear=True):
            # Re-import to get fresh environment variable reading
            from .. import config
            importlib.reload(config)
            # Default should be 2
            assert config.iteration == 2
    
    @patch.dict(os.environ, {'PIPELINE_ITERATION': '5'}, clear=True)
    def test_custom_iteration_value(self):
        """Test custom iteration value from environment variable."""
        import importlib
        from .. import config
        importlib.reload(config)
        assert config.iteration == 5
    
    @patch.dict(os.environ, {'PIPELINE_INPUT_FILE': '/custom/input.json'}, clear=True)
    def test_custom_input_file_path(self):
        """Test custom input file path from environment variable."""
        import importlib
        from .. import config
        importlib.reload(config)
        assert config.input_file == '/custom/input.json'
    
    @patch.dict(os.environ, {'PIPELINE_OUTPUT_FILE': '/custom/output.csv'}, clear=True)
    def test_custom_output_file_path(self):
        """Test custom output file path from environment variable."""
        import importlib
        from .. import config
        importlib.reload(config)
        assert config.output_file == '/custom/output.csv'
    
    def test_file_path_generation(self):
        """Test automatic file path generation with iteration."""
        # Test that paths include iteration number
        assert str(iteration) in input_file
        assert str(iteration) in output_file
        
        # Test file extensions
        assert input_file.endswith('.json')
        assert output_file.endswith('.csv')
        
        # Test folder structure
        assert "KIMI_result_DreamOf_RedChamber" in input_file
        assert "KIMI_result_DreamOf_RedChamber" in output_file


class TestConfigValidation:
    """Test configuration validation and constraints."""
    
    def test_processing_configuration_values(self):
        """Test processing configuration values are valid."""
        from .. import config
        
        # Check if these constants exist and have reasonable values
        if hasattr(config, 'DEFAULT_TEMPERATURE'):
            assert 0.0 <= config.DEFAULT_TEMPERATURE <= 2.0
        
        if hasattr(config, 'DEFAULT_MAX_TOKENS'):
            assert config.DEFAULT_MAX_TOKENS > 0
        
        if hasattr(config, 'PERPLEXITY_CONCURRENT_LIMIT'):
            assert 1 <= config.PERPLEXITY_CONCURRENT_LIMIT <= 10
        
        if hasattr(config, 'PERPLEXITY_RETRY_ATTEMPTS'):
            assert 1 <= config.PERPLEXITY_RETRY_ATTEMPTS <= 10
    
    def test_error_handling_configuration(self):
        """Test error handling configuration values."""
        from .. import config
        
        if hasattr(config, 'MAX_RETRY_ATTEMPTS'):
            assert config.MAX_RETRY_ATTEMPTS > 0
        
        if hasattr(config, 'BASE_RETRY_DELAY'):
            assert config.BASE_RETRY_DELAY > 0
            
        if hasattr(config, 'MAX_RETRY_DELAY') and hasattr(config, 'BASE_RETRY_DELAY'):
            assert config.MAX_RETRY_DELAY > config.BASE_RETRY_DELAY
    
    def test_validation_configuration(self):
        """Test input validation configuration."""
        from .. import config
        
        if hasattr(config, 'MIN_INSTRUCTION_LENGTH'):
            assert config.MIN_INSTRUCTION_LENGTH > 0
            
        if hasattr(config, 'MAX_INSTRUCTION_LENGTH') and hasattr(config, 'MIN_INSTRUCTION_LENGTH'):
            assert config.MAX_INSTRUCTION_LENGTH > config.MIN_INSTRUCTION_LENGTH
            
        if hasattr(config, 'REQUIRED_FIELDS'):
            assert isinstance(config.REQUIRED_FIELDS, list)
            assert "instruction" in config.REQUIRED_FIELDS
    
    def test_output_configuration(self):
        """Test output configuration settings."""
        from .. import config
        
        if hasattr(config, 'DEFAULT_ENCODING'):
            assert config.DEFAULT_ENCODING == "utf-8"
            
        if hasattr(config, 'CSV_DELIMITER'):
            assert config.CSV_DELIMITER == ","


class TestConfigModuleImport:
    """Test config module import and accessibility."""
    
    def test_all_required_constants_available(self):
        """Test that all required constants are available for import."""
        from .. import config
        
        required_constants = [
            'PERPLEXITY_MODEL',
            'PERPLEXITY_MODELS', 
            'GOLD_BOOTSTRAP_CONFIG',
            'LOG_DIR',
            'iteration',
            'input_file',
            'output_file'
        ]
        
        for constant in required_constants:
            assert hasattr(config, constant)
    
    def test_config_constants_are_immutable_types(self):
        """Test that config constants use immutable types where appropriate."""
        # String constants should be strings
        assert isinstance(PERPLEXITY_MODEL, str)
        assert isinstance(LOG_DIR, str)
        
        # Numeric constants should be numbers
        assert isinstance(iteration, int)
        
        # Dict constants should be dicts (mutable but convention)
        assert isinstance(PERPLEXITY_MODELS, dict)
        assert isinstance(GOLD_BOOTSTRAP_CONFIG, dict)


class TestConfigEnvironmentIntegration:
    """Test integration with environment and pipeline."""
    
    @patch.dict(os.environ, {
        'PIPELINE_ITERATION': '10',
        'PIPELINE_INPUT_FILE': '/test/integration/input.json',
        'PIPELINE_OUTPUT_FILE': '/test/integration/output.csv'
    }, clear=False)
    def test_pipeline_environment_integration(self):
        """Test full pipeline environment integration."""
        import importlib
        from .. import config
        importlib.reload(config)
        
        # All environment variables should be respected
        assert config.iteration == 10
        assert config.input_file == '/test/integration/input.json'
        assert config.output_file == '/test/integration/output.csv'
    
    def test_config_for_different_iterations(self):
        """Test configuration works for different iteration numbers."""
        test_iterations = ['1', '2', '5', '10']
        
        for iter_num in test_iterations:
            with patch.dict(os.environ, {'PIPELINE_ITERATION': iter_num}, clear=False):
                import importlib
                from .. import config
                importlib.reload(config)
                
                assert config.iteration == int(iter_num)
                assert f"Iteration{iter_num}" in config.input_file
                assert f"itr{iter_num}" in config.output_file
    
    def test_folder_structure_consistency(self):
        """Test that folder structure is consistent across iterations."""
        # Input and output should use same base folder
        base_folder = "KIMI_result_DreamOf_RedChamber"
        
        assert base_folder in input_file
        assert base_folder in output_file
        
        # Both should reference the same iteration
        assert f"Iteration{iteration}" in input_file or f"Graph_Iteration{iteration}" in input_file
        assert f"itr{iteration}" in output_file
