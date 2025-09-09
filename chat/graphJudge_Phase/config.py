"""
Configuration constants and settings for the GraphJudge system.

This module contains all configuration constants used throughout the
GraphJudge system, including API settings, processing parameters,
and system defaults.
"""

import os
from typing import Dict, Any

# ==================== Perplexity API Configuration ====================

# Default model for graph judgment
PERPLEXITY_MODEL = "perplexity/sonar-reasoning"

# Concurrency and rate limiting settings
PERPLEXITY_CONCURRENT_LIMIT = 3  # Perplexity allows higher concurrency
PERPLEXITY_RETRY_ATTEMPTS = 3
PERPLEXITY_BASE_DELAY = 0.5  # Faster response times
PERPLEXITY_REASONING_EFFORT = "medium"  # For graph judgment accuracy

# Model selection options
PERPLEXITY_MODELS = {
    "sonar-pro": "perplexity/sonar-pro",
    "sonar-reasoning": "perplexity/sonar-reasoning",
    "sonar-reasoning-pro": "perplexity/sonar-reasoning-pro"
}

# ==================== Gold Label Bootstrapping Configuration ====================

GOLD_BOOTSTRAP_CONFIG = {
    'fuzzy_threshold': 0.8,      # RapidFuzz similarity threshold
    'sample_rate': 0.15,         # 15% sampling rate for manual review
    'llm_batch_size': 10,        # Batch size for LLM semantic evaluation
    'max_source_lines': 1000,    # Maximum source lines to process
    'random_seed': 42            # For reproducible sampling
}

# ==================== Logging Configuration ====================

# Log directory path
LOG_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "logs", "iteration2")

# ==================== Dataset Configuration ====================

# Dataset folder and iteration settings
folder = "KIMI_result_DreamOf_RedChamber"

# Support environment variables for pipeline integration
iteration = int(os.environ.get('PIPELINE_ITERATION', '2'))
input_file = os.environ.get('PIPELINE_INPUT_FILE', 
                           f"../datasets/{folder}/Graph_Iteration{iteration}/test_instructions_context_kimi_v2.json")
output_file = os.environ.get('PIPELINE_OUTPUT_FILE', 
                            f"../datasets/{folder}/Graph_Iteration{iteration}/pred_instructions_context_perplexity_itr{iteration}.csv")

# ==================== Processing Configuration ====================

# Default processing parameters
DEFAULT_TEMPERATURE = 0.2
DEFAULT_MAX_TOKENS = 2000

# ==================== Error Handling Configuration ====================

# Error handling settings
MAX_RETRY_ATTEMPTS = 3
BASE_RETRY_DELAY = 0.5
MAX_RETRY_DELAY = 10.0

# ==================== Output Configuration ====================

# Output file settings
DEFAULT_ENCODING = "utf-8"
CSV_DELIMITER = ","

# ==================== Validation Configuration ====================

# Input validation settings
MIN_INSTRUCTION_LENGTH = 10
MAX_INSTRUCTION_LENGTH = 1000
REQUIRED_FIELDS = ["instruction"]

# ==================== Performance Configuration ====================

# Performance optimization settings
STREAM_UPDATE_FREQUENCY = 5  # Update streaming display every N chunks
STREAM_CHUNK_DELAY = 0.02    # Delay between stream chunks
