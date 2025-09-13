"""
GraphJudge Streamlit Pipeline Package

A refactored, modular implementation of the GraphJudge text-to-knowledge-graph 
pipeline optimized for Streamlit integration.

This package contains:
- Core pipeline modules (entity_processor, triple_generator, graph_judge)
- Utility modules (api_client, validation, error_handling)
- UI components for Streamlit
- Comprehensive test suite
"""

__version__ = "2.0.0"
__author__ = "GraphJudge Research Team"
__description__ = "GraphJudge Streamlit Pipeline - Text to Knowledge Graph"

# Make core modules easily accessible
from .core import (
    entity_processor,
    triple_generator, 
    graph_judge,
    pipeline,
    models,
    config
)

from .utils import (
    api_client,
    validation,
    error_handling,
    session_state,
    state_persistence,
    state_cleanup
)

__all__ = [
    'entity_processor',
    'triple_generator',
    'graph_judge', 
    'pipeline',
    'models',
    'config',
    'api_client',
    'validation',
    'error_handling',
    'session_state',
    'state_persistence',
    'state_cleanup'
]