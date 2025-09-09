"""
GraphJudge Phase - Modularized Perplexity API Graph Judge System

This module provides a modular implementation of the GraphJudge system
using Perplexity API for knowledge graph triple validation.
"""

from .config import *
from .data_structures import TripleData, BootstrapResult, ExplainableJudgment
from .graph_judge_core import PerplexityGraphJudge
from .gold_label_bootstrapping import GoldLabelBootstrapper
from .processing_pipeline import ProcessingPipeline
from .utilities import validate_input_file, create_output_directory
from .logging_system import setup_terminal_logging, TerminalLogger

__version__ = "1.0.0"
__author__ = "CongJie Pan"

# Main exports
__all__ = [
    'PerplexityGraphJudge',
    'GoldLabelBootstrapper', 
    'ProcessingPipeline',
    'TripleData',
    'BootstrapResult',
    'ExplainableJudgment',
    'validate_input_file',
    'create_output_directory',
    'setup_terminal_logging',
    'TerminalLogger'
]
