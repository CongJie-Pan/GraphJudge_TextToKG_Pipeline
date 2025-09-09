#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unified CLI Pipeline Architecture Package

This package provides a unified command-line interface for the knowledge graph
generation pipeline with interactive iteration management and automated workflows.

Components:
- cli.py: Main CLI controller with command parsing and orchestration
- iteration_manager.py: Iteration directory creation, tracking, and management  
- config_manager.py: Configuration file creation, loading, and management
- stage_manager.py: Pipeline stage execution and dependency management
- pipeline_monitor.py: Real-time monitoring, logging, and performance tracking

Features:
- Interactive iteration number prompting with suggestions
- Automatic directory structure creation following improvement_plan2.md
- Dynamic configuration management with iteration-specific settings
- Comprehensive progress monitoring and recovery mechanisms
- Unified interface for existing scripts (run_entity.py, run_triple.py, 
  run_gj.py, convert_Judge_To_jsonGraph.py)

Usage:
    # Interactive mode
    python cli.py run-pipeline --input ./datasets/DreamOf_RedChamber/chapter1_raw.txt
    
    # Direct iteration specification  
    python cli.py run-pipeline --input ./datasets/DreamOf_RedChamber/chapter1_raw.txt --iteration 3
    
    # Individual stages
    python cli.py run-ectd --parallel-workers 5
    python cli.py run-triple-generation --batch-size 10
    python cli.py run-graph-judge --explainable
    
    # Monitoring
    python cli.py status
    python cli.py logs --tail 100

Author: Engineering Team
Date: 2025-01-15
Version: 1.0.0
"""

# Add current directory to path for imports
import sys
from pathlib import Path
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

try:
    # Use absolute imports only for test compatibility
    from cli import KGPipeline
    from iteration_manager import IterationManager  
    from config_manager import ConfigManager, PipelineConfig
    from stage_manager import StageManager, ECTDStage, TripleGenerationStage, GraphJudgeStage, EvaluationStage
    from pipeline_monitor import PipelineMonitor, PerformanceMetrics, StageMetrics, PipelineMetrics
except ImportError as e:
    print(f"CLI import error: {e}")
    # Set all to None for graceful degradation
    KGPipeline = None
    IterationManager = None
    ConfigManager = None
    PipelineConfig = None
    StageManager = None
    ECTDStage = None
    TripleGenerationStage = None
    GraphJudgeStage = None
    EvaluationStage = None
    PipelineMonitor = None
    PerformanceMetrics = None
    StageMetrics = None
    PipelineMetrics = None

__version__ = "1.0.0"
__author__ = "Engineering Team"
__email__ = "engineering@example.com"

__all__ = [
    # Main classes
    "KGPipeline",
    "PipelineConfig",
    
    # Managers
    "IterationManager", 
    "ConfigManager",
    "StageManager",
    "PipelineMonitor",
    
    # Stage implementations
    "ECTDStage",
    "TripleGenerationStage", 
    "GraphJudgeStage",
    "EvaluationStage",
    
    # Data structures
    "PerformanceMetrics",
    "StageMetrics", 
    "PipelineMetrics"
]

# Package metadata
__package_info__ = {
    "name": "Unified CLI Pipeline Architecture",
    "description": "Unified command-line interface for knowledge graph generation pipeline",
    "version": __version__,
    "author": __author__,
    "features": [
        "Interactive iteration management",
        "Automatic directory structure creation", 
        "Dynamic configuration management",
        "Real-time progress monitoring",
        "Unified script interface",
        "Comprehensive error handling",
        "Performance tracking",
        "Recovery mechanisms"
    ],
    "supported_stages": [
        "ECTD (Entity Extraction & Text Denoising)",
        "Triple Generation", 
        "Graph Judge",
        "Evaluation"
    ],
    "requirements": [
        "python >= 3.7",
        "pyyaml",
        "psutil", 
        "asyncio"
    ]
}


def get_package_info():
    """
    Get package information.
    
    Returns:
        Dictionary with package metadata
    """
    return __package_info__


def print_package_info():
    """
    Print package information to console.
    """
    info = __package_info__
    
    print("=" * 60)
    print(f" {info['name']}")
    print("=" * 60)
    print(f"Version: {info['version']}")
    print(f"Author: {info['author']}")
    print(f"Description: {info['description']}")
    
    print(f"\n Features:")
    for feature in info['features']:
        print(f"  - {feature}")
    
    print(f"\n Supported Stages:")
    for stage in info['supported_stages']:
        print(f"  - {stage}")
    
    print(f"\n Requirements:")
    for req in info['requirements']:
        print(f"  - {req}")
    
    print("=" * 60)


# Initialization message
def _init_message():
    """Display initialization message when package is imported."""
    print(" Unified CLI Pipeline Architecture - Package Loaded")
    print(f"   Version: {__version__}")
    print(f"   Components: {len(__all__)} modules available")


# Only show init message when running as main module
if __name__ == "__main__":
    print_package_info()
else:
    # Show brief init message when imported
    pass  # Commented out to avoid spam during imports
    # _init_message()
