"""
Entity Extraction Phase Module

This package provides a modular architecture for GPT-5-mini Entity Extraction and Text Denoising (ECTD) Pipeline.
The module is designed following the condense principle with high cohesion, low coupling, and comprehensive testing.

Package Structure:
- core/: Core business logic (entity extraction, text denoising, pipeline orchestration)
- api/: API integration layer (GPT-5-mini client, rate limiting, caching)
- infrastructure/: Infrastructure concerns (logging, configuration, file management)
- models/: Data models and types (entities, pipeline state)
- utils/: Utility functions (Chinese text processing, validation, statistics)
- tests/: Comprehensive test suite (unit, integration, fixtures)

Usage:
    from extractEntity_Phase.core import EntityExtractor
    from extractEntity_Phase.api import GPT5MiniClient
    from extractEntity_Phase.infrastructure import Logger, Config

Author: Senior Google Engineer (AI Assistant)
Version: 1.0.0
Date: 2025-01-27
"""

__version__ = "1.0.0"
__author__ = "Senior Google Engineer (AI Assistant)"
__description__ = "Modular GPT-5-mini ECTD Pipeline for Chinese Text Processing"

# Core module imports
try:
    from .core.entity_extractor import EntityExtractor
    from .core.text_denoiser import TextDenoiser
    from .core.pipeline_orchestrator import PipelineOrchestrator
except ImportError:
    # Modules not yet implemented
    pass

# API module imports
try:
    from .api.gpt5mini_client import GPT5MiniClient
    from .api.rate_limiter import RateLimiter
    from .api.cache_manager import CacheManager
except ImportError:
    # Modules not yet implemented
    pass

# Infrastructure module imports
try:
    from .infrastructure.logging import Logger
    from .infrastructure.config import Config
except ImportError:
    # Modules not yet implemented
    pass

# Model imports
try:
    from .models.entities import Entity, EntityList, EntityType
    from .models.pipeline_state import PipelineState, ProcessingStatus
except ImportError:
    # Models not yet implemented
    pass

# Utility imports
try:
    from .utils.chinese_text import ChineseTextProcessor
except ImportError:
    # Utilities not yet implemented
    pass

__all__ = [
    # Core modules
    "EntityExtractor",
    "TextDenoiser", 
    "PipelineOrchestrator",
    
    # API modules
    "GPT5MiniClient",
    "RateLimiter",
    "CacheManager",
    
    # Infrastructure modules
    "Logger",
    "Config",
    
    # Models
    "Entity",
    "EntityList",
    "EntityType",
    "PipelineState",
    "ProcessingStatus",
    
    # Utilities
    "ChineseTextProcessor",
    
    # Backward compatibility
    "extract_entities",
    "denoise_text", 
    "main"
]


# Backward compatibility API
async def extract_entities(texts):
    """Extract entities from texts using GPT-5-mini."""
    from .core.entity_extractor import EntityExtractor
    from .api.gpt5mini_client import GPT5MiniClient
    
    client = GPT5MiniClient()
    extractor = EntityExtractor(client)
    results = await extractor.extract_entities_from_texts(texts)
    
    # Convert to original format
    return [[entity.text for entity in collection.entities] for collection in results]


async def denoise_text(texts, entities_list):
    """Denoise texts based on extracted entities."""
    from .core.text_denoiser import TextDenoiser
    from .api.gpt5mini_client import GPT5MiniClient
    
    client = GPT5MiniClient()
    denoiser = TextDenoiser(client)
    return await denoiser.denoise_texts(texts, entities_list)


async def main():
    """Main execution function for backward compatibility."""
    from .core.pipeline_orchestrator import PipelineOrchestrator
    
    orchestrator = PipelineOrchestrator()
    success = await orchestrator.run_pipeline()
    return success
