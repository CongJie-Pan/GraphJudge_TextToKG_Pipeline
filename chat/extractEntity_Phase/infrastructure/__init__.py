"""
Infrastructure Module

This module provides core infrastructure services:
- Logger: Dual-output logging system (terminal + file)
- Config: Configuration management with environment variable support

All modules are designed for high reliability and easy testing through
dependency injection and interface abstraction.
"""

from .logging import Logger
from .config import Config

__all__ = [
    "Logger",
    "Config",
]
