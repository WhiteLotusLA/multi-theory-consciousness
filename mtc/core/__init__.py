"""
MTC Core Module
================

Core framework utilities:
- Settings: Pydantic-based configuration with env var support
- get_settings: Singleton settings accessor
"""

from mtc.core.config import Settings, get_settings

__all__ = [
    "Settings",
    "get_settings",
]
