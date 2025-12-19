"""
OntoRAG Core Configuration Module

This module provides global configuration settings for the OntoRAG library.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

# Configure default logger
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger("ontorag")


@dataclass
class OntoRAGConfig:
    """
    Global configuration for OntoRAG.
    
    Attributes:
        embedding_model: Name of the sentence transformer model for embeddings.
        api_delay_seconds: Delay between API calls to respect rate limits.
        max_retries: Maximum number of retries for failed API calls.
        user_agent: User agent string for HTTP requests.
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR).
        cache_embeddings: Whether to cache computed embeddings.
        default_language: Default language for text processing.
    
    Example:
        >>> from ontorag.core import config, OntoRAGConfig
        >>> 
        >>> # Modify global config
        >>> config.api_delay_seconds = 5.0
        >>> config.log_level = "DEBUG"
        >>> 
        >>> # Or create custom config
        >>> my_config = OntoRAGConfig(embedding_model="all-mpnet-base-v2")
    """
    
    # Embedding settings
    embedding_model: str = "all-MiniLM-L6-v2"
    embedding_dimension: int = 384
    
    # API settings
    api_delay_seconds: float = 2.0
    max_retries: int = 3
    request_timeout: int = 30
    user_agent: str = "OntoRAG/0.2.0 (https://github.com/ontorag/ontorag)"
    
    # Logging
    log_level: str = "INFO"
    
    # Caching
    cache_embeddings: bool = True
    cache_directory: str = ".ontorag_cache"
    
    # Language
    default_language: str = "en"
    
    # LLM defaults
    default_llm_temperature: float = 0.2
    default_llm_max_tokens: int = 2048
    
    # Search defaults
    default_search_k: int = 10
    semantic_weight: float = 0.6
    structural_weight: float = 0.4
    
    def set_log_level(self, level: str) -> None:
        """Set the logging level."""
        self.log_level = level
        logger.setLevel(getattr(logging, level.upper()))
    
    @property
    def http_headers(self) -> dict:
        """Default HTTP headers for API requests."""
        return {
            "User-Agent": self.user_agent,
            "Accept": "application/json",
        }


# Global configuration instance
config = OntoRAGConfig()


def get_config() -> OntoRAGConfig:
    """Get the global configuration instance."""
    return config


def set_config(**kwargs) -> None:
    """
    Update global configuration settings.
    
    Example:
        >>> from ontorag.core import set_config
        >>> set_config(api_delay_seconds=5.0, log_level="DEBUG")
    """
    global config
    for key, value in kwargs.items():
        if hasattr(config, key):
            setattr(config, key, value)
        else:
            raise ValueError(f"Unknown config option: {key}")


__all__ = [
    "OntoRAGConfig",
    "config",
    "get_config",
    "set_config",
    "logger",
]
