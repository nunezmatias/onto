"""
OntoRAG Extraction Module

This module provides tools for extracting entities and relations from text.

Key Components:
    - RuleBasedExtractor: Dictionary and pattern-based extraction
    - LLMExtractor: LLM-powered extraction (OpenAI, Anthropic, etc.)
    - HybridExtractor: Combines rule-based and LLM approaches
    - EntityDictionary: Configurable entity dictionaries

Quick Start:
    >>> from ontorag.extraction import RuleBasedExtractor
    >>> 
    >>> # Rule-based extraction (no API needed)
    >>> extractor = RuleBasedExtractor()
    >>> result = extractor.extract("Green roofs reduce urban heat by 2-3°C")
    >>> 
    >>> print(result.entities)  # Extracted entities
    >>> print(result.triplets)  # Extracted triplets

Example with LLM:
    >>> from ontorag.extraction import LLMExtractor
    >>> 
    >>> extractor = LLMExtractor(
    ...     provider="openai",
    ...     api_key="sk-..."
    ... )
    >>> result = extractor.extract(text)

Example with custom dictionary:
    >>> from ontorag.extraction import RuleBasedExtractor
    >>> 
    >>> my_entities = {
    ...     "DRUG": ["aspirin", "ibuprofen"],
    ...     "CONDITION": ["headache", "fever"]
    ... }
    >>> extractor = RuleBasedExtractor(entity_dict=my_entities)
"""

from .dictionaries import (
    CLIMATE_ENTITIES,
    RELATION_PATTERNS,
    EntityMatch,
    EntityDictionary,
)

from .rule_based import (
    ExtractedEntity,
    ExtractedTriplet,
    ExtractionResult,
    RuleBasedExtractor,
)

from .prompts import (
    EXTRACTION_PROMPT,
    format_extraction_prompt,
    get_domain_prompt,
    create_custom_prompt,
)

# LLM components (may fail if dependencies not installed)
try:
    from .llm_based import (
        LLMConfig,
        LLMExtractor,
        HybridExtractor,
    )
    _LLM_AVAILABLE = True
except ImportError:
    _LLM_AVAILABLE = False
    LLMConfig = None
    LLMExtractor = None
    HybridExtractor = None


# Convenience aliases
TripletExtractor = RuleBasedExtractor  # Backward compatibility


__all__ = [
    # Dictionaries
    "CLIMATE_ENTITIES",
    "RELATION_PATTERNS",
    "EntityMatch",
    "EntityDictionary",
    
    # Data classes
    "ExtractedEntity",
    "ExtractedTriplet",
    "ExtractionResult",
    
    # Extractors
    "RuleBasedExtractor",
    "TripletExtractor",
    "LLMConfig",
    "LLMExtractor",
    "HybridExtractor",
    
    # Prompts
    "EXTRACTION_PROMPT",
    "format_extraction_prompt",
    "get_domain_prompt",
    "create_custom_prompt",
]
