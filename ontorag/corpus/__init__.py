"""
OntoRAG Corpus Module

This module provides tools for collecting and managing academic literature.

Key Components:
    - Document: Standardized document representation
    - Corpus: Collection manager with multi-source search
    - CorpusAnalyzer: Statistical analysis and visualization
    
Source Connectors:
    - ArxivSource: arXiv.org preprints
    - SemanticScholarSource: Semantic Scholar database
    - OpenAlexSource: OpenAlex academic catalog

Quick Start:
    >>> from ontorag.corpus import Corpus
    >>> 
    >>> # Create corpus and search
    >>> corpus = Corpus(name="climate_research")
    >>> corpus.search_all(
    ...     "climate adaptation flood mitigation",
    ...     max_per_source=100
    ... )
    >>> 
    >>> # Analyze
    >>> print(corpus.summary())
    >>> 
    >>> # Export
    >>> corpus.save("corpus.json")

Example - Search specific sources:
    >>> from ontorag.corpus.sources import ArxivSource, SemanticScholarSource
    >>> 
    >>> arxiv = ArxivSource()
    >>> papers = arxiv.search(
    ...     'abs:"climate change" AND abs:"adaptation"',
    ...     max_results=50
    ... )

Installation:
    For full corpus functionality, install the corpus extras:
    
    $ pip install ontorag[corpus]
    
    This includes: arxiv, requests, beautifulsoup4
"""

from .base import Document, BaseSource, SearchQuery
from .corpus import Corpus
from .analysis import CorpusAnalyzer, STOPWORDS

# Import sources (may fail if dependencies not installed)
try:
    from .sources import (
        ArxivSource,
        SemanticScholarSource,
        OpenAlexSource,
        AVAILABLE_SOURCES,
        get_source,
    )
    _SOURCES_AVAILABLE = True
except ImportError:
    _SOURCES_AVAILABLE = False
    ArxivSource = None
    SemanticScholarSource = None
    OpenAlexSource = None
    AVAILABLE_SOURCES = {}
    get_source = None


__all__ = [
    # Core
    "Document",
    "BaseSource",
    "SearchQuery",
    "Corpus",
    
    # Analysis
    "CorpusAnalyzer",
    "STOPWORDS",
    
    # Sources
    "ArxivSource",
    "SemanticScholarSource",
    "OpenAlexSource",
    "AVAILABLE_SOURCES",
    "get_source",
]
