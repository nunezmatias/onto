"""
Academic Source Connectors

This package provides connectors to various academic databases:
- ArxivSource: arXiv.org preprint server
- SemanticScholarSource: Semantic Scholar academic graph
- OpenAlexSource: OpenAlex open academic catalog

Example - Search multiple sources:
    >>> from ontorag.corpus.sources import ArxivSource, SemanticScholarSource, OpenAlexSource
    >>> 
    >>> # Initialize sources
    >>> arxiv = ArxivSource()
    >>> ss = SemanticScholarSource()
    >>> openalex = OpenAlexSource()
    >>> 
    >>> # Search each
    >>> query = "climate adaptation flood"
    >>> arxiv_docs = arxiv.search(query, max_results=50)
    >>> ss_docs = ss.search(query, max_results=50)
    >>> openalex_docs = openalex.search(query, max_results=50)
"""

from .arxiv import ArxivSource
from .semantic_scholar import SemanticScholarSource
from .openalex import OpenAlexSource

# Registry of available sources
AVAILABLE_SOURCES = {
    "arxiv": ArxivSource,
    "semantic_scholar": SemanticScholarSource,
    "openalex": OpenAlexSource,
}


def get_source(name: str, **kwargs):
    """
    Get a source instance by name.
    
    Args:
        name: Source name ("arxiv", "semantic_scholar", "openalex")
        **kwargs: Arguments to pass to source constructor
        
    Returns:
        Source instance
        
    Example:
        >>> source = get_source("arxiv")
        >>> docs = source.search("climate change")
    """
    if name not in AVAILABLE_SOURCES:
        raise ValueError(f"Unknown source: {name}. Available: {list(AVAILABLE_SOURCES.keys())}")
    
    return AVAILABLE_SOURCES[name](**kwargs)


__all__ = [
    "ArxivSource",
    "SemanticScholarSource", 
    "OpenAlexSource",
    "AVAILABLE_SOURCES",
    "get_source",
]
