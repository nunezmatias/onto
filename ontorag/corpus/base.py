"""
Base Classes for Corpus Collection

This module provides abstract base classes for implementing data sources.
Extend these classes to add new repositories.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Iterator
import hashlib


@dataclass
class Document:
    """
    Represents a document in the corpus.
    
    This is the standardized format that all sources convert to.
    
    Attributes:
        id: Unique identifier (generated from source + source_id)
        source: Name of the source (arxiv, semantic_scholar, openalex)
        source_id: Original ID from the source
        title: Document title
        abstract: Document abstract/summary
        authors: List of author names
        year: Publication year
        url: Link to the document
        doi: Digital Object Identifier
        pdf_url: Link to PDF if available
        metadata: Additional source-specific metadata
    
    Example:
        >>> doc = Document(
        ...     source="arxiv",
        ...     source_id="2301.00001",
        ...     title="Climate Adaptation Study",
        ...     abstract="This paper examines...",
        ...     authors=["Smith, J.", "Jones, M."],
        ...     year=2023
        ... )
    """
    # Required fields
    source: str
    source_id: str
    title: str
    
    # Content
    abstract: str = ""
    full_text: Optional[str] = None
    
    # Authors
    authors: List[str] = field(default_factory=list)
    
    # Dates
    year: Optional[int] = None
    published_date: Optional[str] = None
    
    # Links
    url: Optional[str] = None
    doi: Optional[str] = None
    pdf_url: Optional[str] = None
    
    # Metrics
    citation_count: int = 0
    
    # Classification
    categories: List[str] = field(default_factory=list)
    keywords: List[str] = field(default_factory=list)
    
    # Additional data
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    # Generated
    id: str = field(init=False)
    
    def __post_init__(self):
        """Generate unique ID from source and source_id."""
        raw_id = f"{self.source}:{self.source_id}"
        self.id = hashlib.md5(raw_id.encode()).hexdigest()[:16]
    
    @property
    def authors_str(self) -> str:
        """Get authors as comma-separated string."""
        return ", ".join(self.authors[:5])
        
    @property
    def text_content(self) -> str:
        """Get combined text content."""
        parts = [self.title]
        if self.abstract:
            parts.append(self.abstract)
        if self.full_text:
            parts.append(self.full_text)
        return " ".join(parts)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "source": self.source,
            "source_id": self.source_id,
            "title": self.title,
            "abstract": self.abstract,
            "authors": self.authors,
            "authors_str": self.authors_str,
            "year": self.year,
            "published_date": self.published_date,
            "url": self.url,
            "doi": self.doi,
            "pdf_url": self.pdf_url,
            "citation_count": self.citation_count,
            "categories": self.categories,
            "keywords": self.keywords,
            "metadata": self.metadata,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Document":
        """Create from dictionary."""
        return cls(
            source=data.get("source", "unknown"),
            source_id=data.get("source_id", ""),
            title=data.get("title", ""),
            abstract=data.get("abstract", ""),
            full_text=data.get("full_text"),
            authors=data.get("authors", []),
            year=data.get("year"),
            published_date=data.get("published_date"),
            url=data.get("url"),
            doi=data.get("doi"),
            pdf_url=data.get("pdf_url"),
            citation_count=data.get("citation_count", 0),
            categories=data.get("categories", []),
            keywords=data.get("keywords", []),
            metadata=data.get("metadata", {}),
        )
    
    def __repr__(self) -> str:
        title_short = self.title[:50] + "..." if len(self.title) > 50 else self.title
        return f"Document('{title_short}', source='{self.source}')"


class BaseSource(ABC):
    """
    Abstract base class for corpus sources.
    
    Implement this class to add new data repositories.
    
    Required methods to implement:
        - search(query, max_results) -> List[Document]
        - get_by_id(doc_id) -> Optional[Document]
    
    Example:
        >>> class MySource(BaseSource):
        ...     name = "my_source"
        ...     
        ...     def search(self, query, max_results=100):
        ...         # Implement search logic
        ...         return [Document(...), ...]
        ...     
        ...     def get_by_id(self, doc_id):
        ...         # Implement retrieval by ID
        ...         return Document(...)
    """
    
    # Source identifier - override in subclass
    name: str = "base"
    
    # Base URL for API - override in subclass
    base_url: str = ""
    
    # API rate limiting
    requests_per_minute: int = 30
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the source.
        
        Args:
            api_key: Optional API key for authenticated access
        """
        self.api_key = api_key
        self._last_request_time: Optional[datetime] = None
    
    @abstractmethod
    def search(
        self,
        query: str,
        max_results: int = 100,
        **kwargs
    ) -> List[Document]:
        """
        Search for documents matching query.
        
        Args:
            query: Search query string
            max_results: Maximum number of results to return
            **kwargs: Source-specific parameters
            
        Returns:
            List of Document objects
        """
        pass
    
    @abstractmethod
    def get_by_id(self, doc_id: str) -> Optional[Document]:
        """
        Retrieve a document by its source-specific ID.
        
        Args:
            doc_id: Document ID in the source system
            
        Returns:
            Document if found, None otherwise
        """
        pass
    
    def search_iterator(
        self,
        query: str,
        max_results: int = 1000,
        batch_size: int = 100,
        **kwargs
    ) -> Iterator[Document]:
        """
        Iterate over search results in batches.
        
        Useful for large result sets to avoid memory issues.
        
        Args:
            query: Search query
            max_results: Maximum total results
            batch_size: Results per batch
            **kwargs: Additional search parameters
            
        Yields:
            Document objects
        """
        offset = 0
        total_yielded = 0
        
        while total_yielded < max_results:
            batch = self.search(
                query,
                max_results=min(batch_size, max_results - total_yielded),
                offset=offset,
                **kwargs
            )
            
            if not batch:
                break
            
            for doc in batch:
                yield doc
                total_yielded += 1
                if total_yielded >= max_results:
                    break
            
            offset += len(batch)
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"


@dataclass
class SearchQuery:
    """
    Structured search query.
    
    Allows building complex queries with multiple terms.
    
    Attributes:
        terms: List of search terms
        year_start: Start year for date filter
        year_end: End year for date filter
        categories: List of category filters
        open_access_only: Only return open access documents
    
    Example:
        >>> query = SearchQuery(
        ...     terms=["climate adaptation", "flood mitigation"],
        ...     year_start=2020,
        ...     year_end=2024,
        ...     open_access_only=True
        ... )
    """
    terms: List[str] = field(default_factory=list)
    year_start: Optional[int] = None
    year_end: Optional[int] = None
    categories: List[str] = field(default_factory=list)
    open_access_only: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_string(self) -> str:
        """Convert to simple search string."""
        return " ".join(self.terms)
    
    @classmethod
    def from_string(cls, query: str) -> "SearchQuery":
        """Create from simple string."""
        return cls(terms=[query])


__all__ = [
    "Document",
    "BaseSource",
    "SearchQuery",
]
