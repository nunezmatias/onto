"""
Corpus Management Module

This module provides the Corpus class for managing collections of documents
from multiple academic sources.

Features:
- Multi-source search and aggregation
- Automatic deduplication
- Export to JSON/CSV
- Corpus statistics and analysis

Example:
    >>> from ontorag.corpus import Corpus
    >>> 
    >>> # Create corpus and search multiple sources
    >>> corpus = Corpus()
    >>> corpus.search_all(
    ...     query="climate adaptation flood mitigation",
    ...     sources=["arxiv", "semantic_scholar", "openalex"],
    ...     max_per_source=50
    ... )
    >>> 
    >>> # View statistics
    >>> print(corpus.stats())
    >>> 
    >>> # Export
    >>> corpus.save("my_corpus.json")
"""

from __future__ import annotations

import json
import csv
import logging
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Union

from .base import Document, BaseSource
from .sources import AVAILABLE_SOURCES, get_source

logger = logging.getLogger(__name__)


class Corpus:
    """
    Manage a collection of documents from multiple sources.
    
    The Corpus class provides:
    - Multi-source search with automatic deduplication
    - Document storage and retrieval
    - Export to JSON/CSV formats
    - Basic statistics and filtering
    
    Example:
        >>> corpus = Corpus(name="climate_research")
        >>> 
        >>> # Search specific sources
        >>> corpus.search("flood adaptation", sources=["arxiv"])
        >>> corpus.search("drought mitigation", sources=["semantic_scholar"])
        >>> 
        >>> # Or search all at once
        >>> corpus.search_all("climate resilience", max_per_source=100)
        >>> 
        >>> # Filter and analyze
        >>> recent = corpus.filter(year_min=2020)
        >>> print(f"Documents since 2020: {len(recent)}")
        >>> 
        >>> # Export
        >>> corpus.save("corpus.json")
        >>> corpus.to_csv("corpus.csv")
    
    Attributes:
        name: Corpus name/identifier
        documents: Dictionary of documents by ID
        metadata: Additional corpus metadata
    """
    
    def __init__(
        self,
        name: str = "corpus",
        description: str = "",
    ):
        """
        Initialize a new corpus.
        
        Args:
            name: Name for this corpus
            description: Description of the corpus
        """
        self.name = name
        self.description = description
        self._documents: Dict[str, Document] = {}
        self._sources_used: Set[str] = set()
        self.metadata: Dict[str, Any] = {
            "created_at": datetime.now().isoformat(),
            "queries": [],
        }
    
    # =========================================================================
    # PROPERTIES
    # =========================================================================
    
    @property
    def documents(self) -> List[Document]:
        """Get all documents as a list."""
        return list(self._documents.values())
    
    def __len__(self) -> int:
        """Number of documents."""
        return len(self._documents)
    
    def __iter__(self):
        """Iterate over documents."""
        return iter(self._documents.values())
    
    def __getitem__(self, doc_id: str) -> Optional[Document]:
        """Get document by ID."""
        return self._documents.get(doc_id)
    
    # =========================================================================
    # SEARCH METHODS
    # =========================================================================
    
    def search(
        self,
        query: str,
        sources: List[str] = None,
        max_results: int = 100,
        **kwargs
    ) -> int:
        """
        Search for documents and add to corpus.
        
        Args:
            query: Search query
            sources: List of source names (default: ["arxiv"])
            max_results: Maximum results per source
            **kwargs: Additional source-specific parameters
            
        Returns:
            Number of new documents added
        """
        sources = sources or ["arxiv"]
        added = 0
        
        for source_name in sources:
            try:
                source = get_source(source_name)
                docs = source.search(query, max_results=max_results, **kwargs)
                
                for doc in docs:
                    if self.add(doc):
                        added += 1
                
                self._sources_used.add(source_name)
                
            except Exception as e:
                logger.error(f"Error searching {source_name}: {e}")
        
        # Record query
        self.metadata["queries"].append({
            "query": query,
            "sources": sources,
            "max_results": max_results,
            "timestamp": datetime.now().isoformat(),
            "added": added,
        })
        
        logger.info(f"Search complete: Added {added} new documents")
        return added
    
    def search_all(
        self,
        query: str,
        sources: List[str] = None,
        max_per_source: int = 100,
        **kwargs
    ) -> int:
        """
        Search all available sources.
        
        Args:
            query: Search query
            sources: Sources to use (default: all available)
            max_per_source: Maximum results from each source
            **kwargs: Additional parameters
            
        Returns:
            Total number of new documents added
        """
        sources = sources or list(AVAILABLE_SOURCES.keys())
        return self.search(query, sources=sources, max_results=max_per_source, **kwargs)
    
    # =========================================================================
    # DOCUMENT MANAGEMENT
    # =========================================================================
    
    def add(self, document: Document) -> bool:
        """
        Add a document to the corpus.
        
        Handles deduplication by checking:
        1. Document ID
        2. DOI (if available)
        3. Title similarity
        
        Args:
            document: Document to add
            
        Returns:
            True if document was added (not duplicate)
        """
        # Check by ID
        if document.id in self._documents:
            return False
        
        # Check by DOI
        if document.doi:
            for existing in self._documents.values():
                if existing.doi and existing.doi == document.doi:
                    return False
        
        # Check by title (basic similarity)
        title_normalized = document.title.lower().strip()
        for existing in self._documents.values():
            existing_title = existing.title.lower().strip()
            if title_normalized == existing_title:
                return False
        
        self._documents[document.id] = document
        return True
    
    def remove(self, doc_id: str) -> bool:
        """
        Remove a document from the corpus.
        
        Args:
            doc_id: Document ID to remove
            
        Returns:
            True if document was removed
        """
        if doc_id in self._documents:
            del self._documents[doc_id]
            return True
        return False
    
    def get(self, doc_id: str) -> Optional[Document]:
        """Get document by ID."""
        return self._documents.get(doc_id)
    
    # =========================================================================
    # FILTERING
    # =========================================================================
    
    def filter(
        self,
        year_min: int = None,
        year_max: int = None,
        sources: List[str] = None,
        has_abstract: bool = None,
        has_pdf: bool = None,
        min_citations: int = None,
    ) -> List[Document]:
        """
        Filter documents by criteria.
        
        Args:
            year_min: Minimum publication year
            year_max: Maximum publication year
            sources: Only from these sources
            has_abstract: Must have abstract
            has_pdf: Must have PDF URL
            min_citations: Minimum citation count
            
        Returns:
            List of matching documents
        """
        results = []
        
        for doc in self._documents.values():
            # Year filters
            if year_min and doc.year and doc.year < year_min:
                continue
            if year_max and doc.year and doc.year > year_max:
                continue
            
            # Source filter
            if sources and doc.source not in sources:
                continue
            
            # Content filters
            if has_abstract and not doc.abstract:
                continue
            if has_pdf and not doc.pdf_url:
                continue
            
            # Citation filter
            if min_citations and doc.citation_count < min_citations:
                continue
            
            results.append(doc)
        
        return results
    
    # =========================================================================
    # STATISTICS
    # =========================================================================
    
    def stats(self) -> Dict[str, Any]:
        """
        Get corpus statistics.
        
        Returns:
            Dictionary with:
            - total_documents: Total count
            - by_source: Count per source
            - by_year: Count per year
            - with_abstract: Count with abstracts
            - with_pdf: Count with PDF links
            - year_range: (min_year, max_year)
            - avg_citations: Average citation count
        """
        if not self._documents:
            return {"total_documents": 0}
        
        by_source = Counter(doc.source for doc in self._documents.values())
        by_year = Counter(doc.year for doc in self._documents.values() if doc.year)
        
        years = [doc.year for doc in self._documents.values() if doc.year]
        citations = [doc.citation_count for doc in self._documents.values()]
        
        return {
            "total_documents": len(self._documents),
            "by_source": dict(by_source),
            "by_year": dict(sorted(by_year.items())),
            "with_abstract": sum(1 for d in self._documents.values() if d.abstract),
            "with_pdf": sum(1 for d in self._documents.values() if d.pdf_url),
            "year_range": (min(years), max(years)) if years else None,
            "avg_citations": sum(citations) / len(citations) if citations else 0,
            "sources_used": list(self._sources_used),
        }
    
    def summary(self) -> str:
        """Generate a text summary of the corpus."""
        stats = self.stats()
        
        lines = [
            f"Corpus: {self.name}",
            f"Description: {self.description}",
            "=" * 50,
            f"Total Documents: {stats['total_documents']}",
            "",
            "By Source:",
        ]
        
        for source, count in stats.get("by_source", {}).items():
            lines.append(f"  - {source}: {count}")
        
        if stats.get("year_range"):
            lines.append(f"\nYear Range: {stats['year_range'][0]} - {stats['year_range'][1]}")
        
        lines.append(f"With Abstract: {stats.get('with_abstract', 0)}")
        lines.append(f"With PDF: {stats.get('with_pdf', 0)}")
        lines.append(f"Avg Citations: {stats.get('avg_citations', 0):.1f}")
        
        return "\n".join(lines)
    
    # =========================================================================
    # SERIALIZATION
    # =========================================================================
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert corpus to dictionary."""
        return {
            "name": self.name,
            "description": self.description,
            "metadata": self.metadata,
            "documents": [doc.to_dict() for doc in self._documents.values()]
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Corpus":
        """Create corpus from dictionary."""
        corpus = cls(
            name=data.get("name", "corpus"),
            description=data.get("description", ""),
        )
        corpus.metadata = data.get("metadata", {})
        
        for doc_data in data.get("documents", []):
            doc = Document.from_dict(doc_data)
            corpus._documents[doc.id] = doc
            corpus._sources_used.add(doc.source)
        
        return corpus
    
    def save(self, path: Union[str, Path]) -> None:
        """
        Save corpus to JSON file.
        
        Args:
            path: Output file path
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)
        
        logger.info(f"Corpus saved to {path}")
    
    @classmethod
    def load(cls, path: Union[str, Path]) -> "Corpus":
        """
        Load corpus from JSON file.
        
        Args:
            path: Path to JSON file
            
        Returns:
            Loaded Corpus instance
        """
        with open(path, 'r', encoding='utf-8') as f:
            return cls.from_dict(json.load(f))
    
    def to_csv(self, path: Union[str, Path]) -> None:
        """
        Export corpus to CSV file.
        
        Args:
            path: Output CSV file path
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        if not self._documents:
            logger.warning("Corpus is empty, nothing to export")
            return
        
        fieldnames = [
            "id", "source", "title", "abstract", "authors_str",
            "year", "doi", "url", "pdf_url", "citation_count", "categories"
        ]
        
        with open(path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            
            for doc in self._documents.values():
                row = {
                    "id": doc.id,
                    "source": doc.source,
                    "title": doc.title,
                    "abstract": doc.abstract[:500] if doc.abstract else "",
                    "authors_str": doc.authors_str,
                    "year": doc.year,
                    "doi": doc.doi,
                    "url": doc.url,
                    "pdf_url": doc.pdf_url,
                    "citation_count": doc.citation_count,
                    "categories": "; ".join(doc.categories[:5]),
                }
                writer.writerow(row)
        
        logger.info(f"Corpus exported to {path}")
    
    def to_dataframe(self):
        """
        Convert corpus to pandas DataFrame.
        
        Returns:
            pandas DataFrame with document data
            
        Raises:
            ImportError: If pandas is not installed
        """
        try:
            import pandas as pd
        except ImportError:
            raise ImportError("pandas required for to_dataframe(). Install with: pip install pandas")
        
        data = [doc.to_dict() for doc in self._documents.values()]
        return pd.DataFrame(data)
    
    # =========================================================================
    # STRING REPRESENTATION
    # =========================================================================
    
    def __repr__(self) -> str:
        return f"Corpus('{self.name}', documents={len(self._documents)})"


__all__ = ["Corpus"]
