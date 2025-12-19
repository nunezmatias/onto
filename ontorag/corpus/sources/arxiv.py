"""
arXiv Source for Corpus Collection

Search and retrieve papers from arXiv (arxiv.org).

arXiv is an open-access repository for scientific preprints in:
- Physics, Mathematics, Computer Science
- Quantitative Biology, Quantitative Finance
- Statistics, Electrical Engineering, Economics

Features:
- Full-text search in titles, abstracts, authors
- Filter by category, date range
- Access to PDF links
- No API key required

Example:
    >>> from ontorag.corpus.sources import ArxivSource
    >>> 
    >>> arxiv = ArxivSource()
    >>> docs = arxiv.search("climate adaptation flood mitigation", max_results=50)
    >>> 
    >>> for doc in docs:
    ...     print(f"{doc.title} ({doc.year})")
"""

from __future__ import annotations

import time
import logging
from typing import Any, Dict, List, Optional

from ..base import BaseSource, Document

logger = logging.getLogger(__name__)

# Try to import arxiv library
try:
    import arxiv
    HAS_ARXIV = True
except ImportError:
    HAS_ARXIV = False
    arxiv = None


class ArxivSource(BaseSource):
    """
    arXiv paper source.
    
    Uses the arxiv Python package to search arXiv.org.
    
    Install: pip install arxiv
    
    Search Syntax:
        - abs:"term" - Search in abstract
        - ti:"term" - Search in title  
        - au:"name" - Search by author
        - cat:cs.AI - Filter by category
        - AND, OR - Boolean operators
    
    Categories relevant to climate:
        - physics.ao-ph: Atmospheric and Oceanic Physics
        - physics.geo-ph: Geophysics
        - cs.LG: Machine Learning
        - stat.ML: Statistics - Machine Learning
    
    Example:
        >>> source = ArxivSource()
        >>> 
        >>> # Simple search
        >>> docs = source.search("flood prediction", max_results=20)
        >>> 
        >>> # Advanced query
        >>> docs = source.search(
        ...     'abs:"climate change" AND abs:"adaptation"',
        ...     max_results=50,
        ...     sort_by="relevance"
        ... )
    
    Attributes:
        name: "arxiv"
        base_url: "http://export.arxiv.org/api/query"
        requests_per_minute: 30 (arXiv recommended)
    """
    
    name = "arxiv"
    base_url = "http://export.arxiv.org/api/query"
    requests_per_minute = 30
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize arXiv source.
        
        Args:
            api_key: Not required for arXiv (ignored)
        """
        super().__init__(api_key)
        
        if not HAS_ARXIV:
            raise ImportError(
                "arxiv package not installed. "
                "Install with: pip install arxiv"
            )
        
        self._client = arxiv.Client(
            page_size=100,
            delay_seconds=3.0,
            num_retries=3
        )
    
    def search(
        self,
        query: str,
        max_results: int = 100,
        sort_by: str = "relevance",
        sort_order: str = "descending",
        categories: List[str] = None,
        **kwargs
    ) -> List[Document]:
        """
        Search arXiv for papers.
        
        Args:
            query: Search query (supports arXiv query syntax)
            max_results: Maximum number of results (default 100)
            sort_by: Sort criterion - "relevance", "lastUpdatedDate", "submittedDate"
            sort_order: "ascending" or "descending"
            categories: Filter by arXiv categories (e.g., ["cs.AI", "physics.ao-ph"])
            
        Returns:
            List of Document objects
        """
        # Build query with category filter if provided
        full_query = query
        if categories:
            cat_filter = " OR ".join(f"cat:{cat}" for cat in categories)
            full_query = f"({query}) AND ({cat_filter})"
        
        # Map sort options
        sort_criterion = {
            "relevance": arxiv.SortCriterion.Relevance,
            "lastUpdatedDate": arxiv.SortCriterion.LastUpdatedDate,
            "submittedDate": arxiv.SortCriterion.SubmittedDate,
        }.get(sort_by, arxiv.SortCriterion.Relevance)
        
        sort_order_enum = (
            arxiv.SortOrder.Ascending 
            if sort_order == "ascending" 
            else arxiv.SortOrder.Descending
        )
        
        # Create search
        search = arxiv.Search(
            query=full_query,
            max_results=max_results,
            sort_by=sort_criterion,
            sort_order=sort_order_enum
        )
        
        documents = []
        
        try:
            for result in self._client.results(search):
                doc = self._convert_result(result)
                documents.append(doc)
                
        except Exception as e:
            logger.error(f"arXiv search error: {e}")
        
        logger.info(f"arXiv: Found {len(documents)} papers for query: {query[:50]}...")
        return documents
    
    def get_by_id(self, doc_id: str) -> Optional[Document]:
        """
        Get paper by arXiv ID.
        
        Args:
            doc_id: arXiv ID (e.g., "2301.00001" or "2301.00001v1")
            
        Returns:
            Document if found, None otherwise
        """
        try:
            search = arxiv.Search(id_list=[doc_id])
            results = list(self._client.results(search))
            
            if results:
                return self._convert_result(results[0])
            
        except Exception as e:
            logger.error(f"Error fetching arXiv paper {doc_id}: {e}")
        
        return None
    
    def _convert_result(self, result: "arxiv.Result") -> Document:
        """Convert arXiv result to Document."""
        # Extract year from published date
        year = result.published.year if result.published else None
        
        # Get arXiv ID without version
        arxiv_id = result.entry_id.split("/")[-1]
        if "v" in arxiv_id:
            arxiv_id = arxiv_id.split("v")[0]
        
        return Document(
            source="arxiv",
            source_id=arxiv_id,
            title=result.title.replace("\n", " ").strip(),
            abstract=result.summary.replace("\n", " ").strip(),
            authors=[author.name for author in result.authors],
            year=year,
            published_date=result.published.isoformat() if result.published else None,
            url=result.entry_id,
            doi=result.doi,
            pdf_url=result.pdf_url,
            categories=list(result.categories),
            metadata={
                "primary_category": result.primary_category,
                "comment": result.comment,
                "journal_ref": result.journal_ref,
                "updated": result.updated.isoformat() if result.updated else None,
            }
        )
    
    @staticmethod
    def build_query(
        terms: List[str] = None,
        title_terms: List[str] = None,
        abstract_terms: List[str] = None,
        author: str = None,
        combine: str = "AND"
    ) -> str:
        """
        Helper to build arXiv query strings.
        
        Args:
            terms: General search terms
            title_terms: Terms to search in title
            abstract_terms: Terms to search in abstract
            author: Author name
            combine: How to combine terms ("AND" or "OR")
            
        Returns:
            Formatted arXiv query string
            
        Example:
            >>> query = ArxivSource.build_query(
            ...     abstract_terms=["climate change", "adaptation"],
            ...     title_terms=["flood"],
            ...     combine="AND"
            ... )
            >>> print(query)
            '(abs:"climate change" AND abs:"adaptation") AND ti:"flood"'
        """
        parts = []
        
        if terms:
            parts.append(" ".join(terms))
        
        if title_terms:
            ti_parts = [f'ti:"{t}"' for t in title_terms]
            parts.append(f"({f' {combine} '.join(ti_parts)})")
        
        if abstract_terms:
            abs_parts = [f'abs:"{t}"' for t in abstract_terms]
            parts.append(f"({f' {combine} '.join(abs_parts)})")
        
        if author:
            parts.append(f'au:"{author}"')
        
        return f" {combine} ".join(parts)


__all__ = ["ArxivSource"]
