"""
Semantic Scholar Source for Corpus Collection

Search and retrieve papers from Semantic Scholar (semanticscholar.org).

Semantic Scholar is an AI-powered research tool for scientific literature that:
- Indexes papers from multiple sources
- Provides citation analysis and influential citations
- Covers all scientific domains
- Offers free API access

API Limits (without key):
- 100 requests per 5 minutes

Example:
    >>> from ontorag.corpus.sources import SemanticScholarSource
    >>> 
    >>> ss = SemanticScholarSource()
    >>> docs = ss.search("climate adaptation", max_results=50)
    >>> 
    >>> for doc in docs:
    ...     print(f"{doc.title} - Citations: {doc.citation_count}")
"""

from __future__ import annotations

import time
import logging
from typing import Any, Dict, List, Optional, Tuple

from ..base import BaseSource, Document

logger = logging.getLogger(__name__)

try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False


class SemanticScholarSource(BaseSource):
    """
    Semantic Scholar paper source.
    
    Uses the Semantic Scholar Academic Graph API.
    
    Features:
        - Full-text search across titles, abstracts, authors
        - Citation count and influential citation metrics
        - Year range filtering
        - Fields of study filtering
    
    API Documentation: https://api.semanticscholar.org/
    
    Example:
        >>> source = SemanticScholarSource()
        >>> 
        >>> # Basic search
        >>> docs = source.search("flood mitigation", max_results=20)
        >>> 
        >>> # With year filter
        >>> docs = source.search(
        ...     "climate adaptation",
        ...     max_results=50,
        ...     year_range=(2020, 2024)
        ... )
        >>> 
        >>> # Get highly cited papers
        >>> for doc in sorted(docs, key=lambda d: d.citation_count, reverse=True)[:10]:
        ...     print(f"{doc.citation_count}: {doc.title[:60]}")
    """
    
    name = "semantic_scholar"
    base_url = "https://api.semanticscholar.org/graph/v1"
    requests_per_minute = 20  # Conservative to avoid rate limits
    
    # Fields to retrieve from API
    PAPER_FIELDS = [
        "paperId", "title", "abstract", "year", "venue",
        "authors", "citationCount", "influentialCitationCount",
        "publicationDate", "externalIds", "url", "openAccessPdf",
        "fieldsOfStudy"
    ]
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize Semantic Scholar source.
        
        Args:
            api_key: Optional API key for higher rate limits
        """
        super().__init__(api_key)
        
        if not HAS_REQUESTS:
            raise ImportError(
                "requests package not installed. "
                "Install with: pip install requests"
            )
        
        self._session = requests.Session()
        self._session.headers.update({
            "Accept": "application/json",
        })
        
        if api_key:
            self._session.headers["x-api-key"] = api_key
    
    def search(
        self,
        query: str,
        max_results: int = 100,
        year_range: Tuple[int, int] = None,
        fields_of_study: List[str] = None,
        open_access_only: bool = False,
        **kwargs
    ) -> List[Document]:
        """
        Search Semantic Scholar for papers.
        
        Args:
            query: Search query
            max_results: Maximum results (default 100, API max per request)
            year_range: Tuple of (start_year, end_year)
            fields_of_study: List of fields (e.g., ["Computer Science", "Environmental Science"])
            open_access_only: Only return open access papers
            
        Returns:
            List of Document objects
        """
        url = f"{self.base_url}/paper/search"
        
        params = {
            "query": query,
            "limit": min(max_results, 100),
            "fields": ",".join(self.PAPER_FIELDS)
        }
        
        if year_range:
            params["year"] = f"{year_range[0]}-{year_range[1]}"
        
        if fields_of_study:
            params["fieldsOfStudy"] = ",".join(fields_of_study)
        
        if open_access_only:
            params["openAccessPdf"] = ""
        
        documents = []
        
        try:
            response = self._session.get(url, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()
            
            for paper in data.get("data", []):
                doc = self._convert_paper(paper)
                if doc:
                    documents.append(doc)
            
            # Handle pagination if needed
            total = data.get("total", 0)
            offset = len(documents)
            
            while offset < min(max_results, total) and offset < 1000:  # API limit
                time.sleep(3)  # Rate limiting
                params["offset"] = offset
                
                response = self._session.get(url, params=params, timeout=30)
                response.raise_for_status()
                data = response.json()
                
                for paper in data.get("data", []):
                    doc = self._convert_paper(paper)
                    if doc:
                        documents.append(doc)
                
                offset = len(documents)
                if not data.get("data"):
                    break
                    
        except requests.exceptions.RequestException as e:
            logger.error(f"Semantic Scholar API error: {e}")
        
        logger.info(f"Semantic Scholar: Found {len(documents)} papers for: {query[:50]}...")
        return documents
    
    def get_by_id(
        self,
        doc_id: str,
        id_type: str = "paperId"
    ) -> Optional[Document]:
        """
        Get paper by ID.
        
        Args:
            doc_id: Paper ID
            id_type: Type of ID - "paperId", "DOI", "ArXiv", "PMID"
            
        Returns:
            Document if found, None otherwise
        """
        # Format ID based on type
        if id_type == "DOI":
            paper_id = f"DOI:{doc_id}"
        elif id_type == "ArXiv":
            paper_id = f"ArXiv:{doc_id}"
        elif id_type == "PMID":
            paper_id = f"PMID:{doc_id}"
        else:
            paper_id = doc_id
        
        url = f"{self.base_url}/paper/{paper_id}"
        params = {"fields": ",".join(self.PAPER_FIELDS)}
        
        try:
            response = self._session.get(url, params=params, timeout=30)
            response.raise_for_status()
            return self._convert_paper(response.json())
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Error fetching paper {doc_id}: {e}")
        
        return None
    
    def _convert_paper(self, paper: Dict[str, Any]) -> Optional[Document]:
        """Convert API response to Document."""
        if not paper.get("paperId"):
            return None
        
        # Extract authors
        authors = []
        for author in paper.get("authors", []):
            if author.get("name"):
                authors.append(author["name"])
        
        # Extract DOI
        external_ids = paper.get("externalIds", {}) or {}
        doi = external_ids.get("DOI")
        arxiv_id = external_ids.get("ArXiv")
        
        # Get PDF URL
        pdf_url = None
        if paper.get("openAccessPdf"):
            pdf_url = paper["openAccessPdf"].get("url")
        
        return Document(
            source="semantic_scholar",
            source_id=paper["paperId"],
            title=paper.get("title", ""),
            abstract=paper.get("abstract", "") or "",
            authors=authors,
            year=paper.get("year"),
            published_date=paper.get("publicationDate"),
            url=paper.get("url"),
            doi=doi,
            pdf_url=pdf_url,
            citation_count=paper.get("citationCount", 0) or 0,
            categories=paper.get("fieldsOfStudy", []) or [],
            metadata={
                "venue": paper.get("venue"),
                "influential_citation_count": paper.get("influentialCitationCount", 0),
                "arxiv_id": arxiv_id,
            }
        )


__all__ = ["SemanticScholarSource"]
