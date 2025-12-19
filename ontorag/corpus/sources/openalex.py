"""
OpenAlex Source for Corpus Collection

Search and retrieve papers from OpenAlex (openalex.org).

OpenAlex is an open catalog of the global research system with:
- 250+ million works
- 95+ million authors
- 125,000+ sources (journals, conferences)
- Free, open API with no authentication required

Features:
- Hierarchical concept tagging
- Rich metadata including affiliations
- Open access status tracking
- Citation and reference data

API Documentation: https://docs.openalex.org/

Example:
    >>> from ontorag.corpus.sources import OpenAlexSource
    >>> 
    >>> openalex = OpenAlexSource()
    >>> docs = openalex.search("climate adaptation urban", max_results=100)
    >>> 
    >>> # Filter by concept
    >>> docs = openalex.search(
    ...     "flood mitigation",
    ...     concepts=["Climate Change"],
    ...     from_year=2020
    ... )
"""

from __future__ import annotations

import time
import logging
from typing import Any, Dict, List, Optional

from ..base import BaseSource, Document

logger = logging.getLogger(__name__)

try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False


class OpenAlexSource(BaseSource):
    """
    OpenAlex academic database source.
    
    OpenAlex provides free, open access to academic metadata.
    No API key required, but providing an email in requests
    gets you into the "polite pool" with better rate limits.
    
    Concepts (fields of study):
        OpenAlex uses a hierarchical concept system. Some relevant concepts:
        - "Climate Change" (C14477221)
        - "Environmental Science" (C18903297)
        - "Flood" (C2779438)
        - "Drought" (C2780022)
    
    Example:
        >>> source = OpenAlexSource(email="researcher@university.edu")
        >>> 
        >>> # Basic search
        >>> docs = source.search("green infrastructure flood", max_results=50)
        >>> 
        >>> # With year and concept filters
        >>> docs = source.search(
        ...     "adaptation",
        ...     from_year=2020,
        ...     concepts=["Climate Change"],
        ...     open_access_only=True
        ... )
    """
    
    name = "openalex"
    base_url = "https://api.openalex.org"
    requests_per_minute = 100  # Polite pool
    
    def __init__(self, email: Optional[str] = None, api_key: Optional[str] = None):
        """
        Initialize OpenAlex source.
        
        Args:
            email: Email for polite pool (recommended)
            api_key: Not used by OpenAlex (for interface compatibility)
        """
        super().__init__(api_key)
        self.email = email or "ontorag@example.com"
        
        if not HAS_REQUESTS:
            raise ImportError(
                "requests package not installed. "
                "Install with: pip install requests"
            )
        
        self._session = requests.Session()
        self._session.headers.update({
            "Accept": "application/json",
            "User-Agent": f"OntoRAG/0.2.0 (mailto:{self.email})"
        })
    
    def search(
        self,
        query: str,
        max_results: int = 100,
        from_year: int = None,
        to_year: int = None,
        concepts: List[str] = None,
        open_access_only: bool = False,
        **kwargs
    ) -> List[Document]:
        """
        Search OpenAlex for works.
        
        Args:
            query: Search query
            max_results: Maximum results (default 100, can page up to 10000)
            from_year: Start year filter
            to_year: End year filter
            concepts: List of concept names to filter by
            open_access_only: Only return open access works
            
        Returns:
            List of Document objects
        """
        url = f"{self.base_url}/works"
        
        # Build filter
        filter_parts = [f'default.search:{query}']
        
        if from_year:
            filter_parts.append(f'from_publication_date:{from_year}-01-01')
        
        if to_year:
            filter_parts.append(f'to_publication_date:{to_year}-12-31')
        
        if concepts:
            concept_filter = "|".join(f'concepts.display_name.search:{c}' for c in concepts)
            filter_parts.append(f'({concept_filter})')
        
        if open_access_only:
            filter_parts.append('is_oa:true')
        
        params = {
            "filter": ",".join(filter_parts),
            "per_page": min(max_results, 200),  # API max per page
            "select": "id,doi,title,abstract_inverted_index,authorships,publication_year,cited_by_count,concepts,open_access,primary_location,type",
            "mailto": self.email
        }
        
        documents = []
        cursor = "*"
        
        try:
            while len(documents) < max_results:
                params["cursor"] = cursor
                
                response = self._session.get(url, params=params, timeout=30)
                response.raise_for_status()
                data = response.json()
                
                for work in data.get("results", []):
                    doc = self._convert_work(work)
                    if doc:
                        documents.append(doc)
                        if len(documents) >= max_results:
                            break
                
                # Get next cursor
                meta = data.get("meta", {})
                cursor = meta.get("next_cursor")
                
                if not cursor or not data.get("results"):
                    break
                
                time.sleep(0.5)  # Be polite
                    
        except requests.exceptions.RequestException as e:
            logger.error(f"OpenAlex API error: {e}")
        
        logger.info(f"OpenAlex: Found {len(documents)} works for: {query[:50]}...")
        return documents
    
    def get_by_id(self, doc_id: str) -> Optional[Document]:
        """
        Get work by OpenAlex ID or DOI.
        
        Args:
            doc_id: OpenAlex ID (W123456) or DOI
            
        Returns:
            Document if found, None otherwise
        """
        # Determine ID type
        if doc_id.startswith("W") or doc_id.startswith("https://openalex.org/"):
            work_id = doc_id
        elif doc_id.startswith("10."):
            work_id = f"https://doi.org/{doc_id}"
        else:
            work_id = doc_id
        
        url = f"{self.base_url}/works/{work_id}"
        params = {"mailto": self.email}
        
        try:
            response = self._session.get(url, params=params, timeout=30)
            response.raise_for_status()
            return self._convert_work(response.json())
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Error fetching work {doc_id}: {e}")
        
        return None
    
    def _convert_work(self, work: Dict[str, Any]) -> Optional[Document]:
        """Convert API response to Document."""
        if not work.get("id"):
            return None
        
        # Reconstruct abstract from inverted index
        abstract = ""
        inv_index = work.get("abstract_inverted_index")
        if inv_index:
            try:
                max_pos = max(max(positions) for positions in inv_index.values())
                words = [""] * (max_pos + 1)
                for word, positions in inv_index.items():
                    for pos in positions:
                        words[pos] = word
                abstract = " ".join(words)
            except (ValueError, TypeError):
                pass
        
        # Extract authors
        authors = []
        for authorship in work.get("authorships", []):
            author = authorship.get("author", {})
            if author.get("display_name"):
                authors.append(author["display_name"])
        
        # Extract DOI
        doi = work.get("doi", "")
        if doi:
            doi = doi.replace("https://doi.org/", "")
        
        # Get PDF URL
        pdf_url = None
        primary_location = work.get("primary_location", {}) or {}
        if primary_location.get("pdf_url"):
            pdf_url = primary_location["pdf_url"]
        
        # Extract concepts
        concepts = []
        for concept in work.get("concepts", [])[:10]:
            if concept.get("display_name"):
                concepts.append(concept["display_name"])
        
        # Get OpenAlex ID
        openalex_id = work.get("id", "").replace("https://openalex.org/", "")
        
        return Document(
            source="openalex",
            source_id=openalex_id,
            title=work.get("title", "") or "",
            abstract=abstract,
            authors=authors,
            year=work.get("publication_year"),
            url=work.get("id"),
            doi=doi or None,
            pdf_url=pdf_url,
            citation_count=work.get("cited_by_count", 0) or 0,
            categories=concepts,
            keywords=concepts[:5],
            metadata={
                "type": work.get("type"),
                "is_open_access": work.get("open_access", {}).get("is_oa", False),
                "source_name": primary_location.get("source", {}).get("display_name") if primary_location else None,
            }
        )


__all__ = ["OpenAlexSource"]
