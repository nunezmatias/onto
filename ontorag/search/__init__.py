"""
OntoRAG Search Module

This module provides search functionality for knowledge graphs.

Key Components:
    - HybridSearcher: Combines semantic and structural search
    - SemanticSearcher: Vector similarity search
    - StructuralSearcher: Graph-based search

The search module is designed to work with KnowledgeGraph but can also
be used independently with custom backends.

Example:
    >>> from ontorag.graph import KnowledgeGraph
    >>> from ontorag.search import HybridSearcher
    >>> 
    >>> graph = KnowledgeGraph.load("my_graph.json")
    >>> searcher = HybridSearcher(graph)
    >>> 
    >>> results = searcher.search("flood mitigation strategies", k=10)
    >>> for r in results:
    ...     print(f"{r.node.label}: {r.score:.2f}")
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from ..graph import KnowledgeGraph, Node

# Re-export SearchResult from graph module for convenience
from ..graph import SearchResult


class SearchSource(Enum):
    """Source of a search result."""
    SEMANTIC = "semantic"
    STRUCTURAL = "structural"
    HYBRID = "hybrid"


class ItemType(Enum):
    """Type of search item."""
    NODE = "node"
    EDGE = "edge"
    TRIPLET = "triplet"


class HybridSearcher:
    """
    Hybrid search combining semantic and structural approaches.
    
    This class provides a unified interface for searching knowledge graphs
    using multiple strategies:
    
    - Semantic: Vector similarity using embeddings
    - Structural: Graph traversal and centrality
    - Hybrid: Weighted combination of both
    
    Example:
        >>> searcher = HybridSearcher(graph, semantic_weight=0.7)
        >>> 
        >>> # Hybrid search (default)
        >>> results = searcher.search("climate adaptation", k=10)
        >>> 
        >>> # Semantic only
        >>> results = searcher.search_semantic("urban cooling", k=5)
        >>> 
        >>> # Structural only
        >>> results = searcher.search_structural("green roof", k=5)
    """
    
    def __init__(
        self,
        graph: "KnowledgeGraph",
        semantic_weight: float = 0.6,
        structural_weight: float = 0.4,
    ):
        """
        Initialize the hybrid searcher.
        
        Args:
            graph: Knowledge graph to search
            semantic_weight: Weight for semantic search (0-1)
            structural_weight: Weight for structural search (0-1)
        """
        self.graph = graph
        self.semantic_weight = semantic_weight
        self.structural_weight = structural_weight
    
    def search(
        self,
        query: str,
        k: int = 10,
        node_types: List[str] = None,
        min_score: float = 0.0,
    ) -> List[SearchResult]:
        """
        Perform hybrid search.
        
        Args:
            query: Search query
            k: Number of results
            node_types: Filter by node types
            min_score: Minimum score threshold
            
        Returns:
            List of SearchResult objects
        """
        results = self.graph.search(
            query=query,
            k=k * 2,  # Get more for filtering
            search_type="hybrid",
            semantic_weight=self.semantic_weight,
        )
        
        # Filter by node types
        if node_types:
            results = [
                r for r in results
                if r.node and r.node.node_type in node_types
            ]
        
        # Filter by minimum score
        results = [r for r in results if r.score >= min_score]
        
        return results[:k]
    
    def search_semantic(
        self,
        query: str,
        k: int = 10,
    ) -> List[SearchResult]:
        """
        Perform semantic-only search.
        
        Args:
            query: Search query
            k: Number of results
            
        Returns:
            List of SearchResult objects
        """
        return self.graph.search(
            query=query,
            k=k,
            search_type="semantic",
        )
    
    def search_structural(
        self,
        query: str,
        k: int = 10,
    ) -> List[SearchResult]:
        """
        Perform structural-only search.
        
        Args:
            query: Search query
            k: Number of results
            
        Returns:
            List of SearchResult objects
        """
        return self.graph.search(
            query=query,
            k=k,
            search_type="structural",
        )
    
    def search_by_type(
        self,
        query: str,
        node_type: str,
        k: int = 10,
    ) -> List[SearchResult]:
        """
        Search for nodes of a specific type.
        
        Args:
            query: Search query
            node_type: Type to filter by
            k: Number of results
            
        Returns:
            List of SearchResult objects
        """
        return self.search(query, k=k, node_types=[node_type])
    
    def get_context(
        self,
        node_id: str,
        depth: int = 1,
    ) -> Dict[str, Any]:
        """
        Get context around a node.
        
        Args:
            node_id: Node to get context for
            depth: Traversal depth
            
        Returns:
            Dictionary with node, neighbors, and edges
        """
        node = self.graph.get_node(node_id)
        if not node:
            return {}
        
        neighbors = self.graph.get_neighbors(node_id)
        outgoing = self.graph.get_edges_from(node_id)
        incoming = self.graph.get_edges_to(node_id)
        
        return {
            "node": node.to_dict(),
            "neighbors": [
                {"relation": rel, "direction": direction, "node": n.to_dict()}
                for rel, direction, n in neighbors
            ],
            "outgoing_edges": [e.to_dict() for e in outgoing],
            "incoming_edges": [e.to_dict() for e in incoming],
        }


class SemanticSearcher:
    """
    Wrapper for semantic-only search.
    
    Provides a simplified interface for vector similarity search.
    
    Example:
        >>> searcher = SemanticSearcher(graph)
        >>> results = searcher.search("climate adaptation", k=5)
    """
    
    def __init__(self, graph: "KnowledgeGraph"):
        """Initialize with a knowledge graph."""
        self._hybrid = HybridSearcher(graph)
    
    def search(self, query: str, k: int = 10) -> List[SearchResult]:
        """Search using semantic similarity."""
        return self._hybrid.search_semantic(query, k)


class StructuralSearcher:
    """
    Wrapper for structural-only search.
    
    Provides a simplified interface for graph-based search.
    
    Example:
        >>> searcher = StructuralSearcher(graph)
        >>> results = searcher.search("flood", k=5)
    """
    
    def __init__(self, graph: "KnowledgeGraph"):
        """Initialize with a knowledge graph."""
        self._hybrid = HybridSearcher(graph)
    
    def search(self, query: str, k: int = 10) -> List[SearchResult]:
        """Search using graph structure."""
        return self._hybrid.search_structural(query, k)


__all__ = [
    "SearchResult",
    "SearchSource",
    "ItemType",
    "HybridSearcher",
    "SemanticSearcher",
    "StructuralSearcher",
]
