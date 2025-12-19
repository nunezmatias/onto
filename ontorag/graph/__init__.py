"""
OntoRAG Graph Module

This module provides the KnowledgeGraph class for building and querying
ontology-aware knowledge graphs.

Key Components:
    - Node: Graph node with type and properties
    - Edge: Graph edge with relation type
    - KnowledgeGraph: Main graph class with NetworkX + ChromaDB backend
    - GraphBuilder: Helper for building graphs from triplets

Features:
    - Ontology-aware node/edge validation
    - Semantic search with embeddings
    - Structural graph traversal
    - Hybrid search combining both approaches
    - Import/export to various formats

Example:
    >>> from ontorag.graph import KnowledgeGraph
    >>> from ontorag.ontology import create_climate_ontology
    >>> 
    >>> # Create graph with ontology
    >>> ontology = create_climate_ontology()
    >>> graph = KnowledgeGraph(ontology=ontology)
    >>> 
    >>> # Add nodes and edges
    >>> graph.add_node("green_roof", node_type="green_infrastructure")
    >>> graph.add_node("urban_heat", node_type="heat_wave")
    >>> graph.add_edge("green_roof", "MITIGATES", "urban_heat")
    >>> 
    >>> # Search
    >>> results = graph.search("cooling solutions for cities", k=5)
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union, TYPE_CHECKING

if TYPE_CHECKING:
    from ..ontology import Ontology

logger = logging.getLogger(__name__)

# Try to import optional dependencies
try:
    import networkx as nx
    HAS_NETWORKX = True
except ImportError:
    HAS_NETWORKX = False
    nx = None

try:
    import chromadb
    from chromadb.config import Settings
    HAS_CHROMADB = True
except ImportError:
    HAS_CHROMADB = False
    chromadb = None


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class Node:
    """
    A node in the knowledge graph.
    
    Attributes:
        id: Unique identifier
        node_type: Ontology class type
        label: Display label
        properties: Additional properties
        embedding: Vector embedding (computed automatically)
    """
    id: str
    node_type: str = "entity"
    label: str = ""
    properties: Dict[str, Any] = field(default_factory=dict)
    embedding: Optional[List[float]] = None
    
    def __post_init__(self):
        if not self.label:
            self.label = self.id.replace("_", " ").title()
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "type": self.node_type,
            "label": self.label,
            "properties": self.properties,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Node":
        return cls(
            id=data.get("id", ""),
            node_type=data.get("type", "entity"),
            label=data.get("label", ""),
            properties=data.get("properties", {}),
        )


@dataclass
class Edge:
    """
    An edge in the knowledge graph.
    
    Attributes:
        source: Source node ID
        target: Target node ID
        relation: Relation type
        properties: Additional properties
        weight: Edge weight
    """
    source: str
    target: str
    relation: str
    properties: Dict[str, Any] = field(default_factory=dict)
    weight: float = 1.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "source": self.source,
            "target": self.target,
            "relation": self.relation,
            "properties": self.properties,
            "weight": self.weight,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Edge":
        return cls(
            source=data.get("source", ""),
            target=data.get("target", ""),
            relation=data.get("relation", ""),
            properties=data.get("properties", {}),
            weight=data.get("weight", 1.0),
        )
    
    def to_tuple(self) -> Tuple[str, str, str]:
        return (self.source, self.relation, self.target)


@dataclass
class SearchResult:
    """
    A search result from the knowledge graph.
    
    Attributes:
        node_id: ID of the matching node
        node: The Node object
        score: Relevance score (0-1)
        source: How it was found (semantic, structural, hybrid)
        context: Related nodes and edges
    """
    node_id: str
    node: Optional[Node] = None
    score: float = 0.0
    source: str = "semantic"
    context: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "node_id": self.node_id,
            "node": self.node.to_dict() if self.node else None,
            "score": self.score,
            "source": self.source,
            "context": self.context,
        }


# =============================================================================
# KNOWLEDGE GRAPH
# =============================================================================

class KnowledgeGraph:
    """
    Ontology-aware knowledge graph with hybrid search.
    
    Combines NetworkX for graph structure and ChromaDB for semantic search.
    
    Features:
        - Add nodes and edges with ontology validation
        - Semantic search using embeddings
        - Structural traversal (BFS, paths, neighbors)
        - Hybrid search combining semantic + structural
        - Export to JSON, GraphML, etc.
    
    Example:
        >>> from ontorag.graph import KnowledgeGraph
        >>> from ontorag.ontology import create_climate_ontology
        >>> 
        >>> # Create with ontology
        >>> onto = create_climate_ontology()
        >>> kg = KnowledgeGraph(ontology=onto)
        >>> 
        >>> # Add knowledge
        >>> kg.add_node("green_roof", node_type="green_infrastructure", 
        ...             label="Green Roof", properties={"cooling": "2-3°C"})
        >>> kg.add_node("urban_heat", node_type="heat_wave")
        >>> kg.add_edge("green_roof", "MITIGATES", "urban_heat")
        >>> 
        >>> # Query
        >>> results = kg.search("cooling urban areas", k=5)
        >>> neighbors = kg.get_neighbors("green_roof")
    """
    
    def __init__(
        self,
        ontology: "Ontology" = None,
        embedding_model: str = "all-MiniLM-L6-v2",
        persist_directory: str = None,
        collection_name: str = "ontorag_nodes",
    ):
        """
        Initialize the knowledge graph.
        
        Args:
            ontology: Ontology for validation (optional)
            embedding_model: Sentence transformer model name
            persist_directory: Directory to persist ChromaDB
            collection_name: Name of the ChromaDB collection
        """
        if not HAS_NETWORKX:
            raise ImportError("networkx required. Install: pip install networkx")
        
        self.ontology = ontology
        self.embedding_model = embedding_model
        
        # Initialize NetworkX graph
        self._graph = nx.DiGraph()
        
        # Node storage
        self._nodes: Dict[str, Node] = {}
        self._edges: Dict[Tuple[str, str, str], Edge] = {}
        
        # Initialize ChromaDB for semantic search
        self._collection = None
        self._embedder = None
        
        if HAS_CHROMADB:
            try:
                if persist_directory:
                    self._chroma_client = chromadb.PersistentClient(path=persist_directory)
                else:
                    self._chroma_client = chromadb.Client()
                
                self._collection = self._chroma_client.get_or_create_collection(
                    name=collection_name,
                    metadata={"hnsw:space": "cosine"}
                )
            except Exception as e:
                logger.warning(f"Could not initialize ChromaDB: {e}")
        
        # Initialize embedder
        self._init_embedder()
    
    def _init_embedder(self) -> None:
        """Initialize sentence transformer for embeddings."""
        try:
            from sentence_transformers import SentenceTransformer
            self._embedder = SentenceTransformer(self.embedding_model)
        except ImportError:
            logger.warning("sentence-transformers not installed. Semantic search disabled.")
    
    # =========================================================================
    # PROPERTIES
    # =========================================================================
    
    @property
    def num_nodes(self) -> int:
        """Number of nodes."""
        return len(self._nodes)
    
    @property
    def num_edges(self) -> int:
        """Number of edges."""
        return len(self._edges)
    
    @property
    def nodes(self) -> List[Node]:
        """Get all nodes."""
        return list(self._nodes.values())
    
    @property
    def edges(self) -> List[Edge]:
        """Get all edges."""
        return list(self._edges.values())
    
    # =========================================================================
    # NODE OPERATIONS
    # =========================================================================
    
    def add_node(
        self,
        node_id: str,
        node_type: str = "entity",
        label: str = None,
        properties: Dict[str, Any] = None,
        validate: bool = True,
    ) -> Node:
        """
        Add a node to the graph.
        
        Args:
            node_id: Unique identifier
            node_type: Ontology class type
            label: Display label
            properties: Additional properties
            validate: Whether to validate against ontology
            
        Returns:
            The created Node
        """
        # Normalize ID
        node_id = node_id.lower().replace(" ", "_").replace("-", "_")
        
        # Validate type against ontology
        if validate and self.ontology:
            normalized_type = self.ontology.normalize_entity(node_type)
            if normalized_type:
                node_type = normalized_type
        
        # Create node
        node = Node(
            id=node_id,
            node_type=node_type,
            label=label or node_id.replace("_", " ").title(),
            properties=properties or {},
        )
        
        # Add to storage
        self._nodes[node_id] = node
        self._graph.add_node(node_id, **node.to_dict())
        
        # Add to vector store
        if self._collection and self._embedder:
            try:
                text = f"{node.label} {node_type} {' '.join(str(v) for v in (properties or {}).values())}"
                embedding = self._embedder.encode(text).tolist()
                
                self._collection.upsert(
                    ids=[node_id],
                    embeddings=[embedding],
                    metadatas=[{"type": node_type, "label": node.label}],
                    documents=[text],
                )
            except Exception as e:
                logger.warning(f"Could not add node to vector store: {e}")
        
        return node
    
    def get_node(self, node_id: str) -> Optional[Node]:
        """Get a node by ID."""
        return self._nodes.get(node_id)
    
    def has_node(self, node_id: str) -> bool:
        """Check if node exists."""
        return node_id in self._nodes
    
    def remove_node(self, node_id: str) -> bool:
        """Remove a node and its edges."""
        if node_id not in self._nodes:
            return False
        
        # Remove edges
        edges_to_remove = [
            key for key in self._edges
            if key[0] == node_id or key[2] == node_id
        ]
        for key in edges_to_remove:
            del self._edges[key]
        
        # Remove from storage
        del self._nodes[node_id]
        self._graph.remove_node(node_id)
        
        # Remove from vector store
        if self._collection:
            try:
                self._collection.delete(ids=[node_id])
            except Exception:
                pass
        
        return True
    
    # =========================================================================
    # EDGE OPERATIONS
    # =========================================================================
    
    def add_edge(
        self,
        source: str,
        relation: str,
        target: str,
        properties: Dict[str, Any] = None,
        weight: float = 1.0,
        validate: bool = True,
        create_nodes: bool = True,
    ) -> Edge:
        """
        Add an edge to the graph.
        
        Args:
            source: Source node ID
            relation: Relation type
            target: Target node ID
            properties: Additional properties
            weight: Edge weight
            validate: Whether to validate against ontology
            create_nodes: Create nodes if they don't exist
            
        Returns:
            The created Edge
        """
        # Normalize IDs
        source = source.lower().replace(" ", "_").replace("-", "_")
        target = target.lower().replace(" ", "_").replace("-", "_")
        relation = relation.upper().replace(" ", "_")
        
        # Create nodes if needed
        if create_nodes:
            if source not in self._nodes:
                self.add_node(source, validate=validate)
            if target not in self._nodes:
                self.add_node(target, validate=validate)
        
        # Validate relation against ontology
        if validate and self.ontology:
            normalized_rel = self.ontology.normalize_relation(relation)
            if normalized_rel:
                relation = normalized_rel
        
        # Create edge
        edge = Edge(
            source=source,
            target=target,
            relation=relation,
            properties=properties or {},
            weight=weight,
        )
        
        # Add to storage
        key = (source, relation, target)
        self._edges[key] = edge
        self._graph.add_edge(source, target, relation=relation, weight=weight)
        
        return edge
    
    def get_edge(self, source: str, relation: str, target: str) -> Optional[Edge]:
        """Get an edge by source, relation, target."""
        key = (source, relation, target)
        return self._edges.get(key)
    
    def has_edge(self, source: str, relation: str, target: str) -> bool:
        """Check if edge exists."""
        return (source, relation, target) in self._edges
    
    def get_edges_from(self, node_id: str) -> List[Edge]:
        """Get all outgoing edges from a node."""
        return [e for e in self._edges.values() if e.source == node_id]
    
    def get_edges_to(self, node_id: str) -> List[Edge]:
        """Get all incoming edges to a node."""
        return [e for e in self._edges.values() if e.target == node_id]
    
    # =========================================================================
    # GRAPH TRAVERSAL
    # =========================================================================
    
    def get_neighbors(
        self,
        node_id: str,
        direction: str = "both",
        relation: str = None,
    ) -> List[Tuple[str, str, Node]]:
        """
        Get neighboring nodes.
        
        Args:
            node_id: Node to get neighbors of
            direction: "out", "in", or "both"
            relation: Filter by relation type
            
        Returns:
            List of (relation, direction, node) tuples
        """
        neighbors = []
        
        if direction in ["out", "both"]:
            for edge in self.get_edges_from(node_id):
                if relation and edge.relation != relation:
                    continue
                neighbor = self.get_node(edge.target)
                if neighbor:
                    neighbors.append((edge.relation, "out", neighbor))
        
        if direction in ["in", "both"]:
            for edge in self.get_edges_to(node_id):
                if relation and edge.relation != relation:
                    continue
                neighbor = self.get_node(edge.source)
                if neighbor:
                    neighbors.append((edge.relation, "in", neighbor))
        
        return neighbors
    
    def get_path(
        self,
        source: str,
        target: str,
        max_length: int = 5,
    ) -> Optional[List[str]]:
        """
        Find shortest path between nodes.
        
        Args:
            source: Source node ID
            target: Target node ID
            max_length: Maximum path length
            
        Returns:
            List of node IDs in path, or None if no path
        """
        try:
            path = nx.shortest_path(self._graph, source, target)
            if len(path) <= max_length + 1:
                return path
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            pass
        return None
    
    def get_subgraph(
        self,
        node_id: str,
        depth: int = 2,
    ) -> "KnowledgeGraph":
        """
        Get a subgraph around a node.
        
        Args:
            node_id: Center node
            depth: Traversal depth
            
        Returns:
            New KnowledgeGraph with subgraph
        """
        # BFS to find nodes within depth
        visited = {node_id}
        frontier = {node_id}
        
        for _ in range(depth):
            new_frontier = set()
            for n in frontier:
                for _, _, neighbor in self.get_neighbors(n):
                    if neighbor.id not in visited:
                        visited.add(neighbor.id)
                        new_frontier.add(neighbor.id)
            frontier = new_frontier
        
        # Create subgraph
        subgraph = KnowledgeGraph(ontology=self.ontology)
        
        for nid in visited:
            node = self.get_node(nid)
            if node:
                subgraph.add_node(
                    nid,
                    node_type=node.node_type,
                    label=node.label,
                    properties=node.properties,
                    validate=False,
                )
        
        for edge in self._edges.values():
            if edge.source in visited and edge.target in visited:
                subgraph.add_edge(
                    edge.source,
                    edge.relation,
                    edge.target,
                    properties=edge.properties,
                    weight=edge.weight,
                    validate=False,
                    create_nodes=False,
                )
        
        return subgraph
    
    # =========================================================================
    # SEARCH
    # =========================================================================
    
    def search(
        self,
        query: str,
        k: int = 10,
        search_type: str = "hybrid",
        semantic_weight: float = 0.6,
    ) -> List[SearchResult]:
        """
        Search the knowledge graph.
        
        Args:
            query: Search query
            k: Number of results
            search_type: "semantic", "structural", or "hybrid"
            semantic_weight: Weight for semantic vs structural (hybrid only)
            
        Returns:
            List of SearchResult objects
        """
        if search_type == "semantic":
            return self._semantic_search(query, k)
        elif search_type == "structural":
            return self._structural_search(query, k)
        else:
            return self._hybrid_search(query, k, semantic_weight)
    
    def _semantic_search(self, query: str, k: int) -> List[SearchResult]:
        """Search using embeddings."""
        if not self._collection or not self._embedder:
            logger.warning("Semantic search not available")
            return []
        
        try:
            results = self._collection.query(
                query_texts=[query],
                n_results=min(k, self.num_nodes),
            )
            
            search_results = []
            for i, node_id in enumerate(results["ids"][0]):
                node = self.get_node(node_id)
                distance = results["distances"][0][i] if results.get("distances") else 0
                score = 1 - distance  # Convert distance to similarity
                
                search_results.append(SearchResult(
                    node_id=node_id,
                    node=node,
                    score=score,
                    source="semantic",
                ))
            
            return search_results
            
        except Exception as e:
            logger.error(f"Semantic search failed: {e}")
            return []
    
    def _structural_search(self, query: str, k: int) -> List[SearchResult]:
        """Search using graph structure (keyword match + centrality)."""
        query_terms = set(query.lower().split())
        results = []
        
        # Compute centrality scores
        try:
            centrality = nx.pagerank(self._graph)
        except Exception:
            centrality = {n: 1.0 for n in self._nodes}
        
        for node_id, node in self._nodes.items():
            # Keyword match score
            node_terms = set(f"{node.label} {node.node_type}".lower().split())
            match_score = len(query_terms & node_terms) / max(len(query_terms), 1)
            
            # Combine with centrality
            score = 0.7 * match_score + 0.3 * centrality.get(node_id, 0)
            
            if score > 0:
                results.append(SearchResult(
                    node_id=node_id,
                    node=node,
                    score=score,
                    source="structural",
                ))
        
        # Sort by score and return top k
        results.sort(key=lambda x: x.score, reverse=True)
        return results[:k]
    
    def _hybrid_search(
        self,
        query: str,
        k: int,
        semantic_weight: float,
    ) -> List[SearchResult]:
        """Combine semantic and structural search."""
        semantic_results = self._semantic_search(query, k * 2)
        structural_results = self._structural_search(query, k * 2)
        
        # Combine scores
        scores = {}
        
        for r in semantic_results:
            scores[r.node_id] = semantic_weight * r.score
        
        for r in structural_results:
            current = scores.get(r.node_id, 0)
            scores[r.node_id] = current + (1 - semantic_weight) * r.score
        
        # Build results
        results = []
        for node_id, score in scores.items():
            node = self.get_node(node_id)
            results.append(SearchResult(
                node_id=node_id,
                node=node,
                score=score,
                source="hybrid",
            ))
        
        results.sort(key=lambda x: x.score, reverse=True)
        return results[:k]
    
    # =========================================================================
    # BULK OPERATIONS
    # =========================================================================
    
    def add_triplets(
        self,
        triplets: List[Tuple[str, str, str]],
        validate: bool = True,
    ) -> int:
        """
        Add multiple triplets at once.
        
        Args:
            triplets: List of (subject, relation, object) tuples
            validate: Whether to validate against ontology
            
        Returns:
            Number of triplets added
        """
        count = 0
        for subj, rel, obj in triplets:
            try:
                self.add_edge(subj, rel, obj, validate=validate)
                count += 1
            except Exception as e:
                logger.warning(f"Could not add triplet ({subj}, {rel}, {obj}): {e}")
        return count
    
    # =========================================================================
    # SERIALIZATION
    # =========================================================================
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert graph to dictionary."""
        return {
            "nodes": [n.to_dict() for n in self._nodes.values()],
            "edges": [e.to_dict() for e in self._edges.values()],
            "metadata": {
                "num_nodes": self.num_nodes,
                "num_edges": self.num_edges,
            }
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any], ontology: "Ontology" = None) -> "KnowledgeGraph":
        """Create graph from dictionary."""
        graph = cls(ontology=ontology)
        
        for node_data in data.get("nodes", []):
            graph.add_node(
                node_id=node_data["id"],
                node_type=node_data.get("type", "entity"),
                label=node_data.get("label"),
                properties=node_data.get("properties", {}),
                validate=False,
            )
        
        for edge_data in data.get("edges", []):
            graph.add_edge(
                source=edge_data["source"],
                relation=edge_data["relation"],
                target=edge_data["target"],
                properties=edge_data.get("properties", {}),
                weight=edge_data.get("weight", 1.0),
                validate=False,
                create_nodes=False,
            )
        
        return graph
    
    def save(self, path: Union[str, Path]) -> None:
        """Save graph to JSON file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)
        
        logger.info(f"Graph saved to {path}")
    
    @classmethod
    def load(cls, path: Union[str, Path], ontology: "Ontology" = None) -> "KnowledgeGraph":
        """Load graph from JSON file."""
        with open(path, 'r', encoding='utf-8') as f:
            return cls.from_dict(json.load(f), ontology=ontology)
    
    # =========================================================================
    # UTILITIES
    # =========================================================================
    
    def stats(self) -> Dict[str, Any]:
        """Get graph statistics."""
        stats = {
            "num_nodes": self.num_nodes,
            "num_edges": self.num_edges,
        }
        
        if self._graph.number_of_nodes() > 0:
            stats["density"] = nx.density(self._graph)
            
            # Node type distribution
            type_counts = {}
            for node in self._nodes.values():
                type_counts[node.node_type] = type_counts.get(node.node_type, 0) + 1
            stats["node_types"] = type_counts
            
            # Relation type distribution
            rel_counts = {}
            for edge in self._edges.values():
                rel_counts[edge.relation] = rel_counts.get(edge.relation, 0) + 1
            stats["relation_types"] = rel_counts
        
        return stats
    
    def __repr__(self) -> str:
        return f"KnowledgeGraph(nodes={self.num_nodes}, edges={self.num_edges})"


# =============================================================================
# GRAPH BUILDER
# =============================================================================

class GraphBuilder:
    """
    Helper class for building knowledge graphs from extracted triplets.
    
    Example:
        >>> from ontorag.graph import GraphBuilder
        >>> from ontorag.extraction import RuleBasedExtractor
        >>> 
        >>> extractor = RuleBasedExtractor()
        >>> builder = GraphBuilder(ontology=ontology)
        >>> 
        >>> for text in documents:
        ...     result = extractor.extract(text)
        ...     builder.add_extraction_result(result)
        >>> 
        >>> graph = builder.build()
    """
    
    def __init__(
        self,
        ontology: "Ontology" = None,
        merge_similar: bool = True,
        min_confidence: float = 0.3,
    ):
        """
        Initialize the builder.
        
        Args:
            ontology: Ontology for validation
            merge_similar: Merge similar entity names
            min_confidence: Minimum confidence threshold
        """
        self.ontology = ontology
        self.merge_similar = merge_similar
        self.min_confidence = min_confidence
        
        self._nodes: Dict[str, Dict] = {}
        self._edges: Dict[Tuple, Dict] = {}
    
    def add_extraction_result(self, result) -> None:
        """
        Add entities and triplets from an extraction result.
        
        Args:
            result: ExtractionResult from an extractor
        """
        # Add entities as nodes
        for entity in result.entities:
            if entity.confidence >= self.min_confidence:
                self._add_node(
                    entity.normalized or entity.text.lower().replace(" ", "_"),
                    entity.type,
                )
        
        # Add triplets as edges
        for triplet in result.triplets:
            if triplet.confidence >= self.min_confidence:
                self._add_edge(
                    triplet.subject,
                    triplet.relation,
                    triplet.object,
                )
    
    def add_triplet(self, subject: str, relation: str, obj: str) -> None:
        """Add a single triplet."""
        self._add_node(subject, "entity")
        self._add_node(obj, "entity")
        self._add_edge(subject, relation, obj)
    
    def _add_node(self, node_id: str, node_type: str) -> None:
        """Add or update a node."""
        node_id = node_id.lower().replace(" ", "_")
        
        if node_id not in self._nodes:
            self._nodes[node_id] = {
                "id": node_id,
                "type": node_type,
                "count": 0,
            }
        
        self._nodes[node_id]["count"] += 1
    
    def _add_edge(self, source: str, relation: str, target: str) -> None:
        """Add or update an edge."""
        source = source.lower().replace(" ", "_")
        target = target.lower().replace(" ", "_")
        relation = relation.upper()
        
        key = (source, relation, target)
        
        if key not in self._edges:
            self._edges[key] = {
                "source": source,
                "relation": relation,
                "target": target,
                "count": 0,
            }
        
        self._edges[key]["count"] += 1
    
    def build(self) -> KnowledgeGraph:
        """
        Build the final knowledge graph.
        
        Returns:
            KnowledgeGraph with all added nodes and edges
        """
        graph = KnowledgeGraph(ontology=self.ontology)
        
        # Add nodes
        for node_data in self._nodes.values():
            graph.add_node(
                node_id=node_data["id"],
                node_type=node_data["type"],
                properties={"mention_count": node_data["count"]},
            )
        
        # Add edges
        for edge_data in self._edges.values():
            graph.add_edge(
                source=edge_data["source"],
                relation=edge_data["relation"],
                target=edge_data["target"],
                properties={"mention_count": edge_data["count"]},
                create_nodes=True,
            )
        
        return graph
    
    @property
    def stats(self) -> Dict[str, int]:
        """Get builder statistics."""
        return {
            "pending_nodes": len(self._nodes),
            "pending_edges": len(self._edges),
        }


__all__ = [
    "Node",
    "Edge",
    "SearchResult",
    "KnowledgeGraph",
    "GraphBuilder",
]
