"""
OntoRAG Analytics Module

This module provides analytics for knowledge graphs and ontologies.

Key Components:
    - GraphAnalytics: Analyze graph structure and metrics
    - OntologyAnalytics: Analyze ontology coverage and usage
    - GraphMetrics: Dataclass for graph statistics
    - OntologyMetrics: Dataclass for ontology statistics

Example:
    >>> from ontorag.analytics import GraphAnalytics, OntologyAnalytics
    >>> from ontorag.graph import KnowledgeGraph
    >>> from ontorag.ontology import create_climate_ontology
    >>> 
    >>> # Analyze graph
    >>> graph = KnowledgeGraph.load("my_graph.json")
    >>> analytics = GraphAnalytics(graph)
    >>> metrics = analytics.compute_metrics()
    >>> print(f"Density: {metrics.density:.4f}")
    >>> 
    >>> # Analyze ontology usage
    >>> onto = create_climate_ontology()
    >>> onto_analytics = OntologyAnalytics(onto, graph)
    >>> coverage = onto_analytics.check_coverage()
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from ..graph import KnowledgeGraph
    from ..ontology import Ontology

logger = logging.getLogger(__name__)

# Try to import networkx
try:
    import networkx as nx
    HAS_NETWORKX = True
except ImportError:
    HAS_NETWORKX = False
    nx = None


# =============================================================================
# METRICS DATACLASSES
# =============================================================================

@dataclass
class GraphMetrics:
    """
    Metrics about a knowledge graph.
    
    Attributes:
        num_nodes: Total number of nodes
        num_edges: Total number of edges
        density: Graph density (0-1)
        avg_degree: Average node degree
        avg_in_degree: Average in-degree
        avg_out_degree: Average out-degree
        num_connected_components: Number of weakly connected components
        largest_component_size: Size of the largest component
        node_type_distribution: Count of nodes by type
        relation_type_distribution: Count of edges by relation
        top_nodes_by_degree: Highest degree nodes
        clustering_coefficient: Average clustering coefficient
    """
    num_nodes: int = 0
    num_edges: int = 0
    density: float = 0.0
    avg_degree: float = 0.0
    avg_in_degree: float = 0.0
    avg_out_degree: float = 0.0
    num_connected_components: int = 0
    largest_component_size: int = 0
    node_type_distribution: Dict[str, int] = field(default_factory=dict)
    relation_type_distribution: Dict[str, int] = field(default_factory=dict)
    top_nodes_by_degree: List[Tuple[str, int]] = field(default_factory=list)
    clustering_coefficient: float = 0.0
    diameter: Optional[int] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "num_nodes": self.num_nodes,
            "num_edges": self.num_edges,
            "density": self.density,
            "avg_degree": self.avg_degree,
            "avg_in_degree": self.avg_in_degree,
            "avg_out_degree": self.avg_out_degree,
            "num_connected_components": self.num_connected_components,
            "largest_component_size": self.largest_component_size,
            "node_type_distribution": self.node_type_distribution,
            "relation_type_distribution": self.relation_type_distribution,
            "top_nodes_by_degree": self.top_nodes_by_degree,
            "clustering_coefficient": self.clustering_coefficient,
            "diameter": self.diameter,
        }
    
    def summary(self) -> str:
        """Generate a text summary."""
        lines = [
            "GRAPH METRICS",
            "=" * 40,
            f"Nodes: {self.num_nodes}",
            f"Edges: {self.num_edges}",
            f"Density: {self.density:.4f}",
            f"Avg Degree: {self.avg_degree:.2f}",
            f"Connected Components: {self.num_connected_components}",
            f"Largest Component: {self.largest_component_size} nodes",
            "",
            "Node Types:",
        ]
        
        for ntype, count in sorted(
            self.node_type_distribution.items(),
            key=lambda x: -x[1]
        )[:10]:
            lines.append(f"  {ntype}: {count}")
        
        lines.append("\nRelation Types:")
        for rtype, count in sorted(
            self.relation_type_distribution.items(),
            key=lambda x: -x[1]
        )[:10]:
            lines.append(f"  {rtype}: {count}")
        
        return "\n".join(lines)


@dataclass
class OntologyMetrics:
    """
    Metrics about ontology usage in a knowledge graph.
    
    Attributes:
        num_classes: Total classes in ontology
        num_relations: Total relations in ontology
        classes_used: Classes that appear in the graph
        classes_unused: Classes not in the graph
        relations_used: Relations that appear in the graph
        relations_unused: Relations not in the graph
        coverage_rate: Percentage of ontology used
        validation_errors: Nodes/edges violating ontology
    """
    num_classes: int = 0
    num_relations: int = 0
    classes_used: List[str] = field(default_factory=list)
    classes_unused: List[str] = field(default_factory=list)
    relations_used: List[str] = field(default_factory=list)
    relations_unused: List[str] = field(default_factory=list)
    coverage_rate: float = 0.0
    max_hierarchy_depth: int = 0
    avg_synonyms_per_class: float = 0.0
    validation_errors: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "num_classes": self.num_classes,
            "num_relations": self.num_relations,
            "classes_used": self.classes_used,
            "classes_unused": self.classes_unused,
            "relations_used": self.relations_used,
            "relations_unused": self.relations_unused,
            "coverage_rate": self.coverage_rate,
            "max_hierarchy_depth": self.max_hierarchy_depth,
            "avg_synonyms_per_class": self.avg_synonyms_per_class,
            "validation_errors": self.validation_errors,
        }
    
    def summary(self) -> str:
        """Generate a text summary."""
        lines = [
            "ONTOLOGY METRICS",
            "=" * 40,
            f"Classes: {self.num_classes} ({len(self.classes_used)} used)",
            f"Relations: {self.num_relations} ({len(self.relations_used)} used)",
            f"Coverage: {self.coverage_rate:.1f}%",
            f"Max Hierarchy Depth: {self.max_hierarchy_depth}",
            "",
            "Unused Classes:",
        ]
        
        for cls in self.classes_unused[:10]:
            lines.append(f"  - {cls}")
        
        if self.validation_errors:
            lines.append("\nValidation Errors:")
            for err in self.validation_errors[:10]:
                lines.append(f"  ⚠ {err}")
        
        return "\n".join(lines)


# =============================================================================
# GRAPH ANALYTICS
# =============================================================================

class GraphAnalytics:
    """
    Analyze knowledge graph structure and metrics.
    
    Provides methods for computing various graph statistics,
    finding important nodes, and identifying patterns.
    
    Example:
        >>> analytics = GraphAnalytics(graph)
        >>> 
        >>> # Get all metrics
        >>> metrics = analytics.compute_metrics()
        >>> print(metrics.summary())
        >>> 
        >>> # Find hub nodes
        >>> hubs = analytics.find_hubs(k=10)
        >>> 
        >>> # Find isolated nodes
        >>> isolated = analytics.find_isolated_nodes()
    """
    
    def __init__(self, graph: "KnowledgeGraph"):
        """
        Initialize with a knowledge graph.
        
        Args:
            graph: Knowledge graph to analyze
        """
        self.graph = graph
        
        if not HAS_NETWORKX:
            raise ImportError("networkx required. Install: pip install networkx")
    
    def compute_metrics(self) -> GraphMetrics:
        """
        Compute comprehensive graph metrics.
        
        Returns:
            GraphMetrics with all computed statistics
        """
        g = self.graph._graph
        metrics = GraphMetrics()
        
        metrics.num_nodes = g.number_of_nodes()
        metrics.num_edges = g.number_of_edges()
        
        if metrics.num_nodes == 0:
            return metrics
        
        # Density
        metrics.density = nx.density(g)
        
        # Degree statistics
        degrees = dict(g.degree())
        in_degrees = dict(g.in_degree())
        out_degrees = dict(g.out_degree())
        
        metrics.avg_degree = sum(degrees.values()) / metrics.num_nodes
        metrics.avg_in_degree = sum(in_degrees.values()) / metrics.num_nodes
        metrics.avg_out_degree = sum(out_degrees.values()) / metrics.num_nodes
        
        # Top nodes by degree
        sorted_degrees = sorted(degrees.items(), key=lambda x: -x[1])
        metrics.top_nodes_by_degree = sorted_degrees[:10]
        
        # Connected components
        undirected = g.to_undirected()
        components = list(nx.connected_components(undirected))
        metrics.num_connected_components = len(components)
        metrics.largest_component_size = max(len(c) for c in components) if components else 0
        
        # Clustering coefficient
        try:
            metrics.clustering_coefficient = nx.average_clustering(undirected)
        except Exception:
            metrics.clustering_coefficient = 0.0
        
        # Node type distribution
        for node in self.graph.nodes:
            ntype = node.node_type
            metrics.node_type_distribution[ntype] = metrics.node_type_distribution.get(ntype, 0) + 1
        
        # Relation type distribution
        for edge in self.graph.edges:
            rtype = edge.relation
            metrics.relation_type_distribution[rtype] = metrics.relation_type_distribution.get(rtype, 0) + 1
        
        return metrics
    
    def get_node_statistics(self) -> Dict[str, Any]:
        """Get detailed node statistics."""
        stats = {
            "total": self.graph.num_nodes,
            "by_type": {},
            "with_properties": 0,
            "avg_label_length": 0,
        }
        
        total_label_length = 0
        
        for node in self.graph.nodes:
            # By type
            ntype = node.node_type
            stats["by_type"][ntype] = stats["by_type"].get(ntype, 0) + 1
            
            # With properties
            if node.properties:
                stats["with_properties"] += 1
            
            # Label length
            total_label_length += len(node.label)
        
        if stats["total"] > 0:
            stats["avg_label_length"] = total_label_length / stats["total"]
        
        return stats
    
    def find_hubs(self, k: int = 10, metric: str = "degree") -> List[Tuple[str, float]]:
        """
        Find hub nodes (highest centrality).
        
        Args:
            k: Number of hubs to return
            metric: Centrality metric ("degree", "betweenness", "pagerank")
            
        Returns:
            List of (node_id, score) tuples
        """
        g = self.graph._graph
        
        if g.number_of_nodes() == 0:
            return []
        
        if metric == "degree":
            scores = dict(g.degree())
        elif metric == "betweenness":
            scores = nx.betweenness_centrality(g)
        elif metric == "pagerank":
            try:
                scores = nx.pagerank(g)
            except Exception:
                scores = dict(g.degree())
        else:
            scores = dict(g.degree())
        
        sorted_scores = sorted(scores.items(), key=lambda x: -x[1])
        return sorted_scores[:k]
    
    def find_isolated_nodes(self) -> List[str]:
        """
        Find nodes with no connections.
        
        Returns:
            List of isolated node IDs
        """
        return [
            node_id
            for node_id in self.graph._nodes
            if self.graph._graph.degree(node_id) == 0
        ]
    
    def get_relation_matrix(self) -> Dict[Tuple[str, str], Dict[str, int]]:
        """
        Get relation frequency matrix between node types.
        
        Returns:
            Dict mapping (source_type, target_type) to {relation: count}
        """
        matrix = {}
        
        for edge in self.graph.edges:
            source_node = self.graph.get_node(edge.source)
            target_node = self.graph.get_node(edge.target)
            
            if source_node and target_node:
                key = (source_node.node_type, target_node.node_type)
                if key not in matrix:
                    matrix[key] = {}
                
                rel = edge.relation
                matrix[key][rel] = matrix[key].get(rel, 0) + 1
        
        return matrix


# =============================================================================
# ONTOLOGY ANALYTICS
# =============================================================================

class OntologyAnalytics:
    """
    Analyze ontology usage in a knowledge graph.
    
    Compares the ontology definition with actual graph usage
    to identify coverage gaps and validation issues.
    
    Example:
        >>> onto = create_climate_ontology()
        >>> analytics = OntologyAnalytics(onto, graph)
        >>> 
        >>> metrics = analytics.compute_metrics()
        >>> print(f"Coverage: {metrics.coverage_rate:.1f}%")
        >>> 
        >>> coverage = analytics.check_coverage()
        >>> print("Unused classes:", coverage["unused_classes"])
    """
    
    def __init__(self, ontology: "Ontology", graph: "KnowledgeGraph" = None):
        """
        Initialize with ontology and optional graph.
        
        Args:
            ontology: Ontology to analyze
            graph: Knowledge graph to compare against
        """
        self.ontology = ontology
        self.graph = graph
    
    def compute_metrics(self) -> OntologyMetrics:
        """
        Compute ontology metrics.
        
        Returns:
            OntologyMetrics with all statistics
        """
        metrics = OntologyMetrics()
        
        # Basic counts
        metrics.num_classes = len(self.ontology.classes)
        metrics.num_relations = len(self.ontology.relations)
        
        # Hierarchy depth
        max_depth = 0
        for cls_id in self.ontology.classes:
            depth = len(self.ontology.get_class_hierarchy(cls_id))
            max_depth = max(max_depth, depth)
        metrics.max_hierarchy_depth = max_depth
        
        # Synonyms
        total_synonyms = 0
        for cls in self.ontology.classes.values():
            total_synonyms += len(cls.synonyms)
        metrics.avg_synonyms_per_class = total_synonyms / max(1, metrics.num_classes)
        
        # Coverage analysis if graph provided
        if self.graph:
            coverage = self.check_coverage()
            metrics.classes_used = coverage["used_classes"]
            metrics.classes_unused = coverage["unused_classes"]
            metrics.relations_used = coverage["used_relations"]
            metrics.relations_unused = coverage["unused_relations"]
            
            total_items = metrics.num_classes + metrics.num_relations
            used_items = len(metrics.classes_used) + len(metrics.relations_used)
            metrics.coverage_rate = (used_items / max(1, total_items)) * 100
        
        return metrics
    
    def check_coverage(self) -> Dict[str, Any]:
        """
        Check ontology coverage in the graph.
        
        Returns:
            Dictionary with coverage statistics
        """
        if not self.graph:
            return {}
        
        # Track which classes and relations are used
        used_classes = set()
        used_relations = set()
        
        # Check node types
        for node in self.graph.nodes:
            # Direct match
            if node.node_type in self.ontology.classes:
                used_classes.add(node.node_type)
            
            # Try to normalize
            normalized = self.ontology.normalize_entity(node.node_type)
            if normalized:
                used_classes.add(normalized)
        
        # Check edge relations
        for edge in self.graph.edges:
            if edge.relation in self.ontology.relations:
                used_relations.add(edge.relation)
            
            normalized = self.ontology.normalize_relation(edge.relation)
            if normalized:
                used_relations.add(normalized)
        
        # Find unused
        all_classes = set(self.ontology.classes.keys())
        all_relations = set(self.ontology.relations.keys())
        
        unused_classes = all_classes - used_classes
        unused_relations = all_relations - used_relations
        
        return {
            "used_classes": list(used_classes),
            "unused_classes": list(unused_classes),
            "used_relations": list(used_relations),
            "unused_relations": list(unused_relations),
            "class_coverage": len(used_classes) / max(1, len(all_classes)) * 100,
            "relation_coverage": len(used_relations) / max(1, len(all_relations)) * 100,
        }
    
    def validate_graph(self) -> List[str]:
        """
        Validate graph against ontology.
        
        Returns:
            List of validation error messages
        """
        errors = []
        
        if not self.graph:
            return errors
        
        # Check node types
        for node in self.graph.nodes:
            if node.node_type not in self.ontology.classes:
                normalized = self.ontology.normalize_entity(node.node_type)
                if not normalized:
                    errors.append(f"Unknown node type: {node.node_type} (node: {node.id})")
        
        # Check edge relations
        for edge in self.graph.edges:
            rel = self.ontology.get_relation(edge.relation)
            if not rel:
                errors.append(f"Unknown relation: {edge.relation}")
                continue
            
            # Check domain/range constraints
            source_node = self.graph.get_node(edge.source)
            target_node = self.graph.get_node(edge.target)
            
            if source_node and rel.domain:
                if source_node.node_type not in rel.domain:
                    # Check hierarchy
                    valid = any(
                        self.ontology.is_subclass_of(source_node.node_type, d)
                        for d in rel.domain
                    )
                    if not valid:
                        errors.append(
                            f"Domain violation: {edge.source} ({source_node.node_type}) "
                            f"cannot be source of {edge.relation}"
                        )
            
            if target_node and rel.range:
                if target_node.node_type not in rel.range:
                    valid = any(
                        self.ontology.is_subclass_of(target_node.node_type, r)
                        for r in rel.range
                    )
                    if not valid:
                        errors.append(
                            f"Range violation: {edge.target} ({target_node.node_type}) "
                            f"cannot be target of {edge.relation}"
                        )
        
        return errors


__all__ = [
    "GraphMetrics",
    "OntologyMetrics",
    "GraphAnalytics",
    "OntologyAnalytics",
]
