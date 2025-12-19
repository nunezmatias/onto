"""
Analytics module for OntoRAG.

Provides lightweight metrics and coverage checks used by the public API,
examples, and tests.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

from ontorag.graph import KnowledgeGraph, GraphStatistics
from ontorag.ontology import Ontology


@dataclass
class GraphMetrics:
    """Container for graph statistics."""

    stats: GraphStatistics


@dataclass
class OntologyMetrics:
    """Simple ontology metrics for quick inspection."""

    num_classes: int = 0
    num_relations: int = 0
    num_root_classes: int = 0
    avg_synonyms_per_class: float = 0.0


class GraphAnalytics:
    """Wrapper around KnowledgeGraph statistics."""

    def __init__(self, knowledge_graph: KnowledgeGraph):
        self.kg = knowledge_graph

    def compute_metrics(self, compute_centrality: bool = True) -> GraphStatistics:
        """Return GraphStatistics computed by the knowledge graph."""
        return self.kg.get_statistics(compute_centrality=compute_centrality)

    def summary(self, compute_centrality: bool = True) -> str:
        """Human-friendly summary of current graph metrics."""
        stats = self.compute_metrics(compute_centrality=compute_centrality)
        return stats.summary()


class OntologyAnalytics:
    """Basic analytics for an Ontology instance."""

    def __init__(self, ontology: Ontology):
        self.ontology = ontology

    def compute_metrics(self) -> OntologyMetrics:
        """Compute simple ontology-level metrics."""
        num_classes = len(self.ontology.classes)
        num_relations = len(self.ontology.relations)
        root_classes = [c for c in self.ontology.classes.values() if c.parent is None]

        total_synonyms = sum(len(c.synonyms) for c in self.ontology.classes.values())
        avg_synonyms = total_synonyms / num_classes if num_classes else 0.0

        return OntologyMetrics(
            num_classes=num_classes,
            num_relations=num_relations,
            num_root_classes=len(root_classes),
            avg_synonyms_per_class=avg_synonyms,
        )

    def summary(self) -> str:
        """Return a readable summary of ontology metrics."""
        metrics = self.compute_metrics()
        lines = [
            "\n" + "=" * 60,
            f"ONTOLOGÍA: {self.ontology.name}",
            "=" * 60,
            f"Clases: {metrics.num_classes}",
            f"Relaciones: {metrics.num_relations}",
            f"Clases raíz: {metrics.num_root_classes}",
            f"Promedio de sinónimos por clase: {metrics.avg_synonyms_per_class:.2f}",
        ]
        return "\n".join(lines)

    def check_coverage(self, kg: KnowledgeGraph) -> Dict[str, Dict[str, object]]:
        """Check how well a graph covers the ontology types and relations."""
        node_types = {
            kg.graph.nodes[nid].get("node_type", "")
            for nid in kg.graph.nodes
            if kg.graph.nodes[nid].get("node_type")
        }
        relation_types = {
            data.get("relation", "")
            for _, _, data in kg.graph.edges(data=True)
            if data.get("relation")
        }

        onto_types = set(self.ontology.classes.keys())
        onto_rels = set(self.ontology.relations.keys())

        covered_types = sorted(node_types & onto_types)
        uncovered_types = sorted(node_types - onto_types)
        covered_rels = sorted(relation_types & onto_rels)
        uncovered_rels = sorted(relation_types - onto_rels)

        node_coverage = (
            len(covered_types) / len(node_types) * 100 if node_types else 100.0
        )
        relation_coverage = (
            len(covered_rels) / len(relation_types) * 100 if relation_types else 100.0
        )

        return {
            "node_types": {
                "total": len(node_types),
                "covered": covered_types,
                "uncovered": uncovered_types,
                "coverage_pct": node_coverage,
            },
            "relations": {
                "total": len(relation_types),
                "covered": covered_rels,
                "uncovered": uncovered_rels,
                "coverage_pct": relation_coverage,
            },
        }


__all__ = [
    "GraphAnalytics",
    "GraphMetrics",
    "OntologyAnalytics",
    "OntologyMetrics",
]
