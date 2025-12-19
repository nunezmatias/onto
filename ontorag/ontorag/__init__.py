"""
OntoRAG: Ontology-First Retrieval Augmented Generation

Una biblioteca Python para construir sistemas RAG sobre grafos de conocimiento
con énfasis en la ingeniería ontológica.

Diferenciadores clave vs LlamaIndex/LangChain:
- Ontología como ciudadano de primera clase (editable, verificable)
- Relaciones semánticas ricas (no verbos vacíos)
- Normalización semántica integrada
- Validaciones de consistencia ontológica
- Búsqueda híbrida: semántica + estructural
"""

__version__ = "0.1.0"
__author__ = "OntoRAG Team"

from ontorag.ontology import (
    Ontology,
    OntologyClass,
    OntologyRelation,
    RelationType,
    ValidationResult,
)
from ontorag.graph import KnowledgeGraph, Node, Edge, GraphBuilder
from ontorag.visualization import GraphVisualizer, OntologyVisualizer
from ontorag.analytics import GraphAnalytics, OntologyAnalytics
from ontorag.rag import RAGEngine
from ontorag.extraction import TripletExtractor

__all__ = [
    "Ontology",
    "OntologyClass", 
    "OntologyRelation",
    "RelationType",
    "ValidationResult",
    "KnowledgeGraph",
    "Node",
    "Edge",
    "GraphBuilder",
    "GraphVisualizer",
    "OntologyVisualizer",
    "GraphAnalytics",
    "OntologyAnalytics",
    "RAGEngine",
    "TripletExtractor",
]
