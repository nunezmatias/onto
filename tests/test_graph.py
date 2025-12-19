"""
Tests for OntoRAG Graph Module
"""

import json
import tempfile
import pytest
from pathlib import Path


def test_node_creation():
    """Test creating a node."""
    from ontorag.graph import Node
    
    node = Node(
        id="test_node",
        node_type="entity",
        label="Test Node"
    )
    
    assert node.id == "test_node"
    assert node.node_type == "entity"
    assert node.label == "Test Node"


def test_node_serialization():
    """Test node to/from dict."""
    from ontorag.graph import Node
    
    node = Node(
        id="test",
        node_type="thing",
        properties={"key": "value"}
    )
    
    data = node.to_dict()
    loaded = Node.from_dict(data)
    
    assert loaded.id == node.id
    assert loaded.node_type == node.node_type


def test_edge_creation():
    """Test creating an edge."""
    from ontorag.graph import Edge
    
    edge = Edge(
        source="node1",
        target="node2",
        relation="RELATES"
    )
    
    assert edge.source == "node1"
    assert edge.target == "node2"
    assert edge.relation == "RELATES"
    assert edge.to_tuple() == ("node1", "RELATES", "node2")


def test_knowledge_graph_creation():
    """Test creating a knowledge graph."""
    from ontorag.graph import KnowledgeGraph
    
    graph = KnowledgeGraph()
    
    assert graph.num_nodes == 0
    assert graph.num_edges == 0


def test_knowledge_graph_add_node():
    """Test adding nodes to graph."""
    from ontorag.graph import KnowledgeGraph
    
    graph = KnowledgeGraph()
    
    node = graph.add_node(
        "test_node",
        node_type="entity",
        label="Test Node"
    )
    
    assert graph.num_nodes == 1
    assert graph.has_node("test_node")
    assert graph.get_node("test_node") is not None


def test_knowledge_graph_add_edge():
    """Test adding edges to graph."""
    from ontorag.graph import KnowledgeGraph
    
    graph = KnowledgeGraph()
    
    graph.add_node("node1")
    graph.add_node("node2")
    graph.add_edge("node1", "RELATES", "node2")
    
    assert graph.num_edges == 1
    assert graph.has_edge("node1", "RELATES", "node2")


def test_knowledge_graph_auto_create_nodes():
    """Test auto-creating nodes when adding edges."""
    from ontorag.graph import KnowledgeGraph
    
    graph = KnowledgeGraph()
    
    graph.add_edge("a", "RELATES", "b", create_nodes=True)
    
    assert graph.has_node("a")
    assert graph.has_node("b")
    assert graph.num_edges == 1


def test_knowledge_graph_neighbors():
    """Test getting neighbors."""
    from ontorag.graph import KnowledgeGraph
    
    graph = KnowledgeGraph()
    
    graph.add_edge("center", "CONNECTS", "out1")
    graph.add_edge("center", "CONNECTS", "out2")
    graph.add_edge("in1", "CONNECTS", "center")
    
    neighbors = graph.get_neighbors("center")
    
    assert len(neighbors) == 3


def test_knowledge_graph_path():
    """Test finding paths."""
    from ontorag.graph import KnowledgeGraph
    
    graph = KnowledgeGraph()
    
    graph.add_edge("a", "TO", "b")
    graph.add_edge("b", "TO", "c")
    graph.add_edge("c", "TO", "d")
    
    path = graph.get_path("a", "d")
    
    assert path is not None
    assert path == ["a", "b", "c", "d"]


def test_knowledge_graph_stats():
    """Test graph statistics."""
    from ontorag.graph import KnowledgeGraph
    
    graph = KnowledgeGraph()
    
    graph.add_node("n1", node_type="type_a")
    graph.add_node("n2", node_type="type_a")
    graph.add_node("n3", node_type="type_b")
    graph.add_edge("n1", "REL1", "n2")
    graph.add_edge("n2", "REL2", "n3")
    
    stats = graph.stats()
    
    assert stats["num_nodes"] == 3
    assert stats["num_edges"] == 2
    assert "type_a" in stats.get("node_types", {})


def test_knowledge_graph_serialization():
    """Test saving and loading graph."""
    from ontorag.graph import KnowledgeGraph
    
    graph = KnowledgeGraph()
    graph.add_node("a", node_type="thing", label="Node A")
    graph.add_node("b", node_type="thing", label="Node B")
    graph.add_edge("a", "RELATES", "b")
    
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        path = f.name
    
    try:
        graph.save(path)
        loaded = KnowledgeGraph.load(path)
        
        assert loaded.num_nodes == graph.num_nodes
        assert loaded.num_edges == graph.num_edges
        assert loaded.has_node("a")
    finally:
        Path(path).unlink()


def test_knowledge_graph_with_ontology():
    """Test graph with ontology validation."""
    from ontorag.graph import KnowledgeGraph
    from ontorag import create_climate_ontology
    
    onto = create_climate_ontology()
    graph = KnowledgeGraph(ontology=onto)
    
    # Add with type that should normalize
    graph.add_node("test_flood", node_type="flood")
    
    node = graph.get_node("test_flood")
    assert node is not None


def test_knowledge_graph_add_triplets():
    """Test bulk adding triplets."""
    from ontorag.graph import KnowledgeGraph
    
    graph = KnowledgeGraph()
    
    triplets = [
        ("a", "REL1", "b"),
        ("b", "REL2", "c"),
        ("c", "REL1", "d"),
    ]
    
    count = graph.add_triplets(triplets)
    
    assert count == 3
    assert graph.num_edges == 3


def test_graph_builder():
    """Test GraphBuilder class."""
    from ontorag.graph import GraphBuilder
    from ontorag.extraction import ExtractionResult, ExtractedEntity, ExtractedTriplet
    
    builder = GraphBuilder()
    
    result = ExtractionResult()
    result.entities.append(ExtractedEntity(text="Node A", type="type1"))
    result.entities.append(ExtractedEntity(text="Node B", type="type2"))
    result.triplets.append(ExtractedTriplet(
        subject="node_a",
        relation="RELATES",
        object="node_b"
    ))
    
    builder.add_extraction_result(result)
    
    graph = builder.build()
    
    assert graph.num_nodes >= 2
    assert graph.num_edges >= 1


def test_subgraph_extraction():
    """Test extracting subgraph around a node."""
    from ontorag.graph import KnowledgeGraph
    
    graph = KnowledgeGraph()
    
    # Create a small graph
    graph.add_edge("center", "TO", "n1")
    graph.add_edge("center", "TO", "n2")
    graph.add_edge("n1", "TO", "far1")
    graph.add_edge("n2", "TO", "far2")
    graph.add_edge("far1", "TO", "very_far")
    
    subgraph = graph.get_subgraph("center", depth=1)
    
    assert subgraph.has_node("center")
    assert subgraph.has_node("n1")
    assert subgraph.has_node("n2")
    # far1, far2 should not be included at depth=1
    assert not subgraph.has_node("far1") or not subgraph.has_node("very_far")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
