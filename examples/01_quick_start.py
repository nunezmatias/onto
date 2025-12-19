#!/usr/bin/env python3
"""
OntoRAG Quick Start Example

This example demonstrates the basic workflow of OntoRAG:
1. Create an ontology (or load from JSON)
2. Build a knowledge graph
3. Search and query the graph

Run this example:
    $ python 01_quick_start.py

Prerequisites:
    $ pip install ontorag[all]
"""

from ontorag import (
    Ontology,
    OntologyClass,
    OntologyRelation,
    KnowledgeGraph,
    create_climate_ontology,
)


def main():
    print("=" * 60)
    print("OntoRAG Quick Start")
    print("=" * 60)
    
    # =========================================================================
    # STEP 1: Create or Load Ontology
    # =========================================================================
    print("\n📚 Step 1: Creating ontology...")
    
    # Option A: Use predefined climate ontology
    onto = create_climate_ontology()
    print(f"   Created: {onto.name}")
    print(f"   Classes: {len(onto.classes)}")
    print(f"   Relations: {len(onto.relations)}")
    
    # Option B: Create custom ontology
    # onto = Ontology("my_domain", "My Custom Ontology")
    # onto.add_class(OntologyClass(id="entity", name="Entity"))
    # onto.add_class(OntologyClass(id="person", name="Person", parent="entity"))
    # onto.add_relation(OntologyRelation(id="KNOWS", name="knows"))
    
    # Option C: Load from JSON file
    # onto = Ontology.load("my_ontology.json")
    
    # =========================================================================
    # STEP 2: Build Knowledge Graph
    # =========================================================================
    print("\n🔷 Step 2: Building knowledge graph...")
    
    # Create graph with ontology for validation
    graph = KnowledgeGraph(ontology=onto)
    
    # Add nodes (entities)
    graph.add_node("green_roof", node_type="green_infrastructure",
                   label="Green Roof",
                   properties={"cooling_effect": "2-3°C"})
    
    graph.add_node("urban_heat_island", node_type="heat_wave",
                   label="Urban Heat Island")
    
    graph.add_node("flood", node_type="flood",
                   label="Urban Flooding")
    
    graph.add_node("rotterdam", node_type="urban_area",
                   label="Rotterdam")
    
    graph.add_node("early_warning_system", node_type="early_warning_system",
                   label="Early Warning System")
    
    # Add edges (relationships)
    graph.add_edge("green_roof", "MITIGATES", "urban_heat_island")
    graph.add_edge("green_roof", "MITIGATES", "flood")
    graph.add_edge("green_roof", "IMPLEMENTED_IN", "rotterdam")
    graph.add_edge("early_warning_system", "PROTECTS", "flood")
    
    print(f"   Nodes: {graph.num_nodes}")
    print(f"   Edges: {graph.num_edges}")
    
    # =========================================================================
    # STEP 3: Search the Graph
    # =========================================================================
    print("\n🔍 Step 3: Searching the graph...")
    
    # Semantic search
    query = "cooling solutions for urban areas"
    results = graph.search(query, k=3, search_type="hybrid")
    
    print(f"\n   Query: '{query}'")
    print("   Results:")
    for i, result in enumerate(results, 1):
        print(f"   {i}. {result.node.label} ({result.node.node_type}) - Score: {result.score:.3f}")
    
    # =========================================================================
    # STEP 4: Explore Neighbors
    # =========================================================================
    print("\n🔗 Step 4: Exploring relationships...")
    
    neighbors = graph.get_neighbors("green_roof")
    print(f"\n   Neighbors of 'green_roof':")
    for relation, direction, neighbor in neighbors:
        arrow = "→" if direction == "out" else "←"
        print(f"   {arrow} {relation} {arrow} {neighbor.label}")
    
    # =========================================================================
    # STEP 5: Get Graph Statistics
    # =========================================================================
    print("\n📊 Step 5: Graph statistics...")
    
    stats = graph.stats()
    print(f"   Nodes: {stats['num_nodes']}")
    print(f"   Edges: {stats['num_edges']}")
    print(f"   Density: {stats.get('density', 0):.4f}")
    print(f"   Node types: {list(stats.get('node_types', {}).keys())}")
    
    # =========================================================================
    # STEP 6: Save the Graph
    # =========================================================================
    print("\n💾 Step 6: Saving graph...")
    
    graph.save("quick_start_graph.json")
    print("   Saved to: quick_start_graph.json")
    
    # You can load it later with:
    # loaded_graph = KnowledgeGraph.load("quick_start_graph.json", ontology=onto)
    
    print("\n" + "=" * 60)
    print("✅ Quick start complete!")
    print("=" * 60)
    
    print("""
Next steps:
1. Try RAG queries (see 06_rag_queries.py)
2. Collect documents from academic sources (see 03_corpus_collection.py)
3. Extract triplets from text (see 04_triplet_extraction.py)
4. Visualize your graph (see examples in visualization module)
""")


if __name__ == "__main__":
    main()
