#!/usr/bin/env python3
"""
OntoRAG Example: Building Knowledge Graphs

This example shows how to:
1. Create a knowledge graph from triplets
2. Use GraphBuilder for batch processing
3. Search and explore the graph
4. Generate visualizations

Prerequisites:
    $ pip install ontorag[all]

Run this example:
    $ python 05_build_knowledge_graph.py
"""

from ontorag import (
    KnowledgeGraph,
    GraphBuilder,
    create_climate_ontology,
    GraphAnalytics,
    GraphVisualizer,
)


def main():
    print("=" * 60)
    print("OntoRAG: Building Knowledge Graphs")
    print("=" * 60)
    
    # =========================================================================
    # STEP 1: Create Ontology and Graph
    # =========================================================================
    print("\n📚 Step 1: Setting up ontology and graph...")
    
    # Load ontology
    onto = create_climate_ontology()
    print(f"   Ontology: {onto.name}")
    print(f"   Classes: {len(onto.classes)}")
    print(f"   Relations: {len(onto.relations)}")
    
    # Create knowledge graph with ontology
    graph = KnowledgeGraph(ontology=onto)
    
    # =========================================================================
    # STEP 2: Add Nodes
    # =========================================================================
    print("\n🔷 Step 2: Adding nodes...")
    
    # Adaptation measures
    measures = [
        ("green_roof", "green_infrastructure", "Green Roof", {"cooling": "2-3°C", "cost": "€100-150/m²"}),
        ("rain_garden", "green_infrastructure", "Rain Garden", {"retention": "1000L"}),
        ("urban_forest", "green_infrastructure", "Urban Forest", {"coverage": "40%"}),
        ("flood_barrier", "grey_infrastructure", "Flood Barrier", {"height": "3m"}),
        ("early_warning", "early_warning_system", "Early Warning System", {"lead_time": "24h"}),
        ("mangrove_restoration", "wetland_restoration", "Mangrove Restoration", {}),
    ]
    
    for node_id, node_type, label, props in measures:
        graph.add_node(node_id, node_type=node_type, label=label, properties=props)
    
    # Climate hazards
    hazards = [
        ("urban_heat", "heat_wave", "Urban Heat Island"),
        ("flood", "flood", "Urban Flooding"),
        ("drought", "drought", "Drought"),
        ("storm_surge", "sea_level_rise", "Storm Surge"),
        ("wildfire", "wildfire", "Wildfire"),
    ]
    
    for node_id, node_type, label in hazards:
        graph.add_node(node_id, node_type=node_type, label=label)
    
    # Locations
    locations = [
        ("rotterdam", "urban_area", "Rotterdam"),
        ("copenhagen", "coastal_area", "Copenhagen"),
        ("singapore", "urban_area", "Singapore"),
        ("bangladesh", "coastal_area", "Bangladesh"),
    ]
    
    for node_id, node_type, label in locations:
        graph.add_node(node_id, node_type=node_type, label=label)
    
    print(f"   Added {graph.num_nodes} nodes")
    
    # =========================================================================
    # STEP 3: Add Edges (Relationships)
    # =========================================================================
    print("\n🔗 Step 3: Adding edges...")
    
    # MITIGATES relationships
    mitigates = [
        ("green_roof", "urban_heat"),
        ("green_roof", "flood"),
        ("urban_forest", "urban_heat"),
        ("rain_garden", "flood"),
        ("flood_barrier", "flood"),
        ("flood_barrier", "storm_surge"),
        ("early_warning", "flood"),
        ("mangrove_restoration", "storm_surge"),
    ]
    
    for source, target in mitigates:
        graph.add_edge(source, "MITIGATES", target)
    
    # IMPLEMENTED_IN relationships
    implemented = [
        ("green_roof", "rotterdam"),
        ("green_roof", "copenhagen"),
        ("urban_forest", "singapore"),
        ("flood_barrier", "rotterdam"),
        ("early_warning", "bangladesh"),
        ("mangrove_restoration", "bangladesh"),
    ]
    
    for source, target in implemented:
        graph.add_edge(source, "IMPLEMENTED_IN", target)
    
    print(f"   Added {graph.num_edges} edges")
    
    # =========================================================================
    # STEP 4: Alternative - Use GraphBuilder
    # =========================================================================
    print("\n🔨 Step 4: Using GraphBuilder (alternative)...")
    
    # GraphBuilder is useful for processing extraction results
    builder = GraphBuilder(ontology=onto)
    
    # Add triplets directly
    triplets = [
        ("cool_pavement", "MITIGATES", "urban_heat"),
        ("cool_pavement", "IMPLEMENTED_IN", "los_angeles"),
        ("seawall", "PROTECTS", "storm_surge"),
        ("seawall", "IMPLEMENTED_IN", "miami"),
    ]
    
    for subj, rel, obj in triplets:
        builder.add_triplet(subj, rel, obj)
    
    # Build into a separate graph
    builder_graph = builder.build()
    print(f"   GraphBuilder created: {builder_graph.num_nodes} nodes, {builder_graph.num_edges} edges")
    
    # =========================================================================
    # STEP 5: Search the Graph
    # =========================================================================
    print("\n🔍 Step 5: Searching the graph...")
    
    # Semantic search
    query = "cooling solutions for cities"
    results = graph.search(query, k=5, search_type="hybrid")
    
    print(f"\n   Query: '{query}'")
    print("   Results:")
    for i, r in enumerate(results, 1):
        if r.node:
            print(f"   {i}. {r.node.label} ({r.node.node_type}) - Score: {r.score:.3f}")
    
    # Search by type
    print("\n   Search for flood mitigation:")
    flood_results = graph.search("flood prevention measures", k=5)
    for r in flood_results:
        if r.node and r.node.node_type in ["green_infrastructure", "grey_infrastructure"]:
            print(f"   - {r.node.label}")
    
    # =========================================================================
    # STEP 6: Explore Relationships
    # =========================================================================
    print("\n🔗 Step 6: Exploring relationships...")
    
    # Get neighbors
    print("\n   Neighbors of 'green_roof':")
    neighbors = graph.get_neighbors("green_roof")
    for relation, direction, neighbor in neighbors:
        arrow = "→" if direction == "out" else "←"
        print(f"   {arrow} {relation} {arrow} {neighbor.label}")
    
    # Get what a measure mitigates
    print("\n   What does 'flood_barrier' mitigate?")
    edges = graph.get_edges_from("flood_barrier")
    for edge in edges:
        if edge.relation == "MITIGATES":
            target = graph.get_node(edge.target)
            print(f"   - {target.label}")
    
    # Find path between nodes
    print("\n   Path from 'green_roof' to 'bangladesh':")
    path = graph.get_path("green_roof", "bangladesh")
    if path:
        print(f"   {' → '.join(path)}")
    else:
        print("   No direct path found")
    
    # =========================================================================
    # STEP 7: Graph Analytics
    # =========================================================================
    print("\n📊 Step 7: Graph analytics...")
    
    analytics = GraphAnalytics(graph)
    metrics = analytics.compute_metrics()
    
    print(f"\n   Nodes: {metrics.num_nodes}")
    print(f"   Edges: {metrics.num_edges}")
    print(f"   Density: {metrics.density:.4f}")
    print(f"   Avg degree: {metrics.avg_degree:.2f}")
    
    print("\n   Node types:")
    for ntype, count in metrics.node_type_distribution.items():
        print(f"   - {ntype}: {count}")
    
    print("\n   Relation types:")
    for rtype, count in metrics.relation_type_distribution.items():
        print(f"   - {rtype}: {count}")
    
    # Find hub nodes
    print("\n   Hub nodes (highest degree):")
    hubs = analytics.find_hubs(k=3)
    for node_id, degree in hubs:
        node = graph.get_node(node_id)
        print(f"   - {node.label if node else node_id}: {degree}")
    
    # =========================================================================
    # STEP 8: Get Subgraph
    # =========================================================================
    print("\n🔷 Step 8: Extracting subgraph...")
    
    # Get subgraph around a node
    subgraph = graph.get_subgraph("green_roof", depth=2)
    print(f"   Subgraph around 'green_roof' (depth=2):")
    print(f"   - Nodes: {subgraph.num_nodes}")
    print(f"   - Edges: {subgraph.num_edges}")
    
    # =========================================================================
    # STEP 9: Save and Visualize
    # =========================================================================
    print("\n💾 Step 9: Saving and visualizing...")
    
    # Save graph
    graph.save("climate_knowledge_graph.json")
    print("   Saved to: climate_knowledge_graph.json")
    
    # Generate visualization
    try:
        viz = GraphVisualizer(graph)
        viz.to_html("climate_graph.html", title="Climate Adaptation Knowledge Graph")
        print("   Visualization: climate_graph.html")
    except Exception as e:
        print(f"   Visualization skipped: {e}")
    
    # =========================================================================
    # STEP 10: Load Existing Graph
    # =========================================================================
    print("\n📂 Step 10: Loading existing graph...")
    
    # Load the graph we just saved
    loaded_graph = KnowledgeGraph.load("climate_knowledge_graph.json", ontology=onto)
    print(f"   Loaded: {loaded_graph.num_nodes} nodes, {loaded_graph.num_edges} edges")
    
    print("\n" + "=" * 60)
    print("✅ Knowledge graph building complete!")
    print("=" * 60)
    
    print("""
Next steps:
1. Use the graph for RAG queries (see 06_rag_queries.py)
2. Add more nodes and edges from extracted triplets
3. Generate visualizations for exploration
4. Export to other formats (GraphML, etc.)
    """)


if __name__ == "__main__":
    main()
