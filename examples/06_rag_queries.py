#!/usr/bin/env python3
"""
OntoRAG Example: RAG Queries

This example shows how to:
1. Set up a RAG engine with knowledge graph
2. Query using different LLM providers
3. Customize prompts and context
4. Handle responses

Prerequisites:
    $ pip install ontorag[all]
    
For full functionality:
    $ pip install ontorag[openai]     # OpenAI
    $ pip install ontorag[anthropic]  # Anthropic Claude
    $ pip install ontorag[google]     # Google Gemini

Run this example:
    $ python 06_rag_queries.py

Note: Set your API key as environment variable:
    $ export OPENAI_API_KEY="sk-..."
    $ export ANTHROPIC_API_KEY="..."
"""

import os

from ontorag import (
    KnowledgeGraph,
    create_climate_ontology,
    RAGEngine,
)


def create_sample_graph():
    """Create a sample knowledge graph for demonstration."""
    onto = create_climate_ontology()
    graph = KnowledgeGraph(ontology=onto)
    
    # Add measures
    graph.add_node("green_roof", "green_infrastructure", "Green Roof",
                   {"cooling": "2-3°C", "stormwater": "70% reduction"})
    graph.add_node("urban_forest", "green_infrastructure", "Urban Forest",
                   {"cooling": "1-5°C", "coverage": "recommended 40%"})
    graph.add_node("rain_garden", "green_infrastructure", "Rain Garden",
                   {"retention": "up to 1000L per storm"})
    graph.add_node("flood_barrier", "grey_infrastructure", "Flood Barrier",
                   {"protection": "1-in-100 year events"})
    graph.add_node("early_warning", "early_warning_system", "Early Warning System",
                   {"lead_time": "24-72 hours"})
    graph.add_node("permeable_pavement", "green_infrastructure", "Permeable Pavement",
                   {"infiltration": "reduces runoff by 80%"})
    
    # Add hazards
    graph.add_node("urban_heat", "heat_wave", "Urban Heat Island")
    graph.add_node("flood", "flood", "Urban Flooding")
    graph.add_node("storm_surge", "sea_level_rise", "Storm Surge")
    
    # Add locations
    graph.add_node("rotterdam", "urban_area", "Rotterdam")
    graph.add_node("singapore", "urban_area", "Singapore")
    graph.add_node("copenhagen", "coastal_area", "Copenhagen")
    
    # Add relationships
    graph.add_edge("green_roof", "MITIGATES", "urban_heat")
    graph.add_edge("green_roof", "MITIGATES", "flood")
    graph.add_edge("urban_forest", "MITIGATES", "urban_heat")
    graph.add_edge("rain_garden", "MITIGATES", "flood")
    graph.add_edge("flood_barrier", "MITIGATES", "flood")
    graph.add_edge("flood_barrier", "MITIGATES", "storm_surge")
    graph.add_edge("early_warning", "PROTECTS", "flood")
    graph.add_edge("permeable_pavement", "MITIGATES", "flood")
    
    graph.add_edge("green_roof", "IMPLEMENTED_IN", "rotterdam")
    graph.add_edge("green_roof", "IMPLEMENTED_IN", "singapore")
    graph.add_edge("urban_forest", "IMPLEMENTED_IN", "singapore")
    graph.add_edge("flood_barrier", "IMPLEMENTED_IN", "rotterdam")
    
    return graph


def main():
    print("=" * 60)
    print("OntoRAG: RAG Queries")
    print("=" * 60)
    
    # =========================================================================
    # STEP 1: Create Knowledge Graph
    # =========================================================================
    print("\n📚 Step 1: Creating knowledge graph...")
    
    graph = create_sample_graph()
    print(f"   Nodes: {graph.num_nodes}")
    print(f"   Edges: {graph.num_edges}")
    
    # =========================================================================
    # STEP 2: Set Up RAG Engine
    # =========================================================================
    print("\n⚙️ Step 2: Setting up RAG engine...")
    
    # Check for API keys
    openai_key = os.environ.get("OPENAI_API_KEY")
    anthropic_key = os.environ.get("ANTHROPIC_API_KEY")
    
    rag = None
    
    if openai_key:
        print("   Using OpenAI...")
        rag = RAGEngine(
            graph=graph,
            llm_provider="openai",
            api_key=openai_key,
            k=5,  # Retrieve top 5 nodes
            include_neighbors=True,
        )
    elif anthropic_key:
        print("   Using Anthropic Claude...")
        rag = RAGEngine(
            graph=graph,
            llm_provider="anthropic",
            api_key=anthropic_key,
            k=5,
        )
    else:
        print("   No API key found. Using context-only mode...")
        print("   Set OPENAI_API_KEY or ANTHROPIC_API_KEY for full RAG.")
        rag = RAGEngine(
            graph=graph,
            k=5,
        )
    
    # =========================================================================
    # STEP 3: Simple Queries
    # =========================================================================
    print("\n❓ Step 3: Running queries...")
    
    queries = [
        "What measures can mitigate urban heat?",
        "How can cities protect against flooding?",
        "What adaptation measures are implemented in Rotterdam?",
    ]
    
    for query in queries:
        print(f"\n   Query: {query}")
        print("-" * 50)
        
        response = rag.query(query)
        
        if rag._llm_fn:
            print(f"   Answer: {response.answer[:300]}...")
        else:
            print("   Context retrieved (no LLM configured):")
            for node in response.context.nodes[:3]:
                print(f"   - {node['label']} ({node['type']})")
        
        print(f"\n   Sources: {response.sources[:5]}")
        print(f"   Triplets: {len(response.context.triplets)}")
    
    # =========================================================================
    # STEP 4: Get Context Only
    # =========================================================================
    print("\n📋 Step 4: Getting context only...")
    
    response = rag.query(
        "What are green infrastructure solutions?",
        return_context_only=True
    )
    
    print(f"   Nodes retrieved: {len(response.context.nodes)}")
    print(f"   Triplets: {len(response.context.triplets)}")
    
    print("\n   Context text:")
    print("-" * 40)
    print(response.context.text[:500])
    
    # =========================================================================
    # STEP 5: Get Related Entities
    # =========================================================================
    print("\n\n🔗 Step 5: Finding related entities...")
    
    related = rag.get_related("flood mitigation", k=5)
    
    print("   Related to 'flood mitigation':")
    for item in related:
        print(f"   - {item['label']} ({item['type']}) - Score: {item['score']:.3f}")
    
    # =========================================================================
    # STEP 6: Custom System Prompt
    # =========================================================================
    print("\n✏️ Step 6: Using custom prompts...")
    
    custom_prompt = """You are a climate adaptation expert helping city planners.
    Use the knowledge graph context to provide practical, actionable advice.
    Focus on cost-effective solutions and implementation considerations.
    
    When discussing measures, mention:
    - Effectiveness (quantified if possible)
    - Implementation complexity
    - Co-benefits
    """
    
    if openai_key or anthropic_key:
        custom_rag = RAGEngine(
            graph=graph,
            llm_provider="openai" if openai_key else "anthropic",
            api_key=openai_key or anthropic_key,
            system_prompt=custom_prompt,
        )
        
        response = custom_rag.query("What's the best solution for urban heat in a budget-limited city?")
        print(f"   Custom response: {response.answer[:300]}...")
    else:
        print("   (Requires API key to demonstrate custom prompts)")
    
    # =========================================================================
    # STEP 7: Using Custom LLM Callable
    # =========================================================================
    print("\n🔧 Step 7: Custom LLM integration...")
    
    print("""
   You can provide any callable as the LLM:
   
   def my_llm(prompt: str) -> str:
       # Call your custom model
       return "My response..."
   
   rag = RAGEngine(
       graph=graph,
       llm_callable=my_llm
   )
    """)
    
    # =========================================================================
    # STEP 8: Response Analysis
    # =========================================================================
    print("\n📊 Step 8: Response analysis...")
    
    response = rag.query("What are flood protection measures?")
    
    print(f"   Query: {response.query}")
    print(f"   Confidence: {response.confidence:.3f}")
    print(f"   Sources used: {len(response.sources)}")
    print(f"   Context nodes: {len(response.context.nodes)}")
    print(f"   Context edges: {len(response.context.edges)}")
    print(f"   Context triplets: {len(response.context.triplets)}")
    
    # Export response
    response_dict = response.to_dict()
    print("\n   Response exported to dictionary with keys:")
    print(f"   {list(response_dict.keys())}")
    
    # =========================================================================
    # Summary
    # =========================================================================
    print("\n" + "=" * 60)
    print("✅ RAG queries complete!")
    print("=" * 60)
    
    print("""
Key concepts:
1. RAGEngine retrieves relevant nodes from the knowledge graph
2. Context includes nodes, edges, and triplets
3. The LLM generates answers based on retrieved context
4. You can customize prompts, context size, and LLM provider

Tips:
- Use k=5-10 for most queries
- Enable include_neighbors for richer context
- Custom prompts improve domain-specific responses
- Export response.context for debugging
    """)


if __name__ == "__main__":
    main()
