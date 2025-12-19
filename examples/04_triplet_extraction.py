#!/usr/bin/env python3
"""
OntoRAG Example: Triplet Extraction

This example shows how to:
1. Extract entities and triplets using rule-based methods
2. Extract using LLM (requires API key)
3. Combine both approaches with hybrid extraction
4. Process documents from a corpus

Prerequisites:
    $ pip install ontorag
    
For LLM extraction:
    $ pip install ontorag[openai]  # or [anthropic], [llm]

Run this example:
    $ python 04_triplet_extraction.py
"""

from ontorag.extraction import (
    RuleBasedExtractor,
    EntityDictionary,
    CLIMATE_ENTITIES,
    format_extraction_prompt,
)

# Optional LLM imports
try:
    from ontorag.extraction import LLMExtractor, HybridExtractor, LLMConfig
    HAS_LLM = True
except ImportError:
    HAS_LLM = False


def main():
    print("=" * 60)
    print("OntoRAG: Triplet Extraction")
    print("=" * 60)
    
    # Sample texts for extraction
    sample_texts = [
        """Green roofs can reduce urban heat island effects by 2-3°C. 
        Rotterdam has implemented extensive green roof programs to mitigate 
        flood risk and heat stress. The city's approach combines grey and 
        green infrastructure for comprehensive climate adaptation.""",
        
        """Early warning systems protect communities from floods by providing 
        advance notice. In Bangladesh, flood forecasting systems have reduced 
        mortality by 90% during major flood events.""",
        
        """Mangrove restoration along coastal areas prevents erosion and 
        protects against storm surge. Studies show mangroves can reduce 
        wave energy by 70% within 100 meters of shoreline.""",
    ]
    
    # =========================================================================
    # STEP 1: Rule-Based Extraction
    # =========================================================================
    print("\n📏 Step 1: Rule-based extraction...")
    
    # Create extractor with default climate dictionary
    rule_extractor = RuleBasedExtractor()
    
    # Show dictionary stats
    dict_stats = rule_extractor.entity_dictionary.stats()
    print(f"\n   Entity dictionary:")
    for entity_type, count in dict_stats.items():
        print(f"   - {entity_type}: {count} terms")
    
    # Extract from first text
    print(f"\n   Processing sample text...")
    result = rule_extractor.extract(sample_texts[0])
    
    print(f"\n   Entities found: {len(result.entities)}")
    for entity in result.entities:
        print(f"   - [{entity.type}] {entity.text}")
    
    print(f"\n   Triplets found: {len(result.triplets)}")
    for triplet in result.triplets[:5]:
        print(f"   - ({triplet.subject}, {triplet.relation}, {triplet.object})")
        if triplet.evidence:
            print(f"     Evidence: {triplet.evidence[:80]}...")
    
    # =========================================================================
    # STEP 2: Process Multiple Texts
    # =========================================================================
    print("\n📄 Step 2: Processing multiple texts...")
    
    all_entities = []
    all_triplets = []
    
    for i, text in enumerate(sample_texts, 1):
        result = rule_extractor.extract(text)
        all_entities.extend(result.entities)
        all_triplets.extend(result.triplets)
        print(f"   Text {i}: {len(result.entities)} entities, {len(result.triplets)} triplets")
    
    print(f"\n   Total: {len(all_entities)} entities, {len(all_triplets)} triplets")
    
    # =========================================================================
    # STEP 3: Entity Type Summary
    # =========================================================================
    print("\n📊 Step 3: Entity summary...")
    
    entity_by_type = {}
    for entity in all_entities:
        if entity.type not in entity_by_type:
            entity_by_type[entity.type] = set()
        entity_by_type[entity.type].add(entity.text.lower())
    
    for etype, entities in entity_by_type.items():
        print(f"\n   {etype}:")
        for e in list(entities)[:5]:
            print(f"   - {e}")
    
    # =========================================================================
    # STEP 4: Relation Summary
    # =========================================================================
    print("\n🔗 Step 4: Relation summary...")
    
    relation_counts = {}
    for triplet in all_triplets:
        relation_counts[triplet.relation] = relation_counts.get(triplet.relation, 0) + 1
    
    for relation, count in sorted(relation_counts.items(), key=lambda x: -x[1]):
        print(f"   - {relation}: {count}")
    
    # =========================================================================
    # STEP 5: Custom Dictionary
    # =========================================================================
    print("\n🔧 Step 5: Using custom dictionary...")
    
    # Create a minimal custom dictionary
    custom_dict = {
        "TECHNOLOGY": ["AI", "machine learning", "deep learning", "neural network"],
        "APPLICATION": ["image recognition", "NLP", "prediction", "classification"],
        "METRIC": ["accuracy", "precision", "recall", "F1 score"],
    }
    
    custom_extractor = RuleBasedExtractor(entity_dict=custom_dict)
    
    tech_text = """Machine learning models achieve 95% accuracy in flood prediction.
    Deep learning approaches outperform traditional methods in image recognition
    of weather patterns."""
    
    tech_result = custom_extractor.extract(tech_text)
    
    print(f"   Custom extraction results:")
    for entity in tech_result.entities:
        print(f"   - [{entity.type}] {entity.text}")
    
    # =========================================================================
    # STEP 6: LLM-Based Extraction (Optional)
    # =========================================================================
    print("\n🤖 Step 6: LLM-based extraction...")
    
    if HAS_LLM:
        print("""
   LLM extraction is available. To use it:
   
   # Using OpenAI
   extractor = LLMExtractor(
       provider="openai",
       api_key="sk-your-api-key"
   )
   result = extractor.extract(text)
   
   # Using Anthropic
   extractor = LLMExtractor(
       provider="anthropic",
       api_key="your-api-key"
   )
   
   # Hybrid (rule + LLM)
   hybrid = HybridExtractor(
       llm_provider="openai",
       llm_api_key="sk-..."
   )
   result = hybrid.extract(text)
        """)
        
        # Show prompt template
        print("   Sample extraction prompt:")
        print("-" * 40)
        prompt = format_extraction_prompt(sample_texts[0][:200] + "...")
        print(prompt[:500] + "...")
    else:
        print("""
   LLM extraction requires additional packages:
   
   $ pip install ontorag[openai]     # For OpenAI
   $ pip install ontorag[anthropic]  # For Anthropic
   $ pip install ontorag[llm]        # For all LLM providers
        """)
    
    # =========================================================================
    # STEP 7: Extract from Document Dict
    # =========================================================================
    print("\n📑 Step 7: Extract from document format...")
    
    # Simulate a document from corpus
    document = {
        "id": "doc_001",
        "source": "arxiv",
        "title": "Urban Heat Island Mitigation through Green Infrastructure",
        "abstract": """This study examines how green roofs and urban forests can 
        mitigate heat wave impacts in metropolitan areas. Results show temperature 
        reductions of 2-4°C in areas with extensive green infrastructure 
        implementation."""
    }
    
    doc_result = rule_extractor.extract_from_document(document)
    
    print(f"   Document: {document['title'][:50]}...")
    print(f"   Entities: {len(doc_result.entities)}")
    print(f"   Triplets: {len(doc_result.triplets)}")
    
    # =========================================================================
    # STEP 8: Export Results
    # =========================================================================
    print("\n💾 Step 8: Exporting results...")
    
    # Convert to dictionary for JSON export
    export_data = {
        "entities": [e.to_dict() for e in all_entities],
        "triplets": [t.to_dict() for t in all_triplets],
    }
    
    import json
    with open("extraction_results.json", "w") as f:
        json.dump(export_data, f, indent=2)
    print("   Saved to: extraction_results.json")
    
    # =========================================================================
    # Summary
    # =========================================================================
    print("\n" + "=" * 60)
    print("📊 Extraction Summary")
    print("=" * 60)
    
    print(f"""
   Texts processed: {len(sample_texts)}
   Total entities: {len(all_entities)}
   Total triplets: {len(all_triplets)}
   
   Unique entities by type:""")
    
    for etype, entities in entity_by_type.items():
        print(f"   - {etype}: {len(entities)}")
    
    print(f"""
   Top triplet patterns:""")
    for relation, count in sorted(relation_counts.items(), key=lambda x: -x[1])[:5]:
        print(f"   - {relation}: {count}")
    
    print("\n" + "=" * 60)
    print("✅ Triplet extraction complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
