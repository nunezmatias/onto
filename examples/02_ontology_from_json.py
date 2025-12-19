#!/usr/bin/env python3
"""
OntoRAG Example: Loading Ontology from JSON

This example shows how to:
1. Load an ontology from a JSON file
2. Create an ontology programmatically and save it
3. Validate ontology consistency

Run this example:
    $ python 02_ontology_from_json.py
"""

import json
from pathlib import Path

from ontorag import (
    Ontology,
    OntologyClass,
    OntologyRelation,
    RelationType,
    load_ontology_from_json,
    create_climate_ontology,
)


def main():
    print("=" * 60)
    print("OntoRAG: Working with Ontologies")
    print("=" * 60)
    
    # =========================================================================
    # OPTION 1: Use Predefined Ontology
    # =========================================================================
    print("\n📚 Option 1: Using predefined climate ontology...")
    
    climate_onto = create_climate_ontology()
    print(f"   Name: {climate_onto.name}")
    print(f"   Classes: {len(climate_onto.classes)}")
    print(f"   Relations: {len(climate_onto.relations)}")
    
    # Show some classes
    print("\n   Sample classes:")
    for cls_id in list(climate_onto.classes.keys())[:5]:
        cls = climate_onto.get_class(cls_id)
        print(f"   - {cls.name} (id: {cls_id})")
        if cls.synonyms:
            print(f"     Synonyms: {', '.join(cls.synonyms[:3])}")
    
    # =========================================================================
    # OPTION 2: Load from JSON File
    # =========================================================================
    print("\n📄 Option 2: Loading ontology from JSON...")
    
    # Check if the JSON file exists in the ontologies folder
    json_path = Path(__file__).parent.parent / "ontologies" / "climate_adaptation.json"
    
    if json_path.exists():
        loaded_onto = load_ontology_from_json(json_path)
        print(f"   Loaded: {loaded_onto.name}")
        print(f"   Classes: {len(loaded_onto.classes)}")
        print(f"   Relations: {len(loaded_onto.relations)}")
    else:
        print(f"   File not found: {json_path}")
        print("   Creating a sample JSON ontology...")
        
        # Create a simple ontology and save it
        sample_onto = Ontology("sample", "Sample Ontology")
        sample_onto.add_class(OntologyClass(id="entity", name="Entity"))
        sample_onto.add_class(OntologyClass(
            id="person", name="Person", parent="entity",
            synonyms=["individual", "human"]
        ))
        sample_onto.add_relation(OntologyRelation(
            id="KNOWS", name="knows",
            relation_type=RelationType.ASSOCIATIVE
        ))
        
        sample_onto.save("sample_ontology.json")
        print("   Saved to: sample_ontology.json")
    
    # =========================================================================
    # OPTION 3: Create Ontology Programmatically
    # =========================================================================
    print("\n🔨 Option 3: Creating ontology programmatically...")
    
    # Create a biomedical ontology as an example
    bio_onto = Ontology(
        name="biomedical",
        description="Simple biomedical ontology for drug-disease relations",
        version="1.0.0"
    )
    
    # Add root class
    bio_onto.add_class(OntologyClass(
        id="entity",
        name="Entity",
        description="Root class"
    ))
    
    # Add disease classes
    bio_onto.add_class(OntologyClass(
        id="disease",
        name="Disease",
        description="Medical condition",
        parent="entity",
        synonyms=["condition", "disorder", "illness"]
    ))
    
    bio_onto.add_class(OntologyClass(
        id="infectious_disease",
        name="Infectious Disease",
        parent="disease",
        synonyms=["infection", "communicable disease"]
    ))
    
    bio_onto.add_class(OntologyClass(
        id="chronic_disease",
        name="Chronic Disease",
        parent="disease",
        synonyms=["chronic condition"]
    ))
    
    # Add drug classes
    bio_onto.add_class(OntologyClass(
        id="drug",
        name="Drug",
        description="Pharmaceutical compound",
        parent="entity",
        synonyms=["medication", "medicine", "pharmaceutical"]
    ))
    
    bio_onto.add_class(OntologyClass(
        id="antibiotic",
        name="Antibiotic",
        parent="drug",
        synonyms=["antimicrobial"]
    ))
    
    # Add symptom class
    bio_onto.add_class(OntologyClass(
        id="symptom",
        name="Symptom",
        parent="entity",
        synonyms=["sign", "manifestation"]
    ))
    
    # Add relations
    bio_onto.add_relation(OntologyRelation(
        id="TREATS",
        name="treats",
        description="Drug treats disease",
        relation_type=RelationType.MITIGATIVE,
        domain=["drug"],
        range=["disease"],
        synonyms=["cures", "heals", "alleviates"]
    ))
    
    bio_onto.add_relation(OntologyRelation(
        id="CAUSES",
        name="causes",
        description="Disease causes symptom",
        relation_type=RelationType.CAUSAL,
        domain=["disease"],
        range=["symptom"],
        synonyms=["leads to", "results in"]
    ))
    
    bio_onto.add_relation(OntologyRelation(
        id="PREVENTS",
        name="prevents",
        description="Drug prevents disease",
        relation_type=RelationType.PROTECTIVE,
        domain=["drug"],
        range=["disease"]
    ))
    
    print(f"   Created: {bio_onto.name}")
    print(f"   Classes: {len(bio_onto.classes)}")
    print(f"   Relations: {len(bio_onto.relations)}")
    
    # Save the ontology
    bio_onto.save("biomedical_ontology.json")
    print("   Saved to: biomedical_ontology.json")
    
    # =========================================================================
    # Validate Ontology
    # =========================================================================
    print("\n✅ Validating ontologies...")
    
    for onto in [climate_onto, bio_onto]:
        result = onto.validate()
        status = "✓ Valid" if result.is_valid else "✗ Invalid"
        print(f"\n   {onto.name}: {status}")
        
        if result.errors:
            for error in result.errors:
                print(f"     Error: {error}")
        
        if result.warnings:
            for warning in result.warnings[:3]:
                print(f"     Warning: {warning}")
    
    # =========================================================================
    # Entity Normalization
    # =========================================================================
    print("\n🔄 Testing entity normalization...")
    
    test_terms = ["flooding", "heat wave", "green roofs", "drought"]
    print(f"\n   Using climate ontology:")
    
    for term in test_terms:
        normalized = climate_onto.normalize_entity(term)
        print(f"   '{term}' → '{normalized or 'NOT FOUND'}'")
    
    # =========================================================================
    # Relation Normalization
    # =========================================================================
    print("\n🔄 Testing relation normalization...")
    
    test_relations = ["reduces", "causes", "prevents", "located in"]
    
    for rel in test_relations:
        normalized = climate_onto.normalize_relation(rel)
        print(f"   '{rel}' → '{normalized or 'NOT FOUND'}'")
    
    # =========================================================================
    # Class Hierarchy
    # =========================================================================
    print("\n🌳 Exploring class hierarchy...")
    
    print("\n   Climate ontology - 'flood' hierarchy:")
    hierarchy = climate_onto.get_class_hierarchy("flood")
    print(f"   {' → '.join(hierarchy)}")
    
    print("\n   Biomedical ontology - subclasses of 'disease':")
    subclasses = bio_onto.get_subclasses("disease")
    for sc in subclasses:
        cls = bio_onto.get_class(sc)
        print(f"   - {cls.name}")
    
    print("\n" + "=" * 60)
    print("✅ Ontology examples complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
