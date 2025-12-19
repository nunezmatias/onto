"""
Tests for OntoRAG Extraction Module
"""

import pytest


def test_entity_dictionary_creation():
    """Test creating an entity dictionary."""
    from ontorag.extraction import EntityDictionary
    
    entities = {
        "DRUG": ["aspirin", "ibuprofen"],
        "SYMPTOM": ["headache", "fever"]
    }
    
    dictionary = EntityDictionary(entities)
    
    assert dictionary.get_entity_types() == ["DRUG", "SYMPTOM"]
    assert "aspirin" in dictionary.get_terms("DRUG")


def test_entity_dictionary_matching():
    """Test finding entities in text."""
    from ontorag.extraction import EntityDictionary
    
    entities = {
        "MEASURE": ["green roof", "rain garden"],
        "HAZARD": ["flood", "drought"]
    }
    
    dictionary = EntityDictionary(entities)
    
    matches = dictionary.find_entities(
        "Green roofs can help with flood mitigation"
    )
    
    assert len(matches) >= 2
    types = [m.entity_type for m in matches]
    assert "MEASURE" in types
    assert "HAZARD" in types


def test_climate_entities():
    """Test predefined climate entities dictionary."""
    from ontorag.extraction import CLIMATE_ENTITIES, EntityDictionary
    
    assert "MEASURE" in CLIMATE_ENTITIES
    assert "HAZARD" in CLIMATE_ENTITIES
    assert "green roof" in CLIMATE_ENTITIES["MEASURE"]
    assert "flood" in CLIMATE_ENTITIES["HAZARD"]


def test_rule_based_extractor():
    """Test rule-based extraction."""
    from ontorag.extraction import RuleBasedExtractor
    
    extractor = RuleBasedExtractor()
    
    result = extractor.extract(
        "Green roofs reduce urban heat island effects"
    )
    
    assert len(result.entities) > 0
    assert any(e.type == "MEASURE" for e in result.entities)


def test_extraction_result_structure():
    """Test extraction result data structure."""
    from ontorag.extraction import ExtractionResult, ExtractedEntity, ExtractedTriplet
    
    result = ExtractionResult()
    
    result.entities.append(ExtractedEntity(
        text="green roof",
        type="MEASURE"
    ))
    
    result.triplets.append(ExtractedTriplet(
        subject="green_roof",
        relation="MITIGATES",
        object="urban_heat"
    ))
    
    assert len(result.entities) == 1
    assert len(result.triplets) == 1
    
    # Test serialization
    data = result.to_dict()
    assert "entities" in data
    assert "triplets" in data


def test_extracted_entity():
    """Test ExtractedEntity data class."""
    from ontorag.extraction import ExtractedEntity
    
    entity = ExtractedEntity(
        text="Green Roof",
        type="MEASURE",
        confidence=0.9
    )
    
    # Normalized form should be computed
    assert entity.normalized == "green_roof"
    
    data = entity.to_dict()
    assert data["text"] == "Green Roof"
    assert data["type"] == "MEASURE"


def test_extracted_triplet():
    """Test ExtractedTriplet data class."""
    from ontorag.extraction import ExtractedTriplet
    
    triplet = ExtractedTriplet(
        subject="green_roof",
        relation="MITIGATES",
        object="flood"
    )
    
    assert triplet.to_tuple() == ("green_roof", "MITIGATES", "flood")
    
    data = triplet.to_dict()
    assert data["subject"] == "green_roof"
    assert data["relation"] == "MITIGATES"
    assert data["object"] == "flood"


def test_extractor_with_custom_dictionary():
    """Test extraction with custom entity dictionary."""
    from ontorag.extraction import RuleBasedExtractor
    
    custom_dict = {
        "FRUIT": ["apple", "banana", "orange"],
        "COLOR": ["red", "yellow", "blue"]
    }
    
    extractor = RuleBasedExtractor(entity_dict=custom_dict)
    
    result = extractor.extract("I like red apples and yellow bananas")
    
    types = [e.type for e in result.entities]
    assert "FRUIT" in types
    assert "COLOR" in types


def test_quantity_extraction():
    """Test extraction of quantities."""
    from ontorag.extraction import RuleBasedExtractor
    
    extractor = RuleBasedExtractor(extract_quantities=True)
    
    result = extractor.extract(
        "Temperature reduction of 2-3°C and 50% energy savings"
    )
    
    quantities = [e for e in result.entities if e.type == "QUANTITY"]
    assert len(quantities) >= 2


def test_document_extraction():
    """Test extraction from document dict."""
    from ontorag.extraction import RuleBasedExtractor
    
    extractor = RuleBasedExtractor()
    
    document = {
        "id": "doc1",
        "title": "Green Infrastructure for Flood Control",
        "abstract": "Rain gardens mitigate urban flooding effectively."
    }
    
    result = extractor.extract_from_document(document)
    
    assert len(result.entities) > 0
    assert "document_id" in result.metadata


def test_prompt_formatting():
    """Test extraction prompt formatting."""
    from ontorag.extraction import format_extraction_prompt, EXTRACTION_PROMPT
    
    text = "Green roofs reduce heat."
    
    prompt = format_extraction_prompt(text)
    assert text in prompt
    
    # Custom entity types
    prompt = format_extraction_prompt(
        text,
        entity_types=["CUSTOM_TYPE"]
    )
    assert "CUSTOM_TYPE" in prompt


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
