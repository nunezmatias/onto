"""
Tests for OntoRAG Ontology Module
"""

import json
import tempfile
import pytest
from pathlib import Path


def test_ontology_class_creation():
    """Test creating an ontology class."""
    from ontorag import OntologyClass
    
    cls = OntologyClass(
        id="test_class",
        name="Test Class",
        description="A test class",
        synonyms=["test", "example"]
    )
    
    assert cls.id == "test_class"
    assert cls.name == "Test Class"
    assert "test" in cls.synonyms
    assert cls.parent is None


def test_ontology_class_normalization():
    """Test ID normalization."""
    from ontorag import OntologyClass
    
    cls = OntologyClass(id="Test-Class Name", name="Test")
    assert cls.id == "test_class_name"


def test_ontology_class_matching():
    """Test synonym matching."""
    from ontorag import OntologyClass
    
    cls = OntologyClass(
        id="flood",
        name="Flood",
        synonyms=["flooding", "inundation"]
    )
    
    is_match, score = cls.matches("flooding")
    assert is_match
    assert score == 1.0
    
    is_match, score = cls.matches("earthquake")
    assert not is_match


def test_ontology_relation_creation():
    """Test creating an ontology relation."""
    from ontorag import OntologyRelation
    
    rel = OntologyRelation(
        id="mitigates",
        name="mitigates",
        description="Reduces impact",
        relation_type="mitigative"
    )
    
    assert rel.id == "MITIGATES"  # Normalized to uppercase
    assert rel.name == "mitigates"


def test_ontology_creation():
    """Test creating an ontology."""
    from ontorag import Ontology, OntologyClass, OntologyRelation
    
    onto = Ontology("test", "Test Ontology")
    
    onto.add_class(OntologyClass(id="entity", name="Entity"))
    onto.add_class(OntologyClass(id="thing", name="Thing", parent="entity"))
    onto.add_relation(OntologyRelation(id="RELATES", name="relates"))
    
    assert len(onto.classes) == 2
    assert len(onto.relations) == 1
    assert onto.get_class("entity") is not None


def test_ontology_hierarchy():
    """Test class hierarchy."""
    from ontorag import Ontology, OntologyClass
    
    onto = Ontology("test", "Test")
    onto.add_class(OntologyClass(id="a", name="A"))
    onto.add_class(OntologyClass(id="b", name="B", parent="a"))
    onto.add_class(OntologyClass(id="c", name="C", parent="b"))
    
    hierarchy = onto.get_class_hierarchy("c")
    assert hierarchy == ["c", "b", "a"]
    
    assert onto.is_subclass_of("c", "a")
    assert not onto.is_subclass_of("a", "c")


def test_ontology_validation():
    """Test ontology validation."""
    from ontorag import Ontology, OntologyClass, OntologyRelation
    
    onto = Ontology("test", "Test")
    onto.add_class(OntologyClass(id="entity", name="Entity"))
    onto.add_relation(OntologyRelation(
        id="AFFECTS",  # Vague name
        name="affects",
        domain=["nonexistent"]  # Invalid reference
    ))
    
    result = onto.validate()
    assert len(result.warnings) > 0


def test_ontology_serialization():
    """Test saving and loading ontology."""
    from ontorag import Ontology, OntologyClass, OntologyRelation
    
    onto = Ontology("test", "Test Ontology", "1.0.0")
    onto.add_class(OntologyClass(id="entity", name="Entity"))
    onto.add_class(OntologyClass(id="thing", name="Thing", parent="entity"))
    onto.add_relation(OntologyRelation(id="RELATES", name="relates"))
    
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        path = f.name
    
    try:
        onto.save(path)
        loaded = Ontology.load(path)
        
        assert loaded.name == onto.name
        assert len(loaded.classes) == len(onto.classes)
        assert len(loaded.relations) == len(onto.relations)
    finally:
        Path(path).unlink()


def test_ontology_normalization():
    """Test entity and relation normalization."""
    from ontorag import create_climate_ontology
    
    onto = create_climate_ontology()
    
    # Entity normalization
    assert onto.normalize_entity("flooding") == "flood"
    assert onto.normalize_entity("heatwave") == "heat_wave"
    assert onto.normalize_entity("xyz123") is None
    
    # Relation normalization
    assert onto.normalize_relation("reduces") == "MITIGATES"
    assert onto.normalize_relation("causes") == "CAUSES"


def test_climate_ontology():
    """Test predefined climate ontology."""
    from ontorag import create_climate_ontology
    
    onto = create_climate_ontology()
    
    assert onto.name == "climate_adaptation"
    assert len(onto.classes) > 10
    assert len(onto.relations) > 5
    
    # Check some expected classes exist
    assert onto.get_class("flood") is not None
    assert onto.get_class("drought") is not None
    assert onto.get_class("green_roof") is not None
    
    # Check relations
    assert onto.get_relation("MITIGATES") is not None
    assert onto.get_relation("CAUSES") is not None


def test_load_from_json():
    """Test loading ontology from JSON."""
    from ontorag import load_ontology_from_json
    
    json_data = {
        "name": "test",
        "description": "Test ontology",
        "version": "1.0.0",
        "classes": {
            "entity": {"name": "Entity"},
            "thing": {"name": "Thing", "parent": "entity"}
        },
        "relations": {
            "RELATES": {"name": "relates"}
        }
    }
    
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode='w') as f:
        json.dump(json_data, f)
        path = f.name
    
    try:
        onto = load_ontology_from_json(path)
        assert onto.name == "test"
        assert len(onto.classes) == 2
    finally:
        Path(path).unlink()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
