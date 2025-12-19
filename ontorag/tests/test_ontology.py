"""
Tests para el módulo de ontología
"""

import pytest
import json
import tempfile
from pathlib import Path

# Importar desde el paquete
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from ontorag.ontology import (
    Ontology, 
    OntologyClass, 
    OntologyRelation, 
    RelationType,
    create_climate_adaptation_ontology
)


class TestOntologyClass:
    """Tests para OntologyClass"""
    
    def test_create_basic_class(self):
        cls = OntologyClass(
            id="test_class",
            name="Test Class",
            description="A test class"
        )
        assert cls.id == "test_class"
        assert cls.name == "Test Class"
        assert "Test Class" in cls.synonyms
    
    def test_id_normalization(self):
        cls = OntologyClass(
            id="Test Class With Spaces",
            name="Test"
        )
        assert cls.id == "test_class_with_spaces"
    
    def test_matches_synonym(self):
        cls = OntologyClass(
            id="flood",
            name="Inundación",
            synonyms=["inundaciones", "flooding"]
        )
        assert cls.matches("inundación")
        assert cls.matches("INUNDACIONES")
        assert cls.matches("flooding")
        assert not cls.matches("earthquake")


class TestOntologyRelation:
    """Tests para OntologyRelation"""
    
    def test_create_relation(self):
        rel = OntologyRelation(
            id="mitigates",
            name="mitiga",
            description="Reduces effect",
            relation_type=RelationType.MITIGATIVE
        )
        assert rel.id == "MITIGATES"
        assert rel.relation_type == RelationType.MITIGATIVE
        assert "mitiga" in rel.synonyms


class TestOntology:
    """Tests para Ontology"""
    
    @pytest.fixture
    def basic_ontology(self):
        """Crea una ontología básica para testing"""
        onto = Ontology("test_ontology")
        
        onto.add_class(OntologyClass(
            id="hazard",
            name="Peligro",
            description="Base class for hazards"
        ))
        
        onto.add_class(OntologyClass(
            id="flood",
            name="Inundación",
            parent="hazard",
            synonyms=["flooding", "inundaciones"]
        ))
        
        onto.add_class(OntologyClass(
            id="measure",
            name="Medida",
            description="Base class for measures"
        ))
        
        onto.add_relation(OntologyRelation(
            id="MITIGATES",
            name="mitiga",
            description="Reduces effect of",
            relation_type=RelationType.MITIGATIVE,
            domain=["measure"],
            range=["hazard"],
            synonyms=["reduce", "disminuye"]
        ))
        
        return onto
    
    def test_add_class(self, basic_ontology):
        assert "flood" in basic_ontology.classes
        assert basic_ontology.classes["flood"].parent == "hazard"
    
    def test_add_class_duplicate_raises(self, basic_ontology):
        with pytest.raises(ValueError):
            basic_ontology.add_class(OntologyClass(
                id="flood",
                name="Duplicate"
            ))
    
    def test_add_class_missing_parent_raises(self, basic_ontology):
        with pytest.raises(ValueError):
            basic_ontology.add_class(OntologyClass(
                id="orphan",
                name="Orphan",
                parent="nonexistent"
            ))
    
    def test_get_class_hierarchy(self, basic_ontology):
        hierarchy = basic_ontology.get_class_hierarchy("flood")
        assert hierarchy == ["flood", "hazard"]
    
    def test_is_subclass_of(self, basic_ontology):
        assert basic_ontology.is_subclass_of("flood", "hazard")
        assert not basic_ontology.is_subclass_of("hazard", "flood")
    
    def test_normalize_entity(self, basic_ontology):
        assert basic_ontology.normalize_entity("inundación") == "flood"
        assert basic_ontology.normalize_entity("flooding") == "flood"
        assert basic_ontology.normalize_entity("earthquake") is None
    
    def test_normalize_relation(self, basic_ontology):
        assert basic_ontology.normalize_relation("mitiga") == "MITIGATES"
        assert basic_ontology.normalize_relation("reduce") == "MITIGATES"
        assert basic_ontology.normalize_relation("unknown") is None
    
    def test_validate_valid_ontology(self, basic_ontology):
        result = basic_ontology.validate()
        assert result.is_valid
    
    def test_save_and_load(self, basic_ontology):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "ontology.json"
            basic_ontology.save(str(path))
            
            loaded = Ontology.load(str(path))
            
            assert loaded.name == basic_ontology.name
            assert len(loaded.classes) == len(basic_ontology.classes)
            assert len(loaded.relations) == len(basic_ontology.relations)
            assert "flood" in loaded.classes
    
    def test_to_dict_and_from_dict(self, basic_ontology):
        data = basic_ontology.to_dict()
        
        assert data["name"] == "test_ontology"
        assert "flood" in data["classes"]
        assert "MITIGATES" in data["relations"]
        
        restored = Ontology.from_dict(data)
        assert restored.name == basic_ontology.name


class TestPredefinedOntology:
    """Tests para ontologías predefinidas"""
    
    def test_create_climate_ontology(self):
        onto = create_climate_adaptation_ontology()
        
        assert onto.name == "climate_adaptation"
        assert len(onto.classes) > 5
        assert len(onto.relations) > 3
        
        # Verificar algunas clases esperadas
        assert "climate_hazard" in onto.classes
        assert "adaptation_measure" in onto.classes
        assert "flood" in onto.classes
        
        # Verificar jerarquía
        assert onto.is_subclass_of("flood", "climate_hazard")
        
        # Verificar relaciones
        assert "MITIGATES" in onto.relations


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
