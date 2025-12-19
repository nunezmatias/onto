"""
═══════════════════════════════════════════════════════════════════════════════
Tests para OntoRAG
═══════════════════════════════════════════════════════════════════════════════

Ejecutar con: pytest tests/ -v
"""

import pytest
import json
import tempfile
import os
import sys

# Agregar path del proyecto
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ontorag import (
    Ontology,
    OntologyClass,
    OntologyRelation,
    RelationType,
    ValidationResult,
)


# ═══════════════════════════════════════════════════════════════════════════════
# TESTS DE ONTOLOGYCLASS
# ═══════════════════════════════════════════════════════════════════════════════

class TestOntologyClass:
    """Tests para OntologyClass."""
    
    def test_create_basic_class(self):
        """Crear clase básica."""
        cls = OntologyClass(
            id="person",
            name="Persona",
            description="Una persona humana"
        )
        
        assert cls.id == "person"
        assert cls.name == "Persona"
        assert cls.description == "Una persona humana"
        assert cls.parent is None
    
    def test_class_id_normalization(self):
        """IDs se normalizan a snake_case."""
        cls = OntologyClass(id="My Class Name", name="Test")
        assert cls.id == "my_class_name"
        
        cls2 = OntologyClass(id="UPPER CASE", name="Test")
        assert cls2.id == "upper_case"
    
    def test_synonyms_include_name(self):
        """El nombre se incluye en sinónimos."""
        cls = OntologyClass(
            id="flood",
            name="Inundación",
            synonyms=["flooding", "riada"]
        )
        
        assert "inundación" in cls.synonyms
        assert "flooding" in cls.synonyms
        assert "riada" in cls.synonyms
    
    def test_class_matches(self):
        """Test de coincidencia de texto."""
        cls = OntologyClass(
            id="flood",
            name="Inundación",
            synonyms=["flooding", "riada", "anegamiento"]
        )
        
        # Coincidencia exacta
        matches, score = cls.matches("inundación")
        assert matches is True
        assert score == 1.0
        
        # Coincidencia parcial
        matches, score = cls.matches("inundaciones severas")
        assert matches is True
        assert score == 0.8
        
        # Sin coincidencia
        matches, score = cls.matches("sequía")
        assert matches is False
        assert score == 0.0
    
    def test_class_serialization(self):
        """Serialización y deserialización."""
        cls = OntologyClass(
            id="test",
            name="Test Class",
            description="A test",
            parent="parent_class",
            synonyms=["test", "prueba"],
            properties={"key": "value"}
        )
        
        data = cls.to_dict()
        restored = OntologyClass.from_dict(data)
        
        assert restored.id == cls.id
        assert restored.name == cls.name
        assert restored.parent == cls.parent
        assert restored.properties == cls.properties


# ═══════════════════════════════════════════════════════════════════════════════
# TESTS DE ONTOLOGYRELATION
# ═══════════════════════════════════════════════════════════════════════════════

class TestOntologyRelation:
    """Tests para OntologyRelation."""
    
    def test_create_relation(self):
        """Crear relación básica."""
        rel = OntologyRelation(
            id="MITIGATES",
            name="mitiga",
            description="El sujeto reduce el efecto del objeto",
            relation_type=RelationType.MITIGATIVE
        )
        
        assert rel.id == "MITIGATES"
        assert rel.relation_type == RelationType.MITIGATIVE
        assert rel.is_directional is True
    
    def test_relation_id_normalization(self):
        """IDs se normalizan a UPPER_CASE."""
        rel = OntologyRelation(
            id="works for",
            name="trabaja para",
            description="...",
            relation_type=RelationType.FUNCTIONAL
        )
        
        assert rel.id == "WORKS_FOR"
    
    def test_vague_relation_detection(self):
        """Detecta relaciones semánticamente vacías."""
        vague = OntologyRelation(
            id="AFFECTS",
            name="afecta",
            description="...",
            relation_type=RelationType.ASSOCIATIVE
        )
        assert vague.is_vague is True
        
        specific = OntologyRelation(
            id="MITIGATES",
            name="mitiga",
            description="...",
            relation_type=RelationType.MITIGATIVE
        )
        assert specific.is_vague is False
    
    def test_relation_matches(self):
        """Test de coincidencia de relación."""
        rel = OntologyRelation(
            id="MITIGATES",
            name="mitiga",
            description="...",
            relation_type=RelationType.MITIGATIVE,
            synonyms=["reduce", "atenúa", "disminuye"]
        )
        
        assert rel.matches("mitiga")[0] is True
        assert rel.matches("reduce")[0] is True
        assert rel.matches("causa")[0] is False
    
    def test_validate_triple(self):
        """Validación de tripletas contra dominio/rango."""
        rel = OntologyRelation(
            id="MITIGATES",
            name="mitiga",
            description="...",
            relation_type=RelationType.MITIGATIVE,
            domain=["adaptation_measure"],
            range=["climate_hazard"]
        )
        
        # Tripleta válida
        is_valid, msg = rel.validate_triple(
            "adaptation_measure",
            "climate_hazard",
            {"adaptation_measure": [], "climate_hazard": []}
        )
        assert is_valid is True
        
        # Tripleta inválida (dominio incorrecto)
        is_valid, msg = rel.validate_triple(
            "climate_hazard",
            "climate_hazard",
            {"climate_hazard": []}
        )
        assert is_valid is False


# ═══════════════════════════════════════════════════════════════════════════════
# TESTS DE ONTOLOGY
# ═══════════════════════════════════════════════════════════════════════════════

class TestOntology:
    """Tests para Ontology."""
    
    def test_create_ontology(self):
        """Crear ontología básica."""
        onto = Ontology("test", "Test ontology")
        
        assert onto.name == "test"
        assert onto.description == "Test ontology"
        assert len(onto.classes) == 0
        assert len(onto.relations) == 0
    
    def test_add_class(self):
        """Agregar clases."""
        onto = Ontology("test")
        
        onto.add_class(OntologyClass(id="entity", name="Entity"))
        assert len(onto.classes) == 1
        assert "entity" in onto.classes
        
        # Agregar subclase
        onto.add_class(OntologyClass(id="person", name="Person", parent="entity"))
        assert onto.classes["person"].parent == "entity"
    
    def test_add_class_with_nonexistent_parent_fails(self):
        """Error al agregar clase con padre inexistente."""
        onto = Ontology("test")
        
        with pytest.raises(ValueError):
            onto.add_class(OntologyClass(id="person", name="Person", parent="entity"))
    
    def test_add_duplicate_class_fails(self):
        """Error al agregar clase duplicada."""
        onto = Ontology("test")
        onto.add_class(OntologyClass(id="entity", name="Entity"))
        
        with pytest.raises(ValueError):
            onto.add_class(OntologyClass(id="entity", name="Entity 2"))
    
    def test_class_hierarchy(self):
        """Test de jerarquías de clases."""
        onto = Ontology("test")
        onto.add_class(OntologyClass(id="entity", name="Entity"))
        onto.add_class(OntologyClass(id="living", name="Living", parent="entity"))
        onto.add_class(OntologyClass(id="person", name="Person", parent="living"))
        
        hierarchy = onto.get_class_hierarchy("person")
        assert hierarchy == ["person", "living", "entity"]
        
        assert onto.is_subclass_of("person", "entity") is True
        assert onto.is_subclass_of("entity", "person") is False
    
    def test_get_subclasses(self):
        """Obtener subclases."""
        onto = Ontology("test")
        onto.add_class(OntologyClass(id="entity", name="Entity"))
        onto.add_class(OntologyClass(id="person", name="Person", parent="entity"))
        onto.add_class(OntologyClass(id="org", name="Organization", parent="entity"))
        
        subclasses = onto.get_subclasses("entity")
        assert "person" in subclasses
        assert "org" in subclasses
    
    def test_add_relation(self):
        """Agregar relaciones."""
        onto = Ontology("test")
        onto.add_class(OntologyClass(id="person", name="Person"))
        onto.add_class(OntologyClass(id="org", name="Organization"))
        
        onto.add_relation(OntologyRelation(
            id="WORKS_FOR",
            name="trabaja para",
            description="...",
            relation_type=RelationType.FUNCTIONAL,
            domain=["person"],
            range=["org"]
        ))
        
        assert len(onto.relations) == 1
        assert "WORKS_FOR" in onto.relations
    
    def test_normalize_entity(self):
        """Normalización de entidades."""
        onto = Ontology("test")
        onto.add_class(OntologyClass(
            id="flood",
            name="Inundación",
            synonyms=["flooding", "riada", "anegamiento"]
        ))
        
        assert onto.normalize_entity("inundación") == "flood"
        assert onto.normalize_entity("flooding") == "flood"
        assert onto.normalize_entity("riada") == "flood"
        assert onto.normalize_entity("sequía") is None
    
    def test_normalize_relation(self):
        """Normalización de relaciones."""
        onto = Ontology("test")
        onto.add_relation(OntologyRelation(
            id="MITIGATES",
            name="mitiga",
            description="...",
            relation_type=RelationType.MITIGATIVE,
            synonyms=["reduce", "atenúa"]
        ))
        
        assert onto.normalize_relation("mitiga") == "MITIGATES"
        assert onto.normalize_relation("reduce") == "MITIGATES"
        assert onto.normalize_relation("causa") is None
    
    def test_validation_detects_cycles(self):
        """Validación detecta ciclos en jerarquías."""
        onto = Ontology("test")
        onto.add_class(OntologyClass(id="a", name="A"))
        onto.add_class(OntologyClass(id="b", name="B", parent="a"))
        
        # Forzar ciclo modificando directamente
        onto.classes["a"].parent = "b"
        
        result = onto.validate()
        assert result.is_valid is False
        assert any("Ciclo" in e for e in result.errors)
    
    def test_validation_warns_vague_relations(self):
        """Validación advierte sobre relaciones vagas."""
        onto = Ontology("test")
        onto.add_class(OntologyClass(id="a", name="A"))
        onto.add_relation(OntologyRelation(
            id="AFFECTS",
            name="afecta",
            description="...",
            relation_type=RelationType.ASSOCIATIVE,
            domain=["a"],
            range=["a"]
        ))
        
        result = onto.validate()
        assert any("vaga" in w.lower() for w in result.warnings)
    
    def test_serialization(self):
        """Serialización completa."""
        onto = Ontology("test", "Test ontology")
        onto.add_class(OntologyClass(id="entity", name="Entity"))
        onto.add_class(OntologyClass(id="person", name="Person", parent="entity"))
        onto.add_relation(OntologyRelation(
            id="KNOWS",
            name="conoce",
            description="...",
            relation_type=RelationType.ASSOCIATIVE
        ))
        
        # Serializar a dict
        data = onto.to_dict()
        assert data["name"] == "test"
        assert len(data["classes"]) == 2
        assert len(data["relations"]) == 1
        
        # Deserializar
        restored = Ontology.from_dict(data)
        assert restored.name == onto.name
        assert len(restored.classes) == len(onto.classes)
        assert len(restored.relations) == len(onto.relations)
    
    def test_save_and_load(self):
        """Guardar y cargar desde archivo."""
        onto = Ontology("test")
        onto.add_class(OntologyClass(id="entity", name="Entity"))
        
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "ontology.json")
            onto.save(path)
            
            loaded = Ontology.load(path)
            assert loaded.name == onto.name
            assert len(loaded.classes) == len(onto.classes)


# ═══════════════════════════════════════════════════════════════════════════════
# TESTS DE ONTOLOGÍA PREDEFINIDA
# ═══════════════════════════════════════════════════════════════════════════════

class TestPredefinedOntology:
    """Tests para ontología climática predefinida."""
    
    def test_create_climate_ontology(self):
        """Crear ontología climática."""
        from ontorag.ontology import create_climate_ontology
        
        onto = create_climate_ontology()
        
        assert onto.name == "climate_adaptation"
        assert len(onto.classes) > 10
        assert len(onto.relations) > 5
        
        # Verificar clases clave
        assert "flood" in onto.classes
        assert "green_roof" in onto.classes
        assert "drought" in onto.classes
        
        # Verificar relaciones clave
        assert "MITIGATES" in onto.relations
        assert "PREVENTS" in onto.relations
        assert "CAUSES" in onto.relations
    
    def test_climate_ontology_validates(self):
        """Ontología climática pasa validación."""
        from ontorag.ontology import create_climate_ontology
        
        onto = create_climate_ontology()
        result = onto.validate()
        
        # Debe ser válida (sin errores)
        assert result.is_valid is True
        assert len(result.errors) == 0


# ═══════════════════════════════════════════════════════════════════════════════
# TESTS DE BÚSQUEDA
# ═══════════════════════════════════════════════════════════════════════════════

class TestSearch:
    """Tests para módulo de búsqueda."""
    
    def test_search_result_creation(self):
        """Crear SearchResult."""
        from ontorag.search import SearchResult, ItemType, SearchSource
        
        result = SearchResult(
            item_id="node_1",
            item_type=ItemType.NODE,
            score=0.95,
            source=SearchSource.SEMANTIC,
            data={"name": "Test"},
            explanation="High similarity"
        )
        
        assert result.item_id == "node_1"
        assert result.score == 0.95
        assert result.item_type == ItemType.NODE
    
    def test_search_result_score_clamping(self):
        """Score se limita a [0, 1]."""
        from ontorag.search import SearchResult, ItemType, SearchSource
        
        result = SearchResult(
            item_id="test",
            item_type=ItemType.NODE,
            score=1.5,  # Mayor que 1
            source=SearchSource.SEMANTIC
        )
        assert result.score == 1.0
        
        result2 = SearchResult(
            item_id="test",
            item_type=ItemType.NODE,
            score=-0.5,  # Menor que 0
            source=SearchSource.SEMANTIC
        )
        assert result2.score == 0.0


# ═══════════════════════════════════════════════════════════════════════════════
# TESTS DE EXTRACCIÓN
# ═══════════════════════════════════════════════════════════════════════════════

class TestExtraction:
    """Tests para módulo de extracción."""
    
    def test_rule_based_extraction(self):
        """Extracción basada en reglas."""
        from ontorag.extraction import RuleBasedExtractor
        
        extractor = RuleBasedExtractor()
        
        result = extractor.extract(
            "Los techos verdes mitigan el efecto isla de calor. "
            "Las inundaciones causan daños severos."
        )
        
        assert len(result.triples) >= 2
        
        # Verificar que se encontró la relación MITIGATES
        mitigates = [t for t in result.triples if t.relation == "MITIGATES"]
        assert len(mitigates) >= 1
        
        # Verificar que se encontró la relación CAUSES
        causes = [t for t in result.triples if t.relation == "CAUSES"]
        assert len(causes) >= 1
    
    def test_extraction_to_graph_elements(self):
        """Conversión de extracción a elementos de grafo."""
        from ontorag.extraction import RuleBasedExtractor, extraction_to_graph_elements
        
        extractor = RuleBasedExtractor()
        result = extractor.extract("A mitiga B. C causa D.")
        
        nodes, edges = extraction_to_graph_elements(result)
        
        assert len(nodes) > 0
        assert len(edges) > 0


# ═══════════════════════════════════════════════════════════════════════════════
# TESTS DE ANALYTICS
# ═══════════════════════════════════════════════════════════════════════════════

class TestAnalytics:
    """Tests para módulo de analytics."""
    
    def test_ontology_metrics(self):
        """Métricas de ontología."""
        from ontorag.analytics import OntologyAnalytics
        from ontorag.ontology import create_climate_ontology
        
        onto = create_climate_ontology()
        analytics = OntologyAnalytics(onto)
        
        metrics = analytics.compute_metrics()
        
        assert metrics.num_classes > 0
        assert metrics.num_relations > 0
        assert metrics.num_root_classes > 0
        assert metrics.avg_synonyms_per_class > 0


# ═══════════════════════════════════════════════════════════════════════════════
# TESTS DE RAG
# ═══════════════════════════════════════════════════════════════════════════════

class TestRAG:
    """Tests para módulo RAG."""
    
    def test_mock_llm(self):
        """LLM mock funciona correctamente."""
        from ontorag.rag import create_mock_llm
        
        llm = create_mock_llm()
        response = llm("System prompt", "User prompt")
        
        assert isinstance(response, str)
        assert len(response) > 0
        assert "Mock" in response
    
    def test_rag_context(self):
        """RAGContext genera texto correctamente."""
        from ontorag.rag import RAGContext
        
        context = RAGContext(
            nodes=[{"name": "Test Node", "type": "test", "description": "A test node"}],
            edges=[{"source": "a", "target": "b", "relation": "TESTS"}],
            documents=[{"id": "doc1", "text": "Document text"}]
        )
        
        text = context.to_text()
        
        assert "Test Node" in text
        assert "TESTS" in text
        assert len(text) > 0
    
    def test_rag_response(self):
        """RAGResponse serializa correctamente."""
        from ontorag.rag import RAGResponse, RAGContext
        
        response = RAGResponse(
            query="Test query",
            answer="Test answer",
            context=RAGContext(),
            system_prompt="System",
            user_prompt="User",
            citations=["doc1", "doc2"]
        )
        
        data = response.to_dict()
        
        assert data["query"] == "Test query"
        assert data["answer"] == "Test answer"
        assert len(data["citations"]) == 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
