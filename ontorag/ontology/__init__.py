"""
OntoRAG Ontology Module

This module provides classes for defining and managing ontologies.
Ontologies can be created programmatically or loaded from JSON files.

Key Components:
    - OntologyClass: Define entity types with hierarchies
    - OntologyRelation: Define typed relationships between entities
    - Ontology: Container for classes and relations with validation
    - load_ontology_from_json: Load ontology from JSON file
    - create_climate_ontology: Pre-built climate adaptation ontology

Example - Create ontology programmatically:
    >>> from ontorag.ontology import Ontology, OntologyClass, OntologyRelation
    >>> 
    >>> onto = Ontology("my_domain", "My Domain Ontology")
    >>> onto.add_class(OntologyClass(id="entity", name="Entity"))
    >>> onto.add_class(OntologyClass(id="person", name="Person", parent="entity"))
    >>> onto.add_relation(OntologyRelation(
    ...     id="KNOWS", name="knows", relation_type="associative"
    ... ))

Example - Load from JSON:
    >>> from ontorag.ontology import load_ontology_from_json
    >>> onto = load_ontology_from_json("my_ontology.json")
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union


# =============================================================================
# ENUMS
# =============================================================================

class RelationType(Enum):
    """
    Types of semantic relations in the ontology.
    
    Each type has different semantic implications:
    - HIERARCHICAL: IS-A, PART-OF relationships (implies inheritance)
    - CAUSAL: CAUSES, LEADS-TO relationships (implies directionality)
    - MITIGATIVE: MITIGATES, REDUCES relationships (action on hazard)
    - PROTECTIVE: PROTECTS, PREVENTS relationships
    - SPATIAL: LOCATED-IN, NEAR relationships
    - TEMPORAL: BEFORE, AFTER, DURING relationships
    - ASSOCIATIVE: Generic associations
    - FUNCTIONAL: HAS-FUNCTION, PRODUCES relationships
    """
    HIERARCHICAL = "hierarchical"
    CAUSAL = "causal"
    MITIGATIVE = "mitigative"
    PROTECTIVE = "protective"
    SPATIAL = "spatial"
    TEMPORAL = "temporal"
    ASSOCIATIVE = "associative"
    FUNCTIONAL = "functional"


# =============================================================================
# ONTOLOGY CLASS
# =============================================================================

@dataclass
class OntologyClass:
    """
    A class (type) in the ontology defining a category of entities.
    
    Attributes:
        id: Unique identifier (auto-normalized to snake_case)
        name: Human-readable name
        description: Detailed description of this class
        parent: ID of parent class (for IS-A hierarchy)
        synonyms: Alternative names for matching
        properties: Schema for instances of this class
        examples: Example instances
    
    Example:
        >>> flood = OntologyClass(
        ...     id="flood",
        ...     name="Flood",
        ...     description="Overflow of water onto normally dry land",
        ...     parent="climate_hazard",
        ...     synonyms=["flooding", "inundation", "deluge"]
        ... )
    """
    id: str
    name: str
    description: str = ""
    parent: Optional[str] = None
    synonyms: List[str] = field(default_factory=list)
    properties: Dict[str, Any] = field(default_factory=dict)
    examples: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        # Normalize ID to snake_case
        self.id = self._normalize_id(self.id)
        
        # Ensure name is in synonyms
        all_synonyms = {self.name.lower()}
        all_synonyms.update(s.lower().strip() for s in self.synonyms)
        self.synonyms = list(all_synonyms)
    
    @staticmethod
    def _normalize_id(text: str) -> str:
        """Convert to snake_case."""
        text = re.sub(r'[^\w\s-]', '', text.lower())
        text = re.sub(r'[-\s]+', '_', text)
        return text.strip('_')
    
    def matches(self, text: str, threshold: float = 0.8) -> Tuple[bool, float]:
        """
        Check if text matches this class.
        
        Args:
            text: Text to match
            threshold: Minimum similarity threshold
            
        Returns:
            Tuple of (is_match, score)
        """
        text_lower = text.lower().strip()
        
        # Exact match with synonyms
        if text_lower in self.synonyms or text_lower == self.id:
            return True, 1.0
        
        # Partial match
        for synonym in self.synonyms:
            if synonym in text_lower or text_lower in synonym:
                score = len(min(synonym, text_lower, key=len)) / len(max(synonym, text_lower, key=len))
                if score >= threshold:
                    return True, score
        
        return False, 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "parent": self.parent,
            "synonyms": self.synonyms,
            "properties": self.properties,
            "examples": self.examples,
            "metadata": self.metadata,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "OntologyClass":
        """Create from dictionary."""
        return cls(
            id=data.get("id", ""),
            name=data.get("name", ""),
            description=data.get("description", ""),
            parent=data.get("parent"),
            synonyms=data.get("synonyms", []),
            properties=data.get("properties", {}),
            examples=data.get("examples", []),
            metadata=data.get("metadata", {}),
        )


# =============================================================================
# ONTOLOGY RELATION
# =============================================================================

@dataclass
class OntologyRelation:
    """
    A relation type in the ontology defining how entities can be connected.
    
    Attributes:
        id: Unique identifier (auto-normalized to UPPER_CASE)
        name: Human-readable name
        description: What this relation means
        relation_type: Category of relation (causal, mitigative, etc.)
        domain: List of class IDs that can be subjects
        range: List of class IDs that can be objects
        inverse: ID of the inverse relation
        synonyms: Alternative names for matching
        is_transitive: If A->B and B->C implies A->C
        is_symmetric: If A->B implies B->A
    
    Example:
        >>> mitigates = OntologyRelation(
        ...     id="MITIGATES",
        ...     name="mitigates",
        ...     description="Subject reduces the effect of object",
        ...     relation_type="mitigative",
        ...     domain=["adaptation_measure"],
        ...     range=["climate_hazard"],
        ...     synonyms=["reduces", "lessens", "diminishes"]
        ... )
    """
    id: str
    name: str
    description: str = ""
    relation_type: Union[RelationType, str] = RelationType.ASSOCIATIVE
    domain: List[str] = field(default_factory=list)
    range: List[str] = field(default_factory=list)
    inverse: Optional[str] = None
    synonyms: List[str] = field(default_factory=list)
    is_transitive: bool = False
    is_symmetric: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        # Normalize ID to UPPER_CASE
        self.id = self._normalize_id(self.id)
        
        # Convert string to enum if needed
        if isinstance(self.relation_type, str):
            try:
                self.relation_type = RelationType(self.relation_type.lower())
            except ValueError:
                self.relation_type = RelationType.ASSOCIATIVE
        
        # Ensure name is in synonyms
        all_synonyms = {self.name.lower()}
        all_synonyms.update(s.lower().strip() for s in self.synonyms)
        self.synonyms = list(all_synonyms)
    
    @staticmethod
    def _normalize_id(text: str) -> str:
        """Convert to UPPER_CASE."""
        text = re.sub(r'[^\w\s-]', '', text)
        text = re.sub(r'[-\s]+', '_', text)
        return text.upper().strip('_')
    
    def matches(self, text: str) -> Tuple[bool, float]:
        """Check if text matches this relation."""
        text_lower = text.lower().strip()
        
        if text_lower in self.synonyms or text_lower == self.id.lower():
            return True, 1.0
        
        for synonym in self.synonyms:
            if synonym in text_lower or text_lower in synonym:
                return True, 0.8
        
        return False, 0.0
    
    def validate_triple(self, subject_type: str, object_type: str) -> bool:
        """
        Check if a triple with given types is valid for this relation.
        
        Args:
            subject_type: Type of the subject entity
            object_type: Type of the object entity
            
        Returns:
            True if the triple is valid
        """
        domain_ok = not self.domain or subject_type in self.domain
        range_ok = not self.range or object_type in self.range
        return domain_ok and range_ok
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "relation_type": self.relation_type.value if isinstance(self.relation_type, RelationType) else self.relation_type,
            "domain": self.domain,
            "range": self.range,
            "inverse": self.inverse,
            "synonyms": self.synonyms,
            "is_transitive": self.is_transitive,
            "is_symmetric": self.is_symmetric,
            "metadata": self.metadata,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "OntologyRelation":
        """Create from dictionary."""
        return cls(
            id=data.get("id", ""),
            name=data.get("name", ""),
            description=data.get("description", ""),
            relation_type=data.get("relation_type", "associative"),
            domain=data.get("domain", []),
            range=data.get("range", []),
            inverse=data.get("inverse"),
            synonyms=data.get("synonyms", []),
            is_transitive=data.get("is_transitive", False),
            is_symmetric=data.get("is_symmetric", False),
            metadata=data.get("metadata", {}),
        )


# =============================================================================
# VALIDATION RESULT
# =============================================================================

@dataclass
class ValidationResult:
    """
    Result of ontology validation.
    
    Attributes:
        is_valid: Whether the ontology passed all validations
        errors: List of critical errors
        warnings: List of non-critical warnings
        info: Informational messages
    """
    is_valid: bool = True
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    info: List[str] = field(default_factory=list)
    
    def add_error(self, message: str) -> None:
        """Add an error and mark as invalid."""
        self.errors.append(message)
        self.is_valid = False
    
    def add_warning(self, message: str) -> None:
        """Add a warning."""
        self.warnings.append(message)
    
    def add_info(self, message: str) -> None:
        """Add informational message."""
        self.info.append(message)
    
    def summary(self) -> str:
        """Generate summary string."""
        lines = [
            f"Valid: {self.is_valid}",
            f"Errors: {len(self.errors)}",
            f"Warnings: {len(self.warnings)}",
        ]
        if self.errors:
            lines.append("\nErrors:")
            for e in self.errors:
                lines.append(f"  ✗ {e}")
        if self.warnings:
            lines.append("\nWarnings:")
            for w in self.warnings:
                lines.append(f"  ⚠ {w}")
        return "\n".join(lines)


# =============================================================================
# ONTOLOGY
# =============================================================================

class Ontology:
    """
    Container for ontology classes and relations.
    
    The Ontology class provides methods for:
    - Adding and managing classes and relations
    - Validating ontology consistency
    - Normalizing entities and relations
    - Serialization to/from JSON
    
    Attributes:
        name: Name of the ontology
        description: Description of the domain
        version: Version string
        classes: Dictionary of OntologyClass objects
        relations: Dictionary of OntologyRelation objects
    
    Example:
        >>> onto = Ontology("climate", "Climate Adaptation Ontology")
        >>> 
        >>> # Add classes
        >>> onto.add_class(OntologyClass(id="hazard", name="Climate Hazard"))
        >>> onto.add_class(OntologyClass(id="flood", name="Flood", parent="hazard"))
        >>> 
        >>> # Add relations
        >>> onto.add_relation(OntologyRelation(id="MITIGATES", name="mitigates"))
        >>> 
        >>> # Validate
        >>> result = onto.validate()
        >>> print(result.is_valid)
        True
    """
    
    def __init__(
        self,
        name: str,
        description: str = "",
        version: str = "1.0.0"
    ):
        """
        Initialize a new ontology.
        
        Args:
            name: Name of the ontology
            description: Description of the domain
            version: Version string
        """
        self.name = name
        self.description = description
        self.version = version
        self._classes: Dict[str, OntologyClass] = {}
        self._relations: Dict[str, OntologyRelation] = {}
        self.metadata: Dict[str, Any] = {}
    
    # =========================================================================
    # PROPERTIES
    # =========================================================================
    
    @property
    def classes(self) -> Dict[str, OntologyClass]:
        """Get all classes."""
        return self._classes
    
    @property
    def relations(self) -> Dict[str, OntologyRelation]:
        """Get all relations."""
        return self._relations
    
    # =========================================================================
    # CLASS MANAGEMENT
    # =========================================================================
    
    def add_class(self, ontology_class: OntologyClass) -> "Ontology":
        """
        Add a class to the ontology.
        
        Args:
            ontology_class: The class to add
            
        Returns:
            Self for chaining
            
        Raises:
            ValueError: If class ID already exists or parent doesn't exist
        """
        if ontology_class.id in self._classes:
            raise ValueError(f"Class '{ontology_class.id}' already exists")
        
        if ontology_class.parent and ontology_class.parent not in self._classes:
            raise ValueError(f"Parent class '{ontology_class.parent}' not found")
        
        self._classes[ontology_class.id] = ontology_class
        return self
    
    def get_class(self, class_id: str) -> Optional[OntologyClass]:
        """Get a class by ID."""
        return self._classes.get(class_id)
    
    def get_class_hierarchy(self, class_id: str) -> List[str]:
        """
        Get the hierarchy chain from class to root.
        
        Args:
            class_id: ID of the class
            
        Returns:
            List of class IDs from child to root
        """
        hierarchy = []
        current = class_id
        
        while current:
            if current in hierarchy:
                break  # Avoid cycles
            hierarchy.append(current)
            cls = self._classes.get(current)
            current = cls.parent if cls else None
        
        return hierarchy
    
    def get_subclasses(self, class_id: str, recursive: bool = True) -> List[str]:
        """
        Get all subclasses of a class.
        
        Args:
            class_id: ID of the parent class
            recursive: Whether to include nested subclasses
            
        Returns:
            List of subclass IDs
        """
        subclasses = []
        
        for cls_id, cls in self._classes.items():
            if cls.parent == class_id:
                subclasses.append(cls_id)
                if recursive:
                    subclasses.extend(self.get_subclasses(cls_id, recursive=True))
        
        return subclasses
    
    def is_subclass_of(self, child_id: str, parent_id: str) -> bool:
        """Check if one class is a subclass of another."""
        hierarchy = self.get_class_hierarchy(child_id)
        return parent_id in hierarchy
    
    # =========================================================================
    # RELATION MANAGEMENT
    # =========================================================================
    
    def add_relation(self, relation: OntologyRelation) -> "Ontology":
        """
        Add a relation to the ontology.
        
        Args:
            relation: The relation to add
            
        Returns:
            Self for chaining
            
        Raises:
            ValueError: If relation ID already exists
        """
        if relation.id in self._relations:
            raise ValueError(f"Relation '{relation.id}' already exists")
        
        self._relations[relation.id] = relation
        return self
    
    def get_relation(self, relation_id: str) -> Optional[OntologyRelation]:
        """Get a relation by ID."""
        return self._relations.get(relation_id.upper())
    
    # =========================================================================
    # NORMALIZATION
    # =========================================================================
    
    def normalize_entity(self, text: str) -> Optional[str]:
        """
        Normalize text to a class ID using synonyms.
        
        Args:
            text: Text to normalize
            
        Returns:
            Class ID if matched, None otherwise
        """
        text_lower = text.lower().strip()
        
        best_match = None
        best_score = 0.0
        
        for cls_id, cls in self._classes.items():
            is_match, score = cls.matches(text_lower)
            if is_match and score > best_score:
                best_match = cls_id
                best_score = score
        
        return best_match
    
    def normalize_relation(self, text: str) -> Optional[str]:
        """
        Normalize text to a relation ID using synonyms.
        
        Args:
            text: Text to normalize
            
        Returns:
            Relation ID if matched, None otherwise
        """
        text_lower = text.lower().strip()
        
        for rel_id, rel in self._relations.items():
            is_match, _ = rel.matches(text_lower)
            if is_match:
                return rel_id
        
        return None
    
    # =========================================================================
    # VALIDATION
    # =========================================================================
    
    def validate(self) -> ValidationResult:
        """
        Validate the ontology for consistency.
        
        Checks performed:
        1. All parent references are valid
        2. No circular hierarchies
        3. Domain/range references valid classes
        4. Inverse relations exist
        5. No vague relations
        
        Returns:
            ValidationResult with errors and warnings
        """
        result = ValidationResult()
        
        # Check 1: Parent references
        for cls_id, cls in self._classes.items():
            if cls.parent and cls.parent not in self._classes:
                result.add_error(f"Class '{cls_id}' has invalid parent '{cls.parent}'")
        
        # Check 2: Circular hierarchies
        for cls_id in self._classes:
            hierarchy = self.get_class_hierarchy(cls_id)
            if len(hierarchy) != len(set(hierarchy)):
                result.add_error(f"Circular hierarchy detected involving '{cls_id}'")
        
        # Check 3: Domain/range references
        for rel_id, rel in self._relations.items():
            for domain_cls in rel.domain:
                if domain_cls not in self._classes:
                    result.add_warning(f"Relation '{rel_id}' has unknown domain class '{domain_cls}'")
            for range_cls in rel.range:
                if range_cls not in self._classes:
                    result.add_warning(f"Relation '{rel_id}' has unknown range class '{range_cls}'")
        
        # Check 4: Inverse relations
        for rel_id, rel in self._relations.items():
            if rel.inverse and rel.inverse not in self._relations:
                result.add_warning(f"Relation '{rel_id}' has invalid inverse '{rel.inverse}'")
        
        # Check 5: Vague relations
        vague_terms = ["affects", "relates", "has", "is", "does"]
        for rel_id, rel in self._relations.items():
            if any(term in rel.name.lower() for term in vague_terms):
                result.add_warning(f"Relation '{rel_id}' uses vague term in name")
        
        # Info
        result.add_info(f"Classes: {len(self._classes)}")
        result.add_info(f"Relations: {len(self._relations)}")
        
        return result
    
    # =========================================================================
    # SERIALIZATION
    # =========================================================================
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize ontology to dictionary."""
        return {
            "name": self.name,
            "description": self.description,
            "version": self.version,
            "metadata": self.metadata,
            "classes": {k: v.to_dict() for k, v in self._classes.items()},
            "relations": {k: v.to_dict() for k, v in self._relations.items()},
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Ontology":
        """Create ontology from dictionary."""
        onto = cls(
            name=data.get("name", "unnamed"),
            description=data.get("description", ""),
            version=data.get("version", "1.0.0"),
        )
        onto.metadata = data.get("metadata", {})
        
        # Load classes (respecting parent order)
        classes_data = data.get("classes", {})
        loaded = set()
        
        def load_class(cls_id: str, cls_data: Dict) -> None:
            if cls_id in loaded:
                return
            parent = cls_data.get("parent")
            if parent and parent not in loaded and parent in classes_data:
                load_class(parent, classes_data[parent])
            onto._classes[cls_id] = OntologyClass.from_dict(cls_data)
            loaded.add(cls_id)
        
        for cls_id, cls_data in classes_data.items():
            load_class(cls_id, cls_data)
        
        # Load relations
        for rel_id, rel_data in data.get("relations", {}).items():
            onto._relations[rel_id] = OntologyRelation.from_dict(rel_data)
        
        return onto
    
    def save(self, path: Union[str, Path]) -> None:
        """
        Save ontology to JSON file.
        
        Args:
            path: Path to save the file
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)
    
    @classmethod
    def load(cls, path: Union[str, Path]) -> "Ontology":
        """
        Load ontology from JSON file.
        
        Args:
            path: Path to the JSON file
            
        Returns:
            Loaded Ontology instance
        """
        with open(path, 'r', encoding='utf-8') as f:
            return cls.from_dict(json.load(f))
    
    # =========================================================================
    # STRING REPRESENTATION
    # =========================================================================
    
    def __repr__(self) -> str:
        return f"Ontology('{self.name}', classes={len(self._classes)}, relations={len(self._relations)})"
    
    def summary(self) -> str:
        """Generate a summary string."""
        lines = [
            f"Ontology: {self.name}",
            f"Description: {self.description}",
            f"Version: {self.version}",
            f"Classes: {len(self._classes)}",
            f"Relations: {len(self._relations)}",
        ]
        return "\n".join(lines)


# =============================================================================
# JSON LOADING FUNCTIONS
# =============================================================================

def load_ontology_from_json(path: Union[str, Path]) -> Ontology:
    """
    Load an ontology from a JSON file.
    
    The JSON file should have this structure:
    {
        "name": "my_ontology",
        "description": "Description of the domain",
        "version": "1.0.0",
        "classes": {
            "class_id": {
                "name": "Class Name",
                "description": "...",
                "parent": "parent_id",  // optional
                "synonyms": ["syn1", "syn2"]
            }
        },
        "relations": {
            "RELATION_ID": {
                "name": "relation name",
                "description": "...",
                "relation_type": "mitigative",  // or causal, spatial, etc.
                "domain": ["class1"],
                "range": ["class2"],
                "synonyms": ["syn1"]
            }
        }
    }
    
    Args:
        path: Path to the JSON file
        
    Returns:
        Loaded Ontology instance
        
    Example:
        >>> onto = load_ontology_from_json("my_domain.json")
        >>> print(onto.classes)
    """
    return Ontology.load(path)


def create_ontology_from_dict(data: Dict[str, Any]) -> Ontology:
    """
    Create an ontology from a dictionary.
    
    Useful for creating ontologies from inline JSON or configuration.
    
    Args:
        data: Dictionary with ontology structure
        
    Returns:
        New Ontology instance
    """
    return Ontology.from_dict(data)


# =============================================================================
# PREDEFINED ONTOLOGIES
# =============================================================================

def create_climate_ontology() -> Ontology:
    """
    Create a predefined ontology for climate adaptation domain.
    
    This ontology includes:
    - Climate hazards (flood, drought, heat wave, etc.)
    - Adaptation measures (green roof, early warning, etc.)
    - Sectors (agriculture, infrastructure, health, etc.)
    - Semantic relations (MITIGATES, PROTECTS, CAUSES, etc.)
    
    Returns:
        Climate adaptation Ontology
        
    Example:
        >>> onto = create_climate_ontology()
        >>> print(onto.classes.keys())
        >>> print(onto.normalize_entity("flooding"))
        'flood'
    """
    onto = Ontology(
        name="climate_adaptation",
        description="Ontology for climate change adaptation measures and hazards",
        version="1.0.0"
    )
    
    # ─────────────────────────────────────────────────────────────────────────
    # ROOT CLASSES
    # ─────────────────────────────────────────────────────────────────────────
    onto.add_class(OntologyClass(
        id="entity", name="Entity",
        description="Root class for all entities"
    ))
    
    # ─────────────────────────────────────────────────────────────────────────
    # CLIMATE HAZARDS
    # ─────────────────────────────────────────────────────────────────────────
    onto.add_class(OntologyClass(
        id="climate_hazard", name="Climate Hazard",
        description="Natural hazards exacerbated by climate change",
        parent="entity",
        synonyms=["hazard", "climate risk", "climate threat"]
    ))
    
    onto.add_class(OntologyClass(
        id="flood", name="Flood",
        description="Overflow of water onto normally dry land",
        parent="climate_hazard",
        synonyms=["flooding", "inundation", "flash flood", "coastal flood"]
    ))
    
    onto.add_class(OntologyClass(
        id="drought", name="Drought",
        description="Extended period of water scarcity",
        parent="climate_hazard",
        synonyms=["water scarcity", "water stress", "aridity"]
    ))
    
    onto.add_class(OntologyClass(
        id="heat_wave", name="Heat Wave",
        description="Extended period of extreme heat",
        parent="climate_hazard",
        synonyms=["heatwave", "extreme heat", "heat stress", "urban heat island"]
    ))
    
    onto.add_class(OntologyClass(
        id="sea_level_rise", name="Sea Level Rise",
        description="Long-term increase in sea levels",
        parent="climate_hazard",
        synonyms=["rising seas", "coastal flooding", "storm surge"]
    ))
    
    onto.add_class(OntologyClass(
        id="wildfire", name="Wildfire",
        description="Uncontrolled fire in wildland areas",
        parent="climate_hazard",
        synonyms=["forest fire", "bushfire", "wildland fire"]
    ))
    
    onto.add_class(OntologyClass(
        id="storm", name="Storm",
        description="Severe weather event with strong winds",
        parent="climate_hazard",
        synonyms=["hurricane", "typhoon", "cyclone", "tropical storm", "extreme weather"]
    ))
    
    # ─────────────────────────────────────────────────────────────────────────
    # ADAPTATION MEASURES
    # ─────────────────────────────────────────────────────────────────────────
    onto.add_class(OntologyClass(
        id="adaptation_measure", name="Adaptation Measure",
        description="Actions to reduce vulnerability to climate impacts",
        parent="entity",
        synonyms=["adaptation strategy", "climate measure", "intervention"]
    ))
    
    onto.add_class(OntologyClass(
        id="green_infrastructure", name="Green Infrastructure",
        description="Nature-based solutions for climate adaptation",
        parent="adaptation_measure",
        synonyms=["nature-based solution", "NbS", "ecosystem-based adaptation", "EbA"]
    ))
    
    onto.add_class(OntologyClass(
        id="green_roof", name="Green Roof",
        description="Vegetated roof system",
        parent="green_infrastructure",
        synonyms=["vegetated roof", "living roof", "roof garden"]
    ))
    
    onto.add_class(OntologyClass(
        id="rain_garden", name="Rain Garden",
        description="Planted depression for stormwater management",
        parent="green_infrastructure",
        synonyms=["bioswale", "bioretention", "stormwater garden"]
    ))
    
    onto.add_class(OntologyClass(
        id="urban_forest", name="Urban Forest",
        description="Trees and vegetation in urban areas",
        parent="green_infrastructure",
        synonyms=["street trees", "tree canopy", "urban trees", "tree planting"]
    ))
    
    onto.add_class(OntologyClass(
        id="wetland_restoration", name="Wetland Restoration",
        description="Restoration of wetland ecosystems",
        parent="green_infrastructure",
        synonyms=["wetland", "marsh restoration", "mangrove restoration"]
    ))
    
    onto.add_class(OntologyClass(
        id="grey_infrastructure", name="Grey Infrastructure",
        description="Engineered infrastructure for climate adaptation",
        parent="adaptation_measure",
        synonyms=["hard infrastructure", "engineered solution"]
    ))
    
    onto.add_class(OntologyClass(
        id="flood_barrier", name="Flood Barrier",
        description="Physical barrier to prevent flooding",
        parent="grey_infrastructure",
        synonyms=["sea wall", "dike", "levee", "storm surge barrier", "flood wall"]
    ))
    
    onto.add_class(OntologyClass(
        id="drainage_system", name="Drainage System",
        description="System for managing stormwater",
        parent="grey_infrastructure",
        synonyms=["stormwater system", "drainage", "retention pond"]
    ))
    
    onto.add_class(OntologyClass(
        id="early_warning_system", name="Early Warning System",
        description="System for alerting about impending hazards",
        parent="adaptation_measure",
        synonyms=["EWS", "warning system", "alert system", "flood forecasting"]
    ))
    
    onto.add_class(OntologyClass(
        id="climate_smart_agriculture", name="Climate-Smart Agriculture",
        description="Agricultural practices for climate resilience",
        parent="adaptation_measure",
        synonyms=["CSA", "drought-resistant crops", "conservation agriculture"]
    ))
    
    # ─────────────────────────────────────────────────────────────────────────
    # SECTORS
    # ─────────────────────────────────────────────────────────────────────────
    onto.add_class(OntologyClass(
        id="sector", name="Sector",
        description="Economic or social sector",
        parent="entity"
    ))
    
    onto.add_class(OntologyClass(
        id="agriculture", name="Agriculture",
        description="Farming and food production sector",
        parent="sector",
        synonyms=["farming", "crops", "food production"]
    ))
    
    onto.add_class(OntologyClass(
        id="infrastructure", name="Infrastructure",
        description="Physical infrastructure sector",
        parent="sector",
        synonyms=["buildings", "roads", "utilities"]
    ))
    
    onto.add_class(OntologyClass(
        id="health", name="Health",
        description="Public health sector",
        parent="sector",
        synonyms=["public health", "healthcare", "health sector"]
    ))
    
    onto.add_class(OntologyClass(
        id="water_resources", name="Water Resources",
        description="Water management sector",
        parent="sector",
        synonyms=["water sector", "water management", "hydrology"]
    ))
    
    # ─────────────────────────────────────────────────────────────────────────
    # LOCATIONS
    # ─────────────────────────────────────────────────────────────────────────
    onto.add_class(OntologyClass(
        id="location", name="Location",
        description="Geographic location",
        parent="entity",
        synonyms=["place", "area", "region"]
    ))
    
    onto.add_class(OntologyClass(
        id="coastal_area", name="Coastal Area",
        description="Area near the coast",
        parent="location",
        synonyms=["coast", "coastal zone", "shoreline"]
    ))
    
    onto.add_class(OntologyClass(
        id="urban_area", name="Urban Area",
        description="City or urban region",
        parent="location",
        synonyms=["city", "urban", "metropolitan"]
    ))
    
    # ─────────────────────────────────────────────────────────────────────────
    # RELATIONS
    # ─────────────────────────────────────────────────────────────────────────
    onto.add_relation(OntologyRelation(
        id="MITIGATES", name="mitigates",
        description="Subject reduces the impact of object",
        relation_type=RelationType.MITIGATIVE,
        domain=["adaptation_measure"],
        range=["climate_hazard"],
        synonyms=["reduces", "lessens", "alleviates", "diminishes"]
    ))
    
    onto.add_relation(OntologyRelation(
        id="PROTECTS", name="protects",
        description="Subject protects against object",
        relation_type=RelationType.PROTECTIVE,
        domain=["adaptation_measure"],
        range=["climate_hazard", "sector"],
        synonyms=["defends", "shields", "guards"]
    ))
    
    onto.add_relation(OntologyRelation(
        id="PREVENTS", name="prevents",
        description="Subject prevents object from occurring",
        relation_type=RelationType.PROTECTIVE,
        domain=["adaptation_measure"],
        range=["climate_hazard"],
        synonyms=["stops", "blocks", "averts"]
    ))
    
    onto.add_relation(OntologyRelation(
        id="CAUSES", name="causes",
        description="Subject causes object",
        relation_type=RelationType.CAUSAL,
        domain=["climate_hazard"],
        range=["climate_hazard"],
        synonyms=["leads to", "results in", "triggers"]
    ))
    
    onto.add_relation(OntologyRelation(
        id="AFFECTS", name="affects",
        description="Subject has impact on object",
        relation_type=RelationType.CAUSAL,
        domain=["climate_hazard"],
        range=["sector", "location"],
        synonyms=["impacts", "influences"]
    ))
    
    onto.add_relation(OntologyRelation(
        id="LOCATED_IN", name="located in",
        description="Subject is located in object",
        relation_type=RelationType.SPATIAL,
        synonyms=["in", "at", "within"]
    ))
    
    onto.add_relation(OntologyRelation(
        id="IS_A", name="is a",
        description="Subject is a type of object",
        relation_type=RelationType.HIERARCHICAL,
        is_transitive=True,
        synonyms=["type of", "kind of", "instance of"]
    ))
    
    onto.add_relation(OntologyRelation(
        id="PART_OF", name="part of",
        description="Subject is part of object",
        relation_type=RelationType.HIERARCHICAL,
        is_transitive=True,
        synonyms=["component of", "belongs to"]
    ))
    
    onto.add_relation(OntologyRelation(
        id="IMPLEMENTED_IN", name="implemented in",
        description="Measure is implemented in location",
        relation_type=RelationType.SPATIAL,
        domain=["adaptation_measure"],
        range=["location"],
        synonyms=["deployed in", "installed in"]
    ))
    
    onto.add_relation(OntologyRelation(
        id="REDUCES_BY", name="reduces by",
        description="Quantified reduction",
        relation_type=RelationType.FUNCTIONAL,
        synonyms=["decreases by", "lowers by"]
    ))
    
    return onto


# Alias for backward compatibility
create_climate_adaptation_ontology = create_climate_ontology


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    # Classes
    "OntologyClass",
    "OntologyRelation",
    "RelationType",
    "ValidationResult",
    "Ontology",
    
    # Functions
    "load_ontology_from_json",
    "create_ontology_from_dict",
    "create_climate_ontology",
    "create_climate_adaptation_ontology",
]
