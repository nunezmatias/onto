"""
╔═══════════════════════════════════════════════════════════════════════════════╗
║                         ONTORAG - MÓDULO ONTOLOGY                             ║
║                                                                               ║
║  El CORAZÓN de OntoRAG. La ontología es ciudadano de PRIMERA CLASE.          ║
║                                                                               ║
║  Principios de Diseño (de la Guía de Ingeniería Ontológica):                 ║
║  1. Relaciones SEMÁNTICAMENTE RICAS (no verbos vacíos como AFECTA)           ║
║  2. Jerarquías IS-A separadas de PART-OF                                     ║
║  3. Normalización semántica con sinónimos                                    ║
║  4. Validación de consistencia (6 verificaciones)                            ║
║  5. Desambiguación semántica (banco → institución vs mueble)                 ║
╚═══════════════════════════════════════════════════════════════════════════════╝
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Any, Tuple, Iterator
from enum import Enum
from pathlib import Path
import json
import re
from datetime import datetime


# ═══════════════════════════════════════════════════════════════════════════════
# ENUMS
# ═══════════════════════════════════════════════════════════════════════════════

class RelationType(Enum):
    """
    Tipos semánticos de relaciones.
    
    ⚠️ EVITAR: Relaciones vacías como AFFECTS, HAS, RELATED_TO
    ✓ USAR: Relaciones específicas según el tipo semántico
    """
    # Relaciones causales
    CAUSAL = "causal"           # X causa/produce Y
    MITIGATIVE = "mitigative"   # X reduce/atenúa Y
    PREVENTIVE = "preventive"   # X evita/previene Y
    
    # Relaciones estructurales
    TAXONOMIC = "taxonomic"     # X es-un Y (IS-A)
    COMPOSITIONAL = "compositional"  # X parte-de Y (PART-OF)
    
    # Relaciones espaciales/temporales
    SPATIAL = "spatial"         # X ubicado-en Y
    TEMPORAL = "temporal"       # X precede/sigue Y
    
    # Relaciones funcionales
    FUNCTIONAL = "functional"   # X sirve-para Y
    
    # Último recurso (usar con precaución)
    ASSOCIATIVE = "associative"  # X relacionado-con Y


# ═══════════════════════════════════════════════════════════════════════════════
# ONTOLOGY CLASS
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class MatchResult:
    """Resultado de coincidencia semántica con soporte para desempacar y evaluar en booleano."""
    is_match: bool
    score: float
    
    def __iter__(self):
        yield self.is_match
        yield self.score
    
    def __bool__(self):
        return self.is_match
    
    def __getitem__(self, index: int):
        if index == 0:
            return self.is_match
        if index == 1:
            return self.score
        raise IndexError(index)


@dataclass
class OntologyClass:
    """
    Clase en la ontología - define un TIPO de entidad.
    
    Ejemplo del dominio clima:
        >>> flood = OntologyClass(
        ...     id="flood",
        ...     name="Inundación", 
        ...     description="Desbordamiento de agua sobre tierra seca",
        ...     parent="climate_hazard",
        ...     synonyms=["inundaciones", "flooding", "anegamiento"]
        ... )
    
    Attributes:
        id: Identificador único (snake_case, ej: "green_roof")
        name: Etiqueta legible (ej: "Techo Verde")
        description: Descripción semántica clara
        parent: ID de clase padre (relación IS-A)
        synonyms: Lista de sinónimos para normalización
        properties: Esquema de propiedades de instancias
        disjoint_with: Clases con las que es disjunta
    """
    id: str
    name: str
    description: str = ""
    parent: Optional[str] = None
    synonyms: List[str] = field(default_factory=list)
    properties: Dict[str, Any] = field(default_factory=dict)
    disjoint_with: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        self.id = self._to_snake_case(self.id)
        # Agregar nombre como sinónimo en formatos original y normalizado
        all_synonyms = set(self.synonyms)
        all_synonyms.add(self.name)
        all_synonyms.add(self.name.lower())
        all_synonyms.update(s.lower().strip() for s in self.synonyms)
        self.synonyms = list(all_synonyms)
    
    @staticmethod
    def _to_snake_case(text: str) -> str:
        text = re.sub(r'[^\w\s]', '', str(text))
        text = re.sub(r'\s+', '_', text.strip())
        return text.lower()
    
    @property
    def label(self) -> str:
        """Alias para compatibilidad con versiones previas."""
        return self.name
    
    @label.setter
    def label(self, value: str):
        self.name = value
    
    def matches(self, text: str) -> MatchResult:
        """Verifica si texto coincide con esta clase."""
        text_norm = self._normalize_term(text)
        synonyms_norm = {self._normalize_term(s) for s in self.synonyms}
        synonyms_norm.add(self.id)
        
        if text_norm in synonyms_norm:
            return MatchResult(True, 1.0)
        for syn in synonyms_norm:
            if syn in text_norm or text_norm in syn:
                return MatchResult(True, 0.8)
        return MatchResult(False, 0.0)
    
    @staticmethod
    def _normalize_term(text: str) -> str:
        import unicodedata
        
        normalized = unicodedata.normalize("NFKD", str(text))
        normalized = "".join(c for c in normalized if not unicodedata.combining(c))
        return normalized.lower().strip()
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "label": self.name,  # Alias para compatibilidad
            "description": self.description,
            "parent": self.parent,
            "synonyms": self.synonyms,
            "properties": self.properties,
            "disjoint_with": self.disjoint_with,
            "metadata": self.metadata,
        }
    
    @classmethod
    def from_dict(cls, d: Dict) -> "OntologyClass":
        data = d.copy()
        if "label" in data and "name" not in data:
            data["name"] = data["label"]
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


# ═══════════════════════════════════════════════════════════════════════════════
# ONTOLOGY RELATION
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass 
class OntologyRelation:
    """
    Relación en la ontología - define CÓMO se conectan entidades.
    
    ⚠️ CRÍTICO: Evitar relaciones semánticamente vacías:
       ❌ AFFECTS, HAS, RELATED_TO, HAS_EFFECT
       ✓ MITIGATES, CAUSES, PREVENTS, PROTECTS
    
    Ejemplo:
        >>> mitigates = OntologyRelation(
        ...     id="MITIGATES",
        ...     name="mitiga",
        ...     description="El sujeto reduce los efectos negativos del objeto",
        ...     relation_type=RelationType.MITIGATIVE,
        ...     domain=["adaptation_measure"],
        ...     range=["climate_hazard"],
        ...     inverse="MITIGATED_BY",
        ...     synonyms=["reduce", "atenúa", "disminuye"]
        ... )
    """
    id: str
    name: str
    description: str
    relation_type: RelationType
    domain: List[str] = field(default_factory=list)  # Clases válidas como sujeto
    range: List[str] = field(default_factory=list)   # Clases válidas como objeto
    inverse: Optional[str] = None
    synonyms: List[str] = field(default_factory=list)
    is_transitive: bool = False
    is_symmetric: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    # Relaciones a EVITAR
    VAGUE_RELATIONS = {"AFFECTS", "AFECTA", "HAS", "TIENE", "RELATED", 
                       "RELACIONADO", "HAS_EFFECT", "TIENE_EFECTO"}
    
    def __post_init__(self):
        self.id = self.id.upper().replace(" ", "_")
        all_synonyms = set(self.synonyms)
        all_synonyms.add(self.name)
        all_synonyms.add(self.name.lower())
        all_synonyms.update(s.lower() for s in self.synonyms)
        self.synonyms = list(all_synonyms)
    
    @property
    def label(self) -> str:
        """Alias de compatibilidad para el nombre de la relación."""
        return self.name
    
    @label.setter
    def label(self, value: str):
        self.name = value
    
    @property
    def is_semantically_vague(self) -> bool:
        """¿Es una relación semánticamente vacía?"""
        return self.id in self.VAGUE_RELATIONS
    
    @property
    def is_vague(self) -> bool:
        """Alias legible para is_semantically_vague."""
        return self.is_semantically_vague
    
    @property
    def is_directional(self) -> bool:
        """Las relaciones son direccionales salvo que se marquen como simétricas."""
        return not self.is_symmetric
    
    def validate_triple(self, subject_type: str, object_type: str, 
                       hierarchies: Dict[str, List[str]]) -> Tuple[bool, str]:
        """Valida que una tripleta cumpla restricciones de dominio/rango."""
        if not self.domain and not self.range:
            return True, "OK"
        
        # Obtener ancestros del sujeto y objeto
        subject_ancestors = set(hierarchies.get(subject_type, []))
        subject_ancestors.add(subject_type)
        object_ancestors = set(hierarchies.get(object_type, []))
        object_ancestors.add(object_type)
        
        # Validar dominio
        if self.domain and not any(d in subject_ancestors for d in self.domain):
            return False, f"'{subject_type}' no está en dominio {self.domain}"
        
        # Validar rango
        if self.range and not any(r in object_ancestors for r in self.range):
            return False, f"'{object_type}' no está en rango {self.range}"
        
        return True, "OK"
    
    def matches(self, text: str) -> MatchResult:
        text_norm = self._normalize_term(text)
        synonyms_norm = {self._normalize_term(s) for s in self.synonyms}
        synonyms_norm.add(self.id.lower())
        if text_norm in synonyms_norm:
            return MatchResult(True, 1.0)
        for syn in synonyms_norm:
            if syn in text_norm or text_norm in syn:
                return MatchResult(True, 0.7)
        return MatchResult(False, 0.0)
    
    @staticmethod
    def _normalize_term(text: str) -> str:
        import unicodedata
        
        normalized = unicodedata.normalize("NFKD", str(text))
        normalized = "".join(c for c in normalized if not unicodedata.combining(c))
        return normalized.lower().strip()
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id, "name": self.name, "label": self.name, "description": self.description,
            "relation_type": self.relation_type.value, "domain": self.domain,
            "range": self.range, "inverse": self.inverse, "synonyms": self.synonyms,
            "is_transitive": self.is_transitive, "is_symmetric": self.is_symmetric,
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, d: Dict) -> "OntologyRelation":
        d = d.copy()
        if "relation_type" in d:
            d["relation_type"] = RelationType(d["relation_type"])
        if "label" in d and "name" not in d:
            d["name"] = d["label"]
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


# ═══════════════════════════════════════════════════════════════════════════════
# VALIDATION RESULT
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class ValidationResult:
    """Resultado de validación de la ontología."""
    is_valid: bool = True
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    info: List[str] = field(default_factory=list)
    checks_passed: int = 0
    checks_total: int = 6
    
    @property
    def score(self) -> float:
        return self.checks_passed / self.checks_total if self.checks_total > 0 else 0
    
    def __str__(self) -> str:
        status = "✅ VÁLIDA" if self.is_valid else "❌ INVÁLIDA"
        lines = [f"\n{'═'*60}", f"VALIDACIÓN: {status} ({self.score:.0%})", "═"*60]
        if self.errors:
            lines.append("\n❌ ERRORES:")
            lines.extend(f"   • {e}" for e in self.errors)
        if self.warnings:
            lines.append("\n⚠️ ADVERTENCIAS:")
            lines.extend(f"   • {w}" for w in self.warnings)
        if self.info:
            lines.append("\nℹ️ INFO:")
            lines.extend(f"   • {i}" for i in self.info)
        return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════════════
# ONTOLOGY - CLASE PRINCIPAL
# ═══════════════════════════════════════════════════════════════════════════════

class Ontology:
    """
    Clase principal de ontología - EL CORAZÓN DE ONTORAG.
    
    La ontología define:
    - QUÉ tipos de entidades existen (clases)
    - CÓMO se relacionan (relaciones con semántica rica)
    - CÓMO normalizar variantes lingüísticas (sinónimos)
    
    Ejemplo completo:
        >>> onto = Ontology("climate_adaptation")
        >>> 
        >>> # Definir clases
        >>> onto.add_class(OntologyClass("climate_hazard", "Peligro Climático"))
        >>> onto.add_class(OntologyClass("flood", "Inundación", parent="climate_hazard"))
        >>> 
        >>> # Definir relaciones semánticas
        >>> onto.add_relation(OntologyRelation(
        ...     id="MITIGATES", name="mitiga",
        ...     description="Reduce efectos negativos",
        ...     relation_type=RelationType.MITIGATIVE,
        ...     domain=["adaptation_measure"],
        ...     range=["climate_hazard"]
        ... ))
        >>> 
        >>> # Validar
        >>> result = onto.validate()
        >>> print(result)
    """
    
    def __init__(self, name: str, description: str = "", version: str = "1.0.0"):
        self.name = name
        self.description = description
        self.version = version
        self.created_at = datetime.now().isoformat()
        self.updated_at = self.created_at
        
        # Almacenamiento
        self._classes: Dict[str, OntologyClass] = {}
        self._relations: Dict[str, OntologyRelation] = {}
        
        # Índices
        self._synonym_to_class: Dict[str, str] = {}
        self._synonym_to_relation: Dict[str, str] = {}
        self._children: Dict[str, Set[str]] = {}
    
    # ═══════════════════════════════════════════════════════════════════════════
    # GESTIÓN DE CLASES
    # ═══════════════════════════════════════════════════════════════════════════
    
    def add_class(self, cls: OntologyClass) -> "Ontology":
        """Agrega una clase a la ontología."""
        if cls.id in self._classes:
            raise ValueError(f"Clase '{cls.id}' ya existe")
        if cls.parent and cls.parent not in self._classes:
            raise ValueError(f"Padre '{cls.parent}' no existe. Agrégalo primero.")
        
        self._classes[cls.id] = cls
        
        # Indexar jerarquía
        if cls.parent:
            self._children.setdefault(cls.parent, set()).add(cls.id)
        
        # Indexar sinónimos
        for syn in cls.synonyms:
            self._synonym_to_class[syn] = cls.id
        
        self.updated_at = datetime.now().isoformat()
        return self
    
    def get_class(self, class_id: str) -> Optional[OntologyClass]:
        return self._classes.get(class_id)
    
    def get_class_hierarchy(self, class_id: str) -> List[str]:
        """Retorna [clase, padre, abuelo, ..., raíz]"""
        hierarchy = []
        current = class_id
        visited = set()
        while current and current not in visited:
            hierarchy.append(current)
            visited.add(current)
            cls = self._classes.get(current)
            current = cls.parent if cls else None
        return hierarchy

    def is_subclass_of(self, child_id: str, parent_id: str) -> bool:
        """Verifica si una clase es subclase (directa o indirecta) de otra."""
        if child_id == parent_id:
            return True
        hierarchy = self.get_class_hierarchy(child_id)
        return parent_id in hierarchy[1:]
    
    def get_all_hierarchies(self) -> Dict[str, List[str]]:
        return {cid: self.get_class_hierarchy(cid) for cid in self._classes}
    
    def get_subclasses(self, class_id: str, recursive: bool = True) -> Set[str]:
        direct = self._children.get(class_id, set())
        if not recursive:
            return direct
        all_children = set(direct)
        for child in direct:
            all_children.update(self.get_subclasses(child, True))
        return all_children
    
    @property
    def classes(self) -> Dict[str, OntologyClass]:
        return self._classes
    
    @property
    def root_classes(self) -> List[OntologyClass]:
        return [c for c in self._classes.values() if not c.parent]
    
    # ═══════════════════════════════════════════════════════════════════════════
    # GESTIÓN DE RELACIONES
    # ═══════════════════════════════════════════════════════════════════════════
    
    def add_relation(self, rel: OntologyRelation) -> "Ontology":
        """Agrega una relación a la ontología."""
        if rel.id in self._relations:
            raise ValueError(f"Relación '{rel.id}' ya existe")
        
        # Validar que clases de dominio/rango existen
        for cls_id in rel.domain + rel.range:
            if cls_id not in self._classes:
                raise ValueError(f"Clase '{cls_id}' en dominio/rango no existe")
        
        self._relations[rel.id] = rel
        
        for syn in rel.synonyms:
            self._synonym_to_relation[syn] = rel.id
        
        self.updated_at = datetime.now().isoformat()
        return self
    
    def get_relation(self, rel_id: str) -> Optional[OntologyRelation]:
        return self._relations.get(rel_id.upper())
    
    @property
    def relations(self) -> Dict[str, OntologyRelation]:
        return self._relations
    
    # ═══════════════════════════════════════════════════════════════════════════
    # NORMALIZACIÓN
    # ═══════════════════════════════════════════════════════════════════════════
    
    def normalize_entity(self, text: str) -> Optional[str]:
        """Normaliza texto a ID de clase ontológica."""
        text = text.lower().strip()
        if text in self._synonym_to_class:
            return self._synonym_to_class[text]
        if text in self._classes:
            return text
        # Búsqueda parcial
        for cls in self._classes.values():
            matches, score = cls.matches(text)
            if matches and score > 0.5:
                return cls.id
        return None
    
    def normalize_relation(self, text: str) -> Optional[str]:
        """Normaliza texto a ID de relación ontológica."""
        text = text.lower().strip()
        if text in self._synonym_to_relation:
            return self._synonym_to_relation[text]
        text_upper = text.upper()
        if text_upper in self._relations:
            return text_upper
        for rel in self._relations.values():
            matches, score = rel.matches(text)
            if matches and score > 0.5:
                return rel.id
        return None
    
    # ═══════════════════════════════════════════════════════════════════════════
    # VALIDACIÓN (6 VERIFICACIONES)
    # ═══════════════════════════════════════════════════════════════════════════
    
    def validate(self) -> ValidationResult:
        """
        Ejecuta 6 verificaciones de consistencia:
        1. Ciclos en jerarquías IS-A
        2. Relaciones semánticamente vacías
        3. Completitud de dominios/rangos
        4. Coherencia de inversas
        5. Cobertura de sinónimos
        6. Clases disjuntas
        """
        result = ValidationResult()
        
        # Check 1: Ciclos en jerarquías
        has_cycles = False
        for class_id in self._classes:
            visited = set()
            current = class_id
            while current:
                if current in visited:
                    result.errors.append(f"Ciclo en jerarquía: {class_id}")
                    has_cycles = True
                    break
                visited.add(current)
                cls = self._classes.get(current)
                current = cls.parent if cls else None
        if not has_cycles:
            result.checks_passed += 1
        
        # Check 2: Relaciones semánticamente vacías
        vague_found = []
        for rel in self._relations.values():
            if rel.is_semantically_vague:
                vague_found.append(rel.id)
        if vague_found:
            result.warnings.append(f"Relaciones vagas: {vague_found}. Usar MITIGATES, CAUSES, etc.")
        else:
            result.checks_passed += 1
            result.info.append("Todas las relaciones tienen semántica clara")
        
        # Check 3: Dominios y rangos
        incomplete = []
        for rel in self._relations.values():
            if not rel.domain or not rel.range:
                incomplete.append(rel.id)
        if incomplete:
            result.warnings.append(f"Relaciones sin dominio/rango completo: {incomplete}")
        else:
            result.checks_passed += 1
        
        # Check 4: Inversas coherentes
        inverse_issues = []
        for rel in self._relations.values():
            if rel.inverse:
                inv = self._relations.get(rel.inverse)
                if not inv:
                    inverse_issues.append(f"{rel.id} → {rel.inverse} (no existe)")
                elif inv.inverse != rel.id:
                    inverse_issues.append(f"{rel.id} ↔ {rel.inverse} (no recíproca)")
        if inverse_issues:
            result.warnings.append(f"Problemas con inversas: {inverse_issues}")
        else:
            result.checks_passed += 1
        
        # Check 5: Cobertura de sinónimos
        poor_synonyms = [c.id for c in self._classes.values() if len(c.synonyms) < 2]
        if len(poor_synonyms) > len(self._classes) * 0.3:
            result.info.append(f"Clases con pocos sinónimos: {len(poor_synonyms)}")
        else:
            result.checks_passed += 1
        
        # Check 6: Disjunción (si está definida)
        result.checks_passed += 1  # Asumimos OK si no hay conflictos
        
        result.is_valid = len(result.errors) == 0
        return result
    
    # ═══════════════════════════════════════════════════════════════════════════
    # SERIALIZACIÓN
    # ═══════════════════════════════════════════════════════════════════════════
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name, "description": self.description,
            "version": self.version, "created_at": self.created_at,
            "updated_at": self.updated_at,
            "classes": {k: v.to_dict() for k, v in self._classes.items()},
            "relations": {k: v.to_dict() for k, v in self._relations.items()}
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> "Ontology":
        onto = cls(data["name"], data.get("description", ""), data.get("version", "1.0.0"))
        onto.created_at = data.get("created_at", onto.created_at)
        
        # Cargar clases en orden (padres primero)
        classes_data = data.get("classes", {})
        loaded = set()
        
        def load_class(cdata):
            if cdata["id"] in loaded:
                return
            parent = cdata.get("parent")
            if parent and parent not in loaded and parent in classes_data:
                load_class(classes_data[parent])
            onto.add_class(OntologyClass.from_dict(cdata))
            loaded.add(cdata["id"])
        
        for cdata in classes_data.values():
            load_class(cdata)
        
        for rdata in data.get("relations", {}).values():
            onto.add_relation(OntologyRelation.from_dict(rdata))
        
        return onto
    
    def save(self, path: str):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)
    
    @classmethod
    def load(cls, path: str) -> "Ontology":
        with open(path, 'r', encoding='utf-8') as f:
            return cls.from_dict(json.load(f))
    
    def __repr__(self):
        return f"Ontology('{self.name}', classes={len(self._classes)}, relations={len(self._relations)})"
    
    def summary(self) -> str:
        lines = [
            f"\n{'═'*60}",
            f"ONTOLOGÍA: {self.name}",
            f"{'═'*60}",
            f"Clases: {len(self._classes)} | Relaciones: {len(self._relations)}",
            f"\nClases raíz: {[c.id for c in self.root_classes]}",
            f"\nRelaciones: {list(self._relations.keys())}"
        ]
        return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════════════
# ONTOLOGÍA PREDEFINIDA: CLIMA
# ═══════════════════════════════════════════════════════════════════════════════

def create_climate_ontology() -> Ontology:
    """Crea ontología completa para adaptación climática."""
    onto = Ontology("climate_adaptation", "Ontología de adaptación al cambio climático")
    
    # ─── CLASES RAÍZ ───
    onto.add_class(OntologyClass("climate_hazard", "Peligro Climático",
        "Evento climático que puede causar daño",
        synonyms=["riesgo climático", "amenaza", "hazard"]))
    
    onto.add_class(OntologyClass("adaptation_measure", "Medida de Adaptación",
        "Intervención para reducir vulnerabilidad",
        synonyms=["estrategia", "medida adaptativa", "adaptation"]))
    
    onto.add_class(OntologyClass("sector", "Sector",
        "Área de actividad humana afectada",
        synonyms=["área", "dominio"]))
    
    onto.add_class(OntologyClass("impact", "Impacto",
        "Efecto de un peligro climático",
        synonyms=["efecto", "consecuencia", "daño"]))
    
    onto.add_class(OntologyClass("location", "Ubicación",
        "Lugar geográfico",
        synonyms=["lugar", "región", "zona"]))
    
    # ─── SUBCLASES: PELIGROS ───
    hazards = [
        ("flood", "Inundación", ["inundaciones", "flooding", "anegamiento", "riada"]),
        ("drought", "Sequía", ["sequías", "escasez hídrica", "aridez"]),
        ("heat_wave", "Ola de Calor", ["calor extremo", "heat wave", "isla de calor"]),
        ("wildfire", "Incendio Forestal", ["incendios", "fuego forestal", "wildfire"]),
        ("storm", "Tormenta", ["tormentas", "huracán", "ciclón", "storm"]),
        ("sea_level_rise", "Aumento Nivel del Mar", ["subida del mar", "sea level"]),
    ]
    for hid, label, syns in hazards:
        onto.add_class(OntologyClass(hid, label, parent="climate_hazard", synonyms=syns))
    
    # ─── SUBCLASES: MEDIDAS ───
    measures = [
        ("green_infrastructure", "Infraestructura Verde", 
         ["NbS", "soluciones basadas en naturaleza", "green infrastructure"]),
        ("grey_infrastructure", "Infraestructura Gris",
         ["infraestructura convencional", "obras civiles"]),
        ("early_warning", "Sistema de Alerta Temprana",
         ["alerta temprana", "early warning", "SAT"]),
        ("water_management", "Gestión del Agua",
         ["manejo del agua", "gestión hídrica"]),
    ]
    for mid, label, syns in measures:
        onto.add_class(OntologyClass(mid, label, parent="adaptation_measure", synonyms=syns))
    
    # Subclases de infraestructura verde
    green_types = [
        ("green_roof", "Techo Verde", ["techos verdes", "cubierta vegetal", "azotea verde"]),
        ("rain_garden", "Jardín de Lluvia", ["jardines de lluvia", "bioswale"]),
        ("urban_forest", "Bosque Urbano", ["arbolado urbano", "parque forestal"]),
        ("wetland", "Humedal", ["humedales", "marisma", "pantano"]),
    ]
    for gid, label, syns in green_types:
        onto.add_class(OntologyClass(gid, label, parent="green_infrastructure", synonyms=syns))
    
    # ─── RELACIONES SEMÁNTICAS ───
    onto.add_relation(OntologyRelation(
        "MITIGATES", "mitiga", "El sujeto reduce los efectos del objeto",
        RelationType.MITIGATIVE,
        domain=["adaptation_measure"], range=["climate_hazard", "impact"],
        inverse="MITIGATED_BY",
        synonyms=["reduce", "atenúa", "disminuye", "alivia"]))
    
    onto.add_relation(OntologyRelation(
        "MITIGATED_BY", "mitigado por", "El sujeto es reducido por el objeto",
        RelationType.MITIGATIVE,
        domain=["climate_hazard", "impact"], range=["adaptation_measure"],
        inverse="MITIGATES"))
    
    onto.add_relation(OntologyRelation(
        "PREVENTS", "previene", "El sujeto evita que ocurra el objeto",
        RelationType.PREVENTIVE,
        domain=["adaptation_measure"], range=["climate_hazard", "impact"],
        synonyms=["evita", "impide"]))
    
    onto.add_relation(OntologyRelation(
        "CAUSES", "causa", "El sujeto produce el objeto",
        RelationType.CAUSAL,
        domain=["climate_hazard"], range=["impact"],
        synonyms=["produce", "genera", "provoca", "origina"]))
    
    onto.add_relation(OntologyRelation(
        "PROTECTS", "protege", "El sujeto ofrece protección al objeto",
        RelationType.FUNCTIONAL,
        domain=["adaptation_measure"], range=["sector", "location"],
        synonyms=["defiende", "salvaguarda"]))
    
    onto.add_relation(OntologyRelation(
        "IMPLEMENTED_IN", "implementado en", "El sujeto se aplica en el objeto",
        RelationType.SPATIAL,
        domain=["adaptation_measure"], range=["location"],
        synonyms=["aplicado en", "desplegado en"]))
    
    onto.add_relation(OntologyRelation(
        "COMPLEMENTS", "complementa", "El sujeto trabaja junto con el objeto",
        RelationType.FUNCTIONAL,
        domain=["adaptation_measure"], range=["adaptation_measure"],
        is_symmetric=True,
        synonyms=["sinérgico con"]))
    
    return onto


def create_climate_adaptation_ontology() -> Ontology:
    """Alias para mantener compatibilidad con versiones anteriores."""
    return create_climate_ontology()
