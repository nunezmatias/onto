"""
═══════════════════════════════════════════════════════════════════════════════
MÓDULO: ontorag.extraction
═══════════════════════════════════════════════════════════════════════════════

Extracción de tripletas (sujeto, relación, objeto) desde texto.

Componentes:
- TripletExtractor: Extracción guiada por LLM usando la ontología
- RuleBasedExtractor: Extracción basada en patrones
- ExtractionResult: Resultado estructurado

El extractor usa la ontología para:
1. Generar prompts con clases y relaciones válidas
2. Normalizar entidades extraídas
3. Validar tripletas contra el esquema
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Callable, TYPE_CHECKING
import re
import json

if TYPE_CHECKING:
    from ontorag.ontology import Ontology
    from ontorag.graph import Node, Edge


# ═══════════════════════════════════════════════════════════════════════════════
# TIPOS
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class ExtractedEntity:
    """Entidad extraída del texto."""
    text: str
    normalized_id: Optional[str] = None
    entity_type: Optional[str] = None
    confidence: float = 1.0
    span: tuple = (0, 0)  # (start, end) en el texto


@dataclass
class ExtractedTriple:
    """Tripleta extraída del texto."""
    subject: ExtractedEntity
    relation: str
    object: ExtractedEntity
    confidence: float = 1.0
    source_text: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "subject": {
                "text": self.subject.text,
                "normalized_id": self.subject.normalized_id,
                "type": self.subject.entity_type,
            },
            "relation": self.relation,
            "object": {
                "text": self.object.text,
                "normalized_id": self.object.normalized_id,
                "type": self.object.entity_type,
            },
            "confidence": self.confidence,
            "source_text": self.source_text,
        }


@dataclass
class ExtractionResult:
    """Resultado de extracción completo."""
    triples: List[ExtractedTriple] = field(default_factory=list)
    entities: List[ExtractedEntity] = field(default_factory=list)
    source_text: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "num_triples": len(self.triples),
            "num_entities": len(self.entities),
            "triples": [t.to_dict() for t in self.triples],
        }
    
    def summary(self) -> str:
        lines = [
            f"Extracción: {len(self.triples)} tripletas, {len(self.entities)} entidades",
            "",
            "Tripletas encontradas:",
        ]
        for t in self.triples[:10]:
            lines.append(f"  • ({t.subject.text}) --[{t.relation}]--> ({t.object.text})")
        
        if len(self.triples) > 10:
            lines.append(f"  ... y {len(self.triples) - 10} más")
        
        return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════════════
# TRIPLET EXTRACTOR (LLM-based)
# ═══════════════════════════════════════════════════════════════════════════════

class TripletExtractor:
    """
    Extractor de tripletas guiado por LLM.
    
    Usa la ontología para:
    1. Generar prompts con las clases y relaciones disponibles
    2. Normalizar entidades contra sinónimos
    3. Validar tipos de relación
    
    Example:
        >>> extractor = TripletExtractor(ontology, llm_function)
        >>> result = extractor.extract(
        ...     "Los techos verdes mitigan el efecto de isla de calor urbana."
        ... )
        >>> print(result.triples[0])
        (techos_verdes) --[MITIGATES]--> (urban_heat_island)
    """
    
    EXTRACTION_PROMPT = '''Eres un experto en extracción de información. Tu tarea es extraer tripletas (sujeto, relación, objeto) del texto dado.

## ONTOLOGÍA DISPONIBLE

### Clases de entidades válidas:
{classes}

### Relaciones válidas:
{relations}

## INSTRUCCIONES

1. Extrae SOLO tripletas que puedas mapear a las clases y relaciones de la ontología
2. Usa los IDs de las relaciones tal cual (ej: MITIGATES, CAUSES, PREVENTS)
3. Normaliza las entidades al ID de clase más cercano
4. Solo extrae información explícitamente mencionada, no inferencias
5. Responde SOLO con JSON válido

## FORMATO DE RESPUESTA

```json
{{
  "triples": [
    {{
      "subject": {{"text": "texto original", "type": "class_id"}},
      "relation": "RELATION_ID",
      "object": {{"text": "texto original", "type": "class_id"}},
      "confidence": 0.95
    }}
  ]
}}
```

## TEXTO A PROCESAR

{text}

Extrae las tripletas:'''
    
    def __init__(
        self,
        ontology: "Ontology",
        llm_function: Callable[[str, str], str] = None,
    ):
        self.ontology = ontology
        self.llm = llm_function
    
    def _build_prompt(self, text: str) -> str:
        """Construye el prompt con información de la ontología."""
        # Formatear clases
        classes_text = ""
        for cls in self.ontology.classes.values():
            synonyms = ", ".join(cls.synonyms[:5])
            classes_text += f"- {cls.id}: {cls.name} (sinónimos: {synonyms})\n"
        
        # Formatear relaciones
        relations_text = ""
        for rel in self.ontology.relations.values():
            domain = ", ".join(rel.domain[:3]) if rel.domain else "cualquiera"
            range_ = ", ".join(rel.range[:3]) if rel.range else "cualquiera"
            relations_text += f"- {rel.id}: {rel.description[:50]} (dominio: {domain}, rango: {range_})\n"
        
        return self.EXTRACTION_PROMPT.format(
            classes=classes_text,
            relations=relations_text,
            text=text
        )
    
    def _parse_response(self, response: str) -> List[Dict]:
        """Parsea la respuesta JSON del LLM."""
        # Intentar extraer JSON del response
        try:
            # Buscar bloque JSON
            json_match = re.search(r'```json\s*(.*?)\s*```', response, re.DOTALL)
            if json_match:
                json_str = json_match.group(1)
            else:
                json_str = response
            
            data = json.loads(json_str)
            return data.get("triples", [])
        except json.JSONDecodeError:
            return []
    
    def _normalize_entity(self, text: str, type_hint: str = None) -> ExtractedEntity:
        """Normaliza una entidad usando la ontología."""
        normalized_id = self.ontology.normalize_entity(text)
        
        entity_type = type_hint
        if normalized_id:
            entity_type = normalized_id
        
        return ExtractedEntity(
            text=text,
            normalized_id=normalized_id,
            entity_type=entity_type,
        )
    
    def extract(self, text: str) -> ExtractionResult:
        """
        Extrae tripletas de un texto.
        
        Args:
            text: Texto a procesar
            
        Returns:
            ExtractionResult con las tripletas encontradas
        """
        if not self.llm:
            # Sin LLM, retornar resultado vacío
            return ExtractionResult(source_text=text)
        
        # Construir prompt
        prompt = self._build_prompt(text)
        
        # Llamar al LLM
        response = self.llm("Eres un extractor de información preciso.", prompt)
        
        # Parsear respuesta
        raw_triples = self._parse_response(response)
        
        # Construir resultado
        result = ExtractionResult(source_text=text)
        entities_seen = set()
        
        for raw in raw_triples:
            try:
                # Normalizar entidades
                subject = self._normalize_entity(
                    raw["subject"]["text"],
                    raw["subject"].get("type")
                )
                obj = self._normalize_entity(
                    raw["object"]["text"],
                    raw["object"].get("type")
                )
                
                # Normalizar relación
                relation = raw["relation"].upper()
                normalized_rel = self.ontology.normalize_relation(relation)
                if normalized_rel:
                    relation = normalized_rel
                
                # Crear tripleta
                triple = ExtractedTriple(
                    subject=subject,
                    relation=relation,
                    object=obj,
                    confidence=raw.get("confidence", 0.8),
                    source_text=text[:200],
                )
                result.triples.append(triple)
                
                # Registrar entidades
                for entity in [subject, obj]:
                    if entity.text not in entities_seen:
                        result.entities.append(entity)
                        entities_seen.add(entity.text)
                        
            except (KeyError, TypeError):
                continue
        
        return result
    
    def extract_batch(self, texts: List[str]) -> List[ExtractionResult]:
        """Extrae tripletas de múltiples textos."""
        return [self.extract(text) for text in texts]


# ═══════════════════════════════════════════════════════════════════════════════
# RULE-BASED EXTRACTOR
# ═══════════════════════════════════════════════════════════════════════════════

class RuleBasedExtractor:
    """
    Extractor basado en patrones y reglas.
    
    No requiere LLM. Usa patrones regex y la ontología
    para encontrar tripletas.
    
    Útil para:
    - Extracción rápida sin costo de API
    - Dominios con patrones bien definidos
    - Complementar la extracción con LLM
    """
    
    # Patrones de relación comunes
    DEFAULT_PATTERNS = [
        # mitiga, reduce, atenúa (soporta singular y plural)
        (r'(.+?)\s+(mitig(?:a|an)|reduce(?:n)?|atenú(?:a|an)|disminuye(?:n)?)\s+(.+)', 'MITIGATES'),
        # causa, produce, genera (soporta singular y plural)
        (r'(.+?)\s+(caus(?:a|an)|produce(?:n)?|genera(?:n)?|provoc(?:a|an))\s+(.+)', 'CAUSES'),
        # previene, evita, impide
        (r'(.+?)\s+(previene|evita|impide)\s+(.+)', 'PREVENTS'),
        # protege, defiende
        (r'(.+?)\s+(protege|defiende|salvaguarda)\s+(.+)', 'PROTECTS'),
        # afecta, impacta
        (r'(.+?)\s+(afecta|impacta)\s+(.+)', 'AFFECTS'),
        # implementado en, ubicado en
        (r'(.+?)\s+(implementado en|ubicado en|localizado en)\s+(.+)', 'LOCATED_IN'),
        # es un tipo de, es un
        (r'(.+?)\s+(es un tipo de|es una?)\s+(.+)', 'IS_A'),
        # es parte de
        (r'(.+?)\s+(es parte de|forma parte de|pertenece a)\s+(.+)', 'PART_OF'),
    ]
    
    def __init__(
        self,
        ontology: "Ontology" = None,
        patterns: List[tuple] = None,
    ):
        self.ontology = ontology
        self.patterns = patterns or self.DEFAULT_PATTERNS
        
        # Compilar patrones
        self.compiled_patterns = [
            (re.compile(pattern, re.IGNORECASE), relation)
            for pattern, relation in self.patterns
        ]
    
    def extract(self, text: str) -> ExtractionResult:
        """
        Extrae tripletas usando patrones.
        
        Args:
            text: Texto a procesar
            
        Returns:
            ExtractionResult
        """
        result = ExtractionResult(source_text=text)
        
        # Dividir en oraciones
        sentences = re.split(r'[.!?]', text)
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
            
            for pattern, relation in self.compiled_patterns:
                match = pattern.search(sentence)
                if match:
                    subject_text = match.group(1).strip()
                    object_text = match.group(3).strip()
                    
                    # Normalizar si hay ontología
                    subject = ExtractedEntity(text=subject_text)
                    obj = ExtractedEntity(text=object_text)
                    
                    if self.ontology:
                        subject.normalized_id = self.ontology.normalize_entity(subject_text)
                        subject.entity_type = subject.normalized_id
                        obj.normalized_id = self.ontology.normalize_entity(object_text)
                        obj.entity_type = obj.normalized_id
                    
                    triple = ExtractedTriple(
                        subject=subject,
                        relation=relation,
                        object=obj,
                        confidence=0.7,  # Menor confianza que LLM
                        source_text=sentence,
                    )
                    result.triples.append(triple)
        
        return result


# ═══════════════════════════════════════════════════════════════════════════════
# UTILIDADES
# ═══════════════════════════════════════════════════════════════════════════════

def extraction_to_graph_elements(
    extraction: ExtractionResult,
    ontology: "Ontology" = None,
) -> tuple:
    """
    Convierte resultado de extracción a elementos del grafo.
    
    Args:
        extraction: Resultado de extracción
        ontology: Ontología para validación
        
    Returns:
        Tupla (lista de Nodes, lista de Edges)
    """
    from ontorag.graph import Node, Edge
    
    nodes = []
    edges = []
    seen_nodes = set()
    
    for triple in extraction.triples:
        # Crear nodo sujeto
        subj_id = triple.subject.normalized_id or triple.subject.text.lower().replace(" ", "_")
        if subj_id not in seen_nodes:
            nodes.append(Node(
                id=subj_id,
                name=triple.subject.text,
                node_type=triple.subject.entity_type or "unknown",
                text_content=triple.source_text,
            ))
            seen_nodes.add(subj_id)
        
        # Crear nodo objeto
        obj_id = triple.object.normalized_id or triple.object.text.lower().replace(" ", "_")
        if obj_id not in seen_nodes:
            nodes.append(Node(
                id=obj_id,
                name=triple.object.text,
                node_type=triple.object.entity_type or "unknown",
                text_content=triple.source_text,
            ))
            seen_nodes.add(obj_id)
        
        # Crear arista
        edges.append(Edge(
            source=subj_id,
            target=obj_id,
            relation=triple.relation,
            text_context=triple.source_text,
            confidence=triple.confidence,
        ))
    
    return nodes, edges
