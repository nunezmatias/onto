"""
Rule-Based Triplet Extraction

This module provides rule-based extraction of entities and relations from text.
Uses dictionary matching and pattern-based relation detection.

Features:
- Dictionary-based entity recognition
- Pattern-based relation extraction
- No external API calls required
- Fully reproducible results

Example:
    >>> from ontorag.extraction import RuleBasedExtractor
    >>> 
    >>> extractor = RuleBasedExtractor()
    >>> result = extractor.extract(
    ...     "Green roofs reduce urban heat island effects by 2-3°C"
    ... )
    >>> 
    >>> print(result.entities)
    >>> print(result.triplets)
"""

from __future__ import annotations

import re
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from .dictionaries import (
    CLIMATE_ENTITIES,
    RELATION_PATTERNS,
    EntityDictionary,
    EntityMatch,
)

logger = logging.getLogger(__name__)


@dataclass
class ExtractedEntity:
    """
    An extracted entity.
    
    Attributes:
        text: Original text
        type: Entity type (MEASURE, HAZARD, etc.)
        normalized: Normalized form (snake_case)
        confidence: Extraction confidence
        source: How it was extracted (dictionary, llm, etc.)
    """
    text: str
    type: str
    normalized: str = ""
    confidence: float = 1.0
    source: str = "dictionary"
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        if not self.normalized:
            self.normalized = self.text.lower().replace(" ", "_").replace("-", "_")
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "text": self.text,
            "type": self.type,
            "normalized": self.normalized,
            "confidence": self.confidence,
            "source": self.source,
            "metadata": self.metadata,
        }


@dataclass
class ExtractedTriplet:
    """
    An extracted triplet (subject, relation, object).
    
    Attributes:
        subject: Subject entity
        relation: Relation type
        object: Object entity
        confidence: Extraction confidence
        evidence: Text span that supports this triplet
    """
    subject: str
    relation: str
    object: str
    confidence: float = 1.0
    evidence: str = ""
    source: str = "pattern"
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "subject": self.subject,
            "relation": self.relation,
            "object": self.object,
            "confidence": self.confidence,
            "evidence": self.evidence,
            "source": self.source,
            "metadata": self.metadata,
        }
    
    def to_tuple(self) -> Tuple[str, str, str]:
        return (self.subject, self.relation, self.object)


@dataclass
class ExtractionResult:
    """
    Result of extraction from a document.
    
    Attributes:
        entities: List of extracted entities
        triplets: List of extracted triplets
        source_text: Original text
        metadata: Additional information
    """
    entities: List[ExtractedEntity] = field(default_factory=list)
    triplets: List[ExtractedTriplet] = field(default_factory=list)
    source_text: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "entities": [e.to_dict() for e in self.entities],
            "triplets": [t.to_dict() for t in self.triplets],
            "source_text": self.source_text[:500] if self.source_text else "",
            "metadata": self.metadata,
        }
    
    def summary(self) -> str:
        """Generate summary string."""
        lines = [
            f"Entities: {len(self.entities)}",
            f"Triplets: {len(self.triplets)}",
        ]
        
        if self.entities:
            by_type = {}
            for e in self.entities:
                by_type.setdefault(e.type, []).append(e.text)
            
            lines.append("\nEntities by type:")
            for etype, ents in by_type.items():
                lines.append(f"  {etype}: {', '.join(ents[:5])}")
        
        if self.triplets:
            lines.append("\nSample triplets:")
            for t in self.triplets[:5]:
                lines.append(f"  ({t.subject}, {t.relation}, {t.object})")
        
        return "\n".join(lines)


class RuleBasedExtractor:
    """
    Extract entities and relations using rules and dictionaries.
    
    This extractor uses:
    1. Dictionary matching for entity recognition
    2. Regex patterns for relation extraction
    3. Co-occurrence heuristics for additional triplets
    
    Advantages:
    - No API calls needed
    - Fast and deterministic
    - Full control over what is extracted
    
    Limitations:
    - Only finds predefined entities
    - Limited relation patterns
    - No semantic understanding
    
    Example:
        >>> extractor = RuleBasedExtractor()
        >>> 
        >>> # Extract from text
        >>> result = extractor.extract(text)
        >>> 
        >>> # Extract from document
        >>> result = extractor.extract_from_document({
        ...     "title": "...",
        ...     "abstract": "..."
        ... })
        >>> 
        >>> # Custom dictionary
        >>> extractor = RuleBasedExtractor(
        ...     entity_dict=my_custom_dict,
        ...     relation_patterns=my_patterns
        ... )
    """
    
    def __init__(
        self,
        entity_dict: Dict[str, List[str]] = None,
        relation_patterns: Dict[str, List[str]] = None,
        extract_quantities: bool = True,
    ):
        """
        Initialize the extractor.
        
        Args:
            entity_dict: Custom entity dictionary (default: CLIMATE_ENTITIES)
            relation_patterns: Custom relation patterns
            extract_quantities: Whether to extract numerical values
        """
        entity_dict = entity_dict or CLIMATE_ENTITIES
        self.entity_dictionary = EntityDictionary(entity_dict)
        self.relation_patterns = relation_patterns or RELATION_PATTERNS
        self.extract_quantities = extract_quantities
        
        # Compile relation patterns
        self._compiled_relations = {}
        for rel_type, patterns in self.relation_patterns.items():
            self._compiled_relations[rel_type] = [
                re.compile(p, re.IGNORECASE) for p in patterns
            ]
    
    def extract(self, text: str) -> ExtractionResult:
        """
        Extract entities and triplets from text.
        
        Args:
            text: Input text
            
        Returns:
            ExtractionResult with entities and triplets
        """
        result = ExtractionResult(source_text=text)
        
        if not text:
            return result
        
        # Step 1: Extract entities
        entity_matches = self.entity_dictionary.find_entities(text)
        
        for match in entity_matches:
            result.entities.append(ExtractedEntity(
                text=match.text,
                type=match.entity_type,
                normalized=match.normalized or match.text.lower().replace(" ", "_"),
                confidence=match.confidence,
                source="dictionary",
            ))
        
        # Step 2: Extract quantities if enabled
        if self.extract_quantities:
            quantities = self._extract_quantities(text)
            for q in quantities:
                result.entities.append(q)
        
        # Step 3: Extract relations using patterns
        pattern_triplets = self._extract_pattern_relations(text, entity_matches)
        result.triplets.extend(pattern_triplets)
        
        # Step 4: Extract co-occurrence relations
        cooccur_triplets = self._extract_cooccurrence_relations(text, entity_matches)
        result.triplets.extend(cooccur_triplets)
        
        # Remove duplicate triplets
        result.triplets = self._deduplicate_triplets(result.triplets)
        
        return result
    
    def extract_from_document(
        self,
        document: Dict[str, Any],
        text_fields: List[str] = None
    ) -> ExtractionResult:
        """
        Extract from a document dictionary.
        
        Args:
            document: Document with text fields
            text_fields: Fields to extract from (default: title, abstract)
            
        Returns:
            ExtractionResult
        """
        text_fields = text_fields or ["title", "abstract"]
        
        # Combine text from specified fields
        texts = []
        for field in text_fields:
            value = document.get(field, "")
            if value:
                texts.append(str(value))
        
        combined_text = " ".join(texts)
        result = self.extract(combined_text)
        
        # Add document metadata
        result.metadata["document_id"] = document.get("id", document.get("corpus_id", ""))
        result.metadata["source"] = document.get("source", "unknown")
        
        return result
    
    def _extract_quantities(self, text: str) -> List[ExtractedEntity]:
        """Extract numerical quantities from text."""
        quantities = []
        
        # Pattern for quantities with units
        quantity_patterns = [
            # Temperature: 2-3°C, 5 degrees
            r'(\d+(?:\.\d+)?(?:\s*-\s*\d+(?:\.\d+)?)?)\s*°?[CcFf](?:elsius|ahrenheit)?',
            # Percentage: 70%, 20-30%
            r'(\d+(?:\.\d+)?(?:\s*-\s*\d+(?:\.\d+)?)?)\s*%',
            # Currency: $2.3 billion, €50 million
            r'[$€£]\s*(\d+(?:\.\d+)?)\s*(million|billion|thousand)?',
            # Area: 200,000 square meters
            r'(\d+(?:,\d{3})*(?:\.\d+)?)\s*(?:square\s+)?(?:meters?|km2?|hectares?)',
            # Time: 15-20 years, 24 hours
            r'(\d+(?:\s*-\s*\d+)?)\s*(?:years?|months?|days?|hours?)',
        ]
        
        for pattern in quantity_patterns:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                quantities.append(ExtractedEntity(
                    text=match.group(0),
                    type="QUANTITY",
                    normalized=match.group(0).lower().replace(" ", "_"),
                    confidence=0.9,
                    source="pattern",
                ))
        
        return quantities
    
    def _extract_pattern_relations(
        self,
        text: str,
        entity_matches: List[EntityMatch]
    ) -> List[ExtractedTriplet]:
        """Extract relations using regex patterns."""
        triplets = []
        
        # Create entity lookup by position
        entity_spans = {}
        for em in entity_matches:
            for pos in range(em.start, em.end):
                entity_spans[pos] = em
        
        for rel_type, patterns in self._compiled_relations.items():
            for pattern in patterns:
                for match in pattern.finditer(text):
                    # Get subject and object from match groups
                    if len(match.groups()) >= 2:
                        subj_text = match.group(1).strip()
                        obj_text = match.group(2).strip()
                        
                        # Normalize
                        subj_norm = subj_text.lower().replace(" ", "_")[:50]
                        obj_norm = obj_text.lower().replace(" ", "_")[:50]
                        
                        triplets.append(ExtractedTriplet(
                            subject=subj_norm,
                            relation=rel_type,
                            object=obj_norm,
                            confidence=0.7,
                            evidence=match.group(0),
                            source="pattern",
                        ))
        
        return triplets
    
    def _extract_cooccurrence_relations(
        self,
        text: str,
        entity_matches: List[EntityMatch]
    ) -> List[ExtractedTriplet]:
        """
        Extract relations based on entity co-occurrence in sentences.
        
        Heuristic: If a MEASURE and HAZARD appear in the same sentence,
        assume MITIGATES relationship.
        """
        triplets = []
        
        # Split into sentences
        sentences = re.split(r'[.!?]+', text)
        
        for sentence in sentences:
            sentence_lower = sentence.lower()
            
            # Find entities in this sentence
            measures = []
            hazards = []
            locations = []
            
            for em in entity_matches:
                if em.text.lower() in sentence_lower:
                    if em.entity_type == "MEASURE":
                        measures.append(em)
                    elif em.entity_type == "HAZARD":
                        hazards.append(em)
                    elif em.entity_type == "LOCATION":
                        locations.append(em)
            
            # Generate MITIGATES triplets
            for measure in measures:
                for hazard in hazards:
                    triplets.append(ExtractedTriplet(
                        subject=measure.normalized or measure.text.lower().replace(" ", "_"),
                        relation="MITIGATES",
                        object=hazard.normalized or hazard.text.lower().replace(" ", "_"),
                        confidence=0.5,
                        evidence=sentence.strip()[:200],
                        source="cooccurrence",
                    ))
            
            # Generate IMPLEMENTED_IN triplets
            for measure in measures:
                for location in locations:
                    triplets.append(ExtractedTriplet(
                        subject=measure.normalized or measure.text.lower().replace(" ", "_"),
                        relation="IMPLEMENTED_IN",
                        object=location.normalized or location.text.lower().replace(" ", "_"),
                        confidence=0.4,
                        evidence=sentence.strip()[:200],
                        source="cooccurrence",
                    ))
        
        return triplets
    
    def _deduplicate_triplets(
        self,
        triplets: List[ExtractedTriplet]
    ) -> List[ExtractedTriplet]:
        """Remove duplicate triplets, keeping highest confidence."""
        seen = {}
        
        for t in triplets:
            key = (t.subject, t.relation, t.object)
            if key not in seen or seen[key].confidence < t.confidence:
                seen[key] = t
        
        return list(seen.values())


__all__ = [
    "ExtractedEntity",
    "ExtractedTriplet",
    "ExtractionResult",
    "RuleBasedExtractor",
]
