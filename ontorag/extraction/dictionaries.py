"""
Entity Dictionaries for Extraction

This module provides configurable dictionaries of domain-specific entities.
Dictionaries are used for rule-based extraction and entity recognition.

Usage:
    >>> from ontorag.extraction.dictionaries import CLIMATE_ENTITIES, EntityDictionary
    >>> 
    >>> # Use predefined climate dictionary
    >>> climate_dict = EntityDictionary(CLIMATE_ENTITIES)
    >>> entities = climate_dict.find_entities("Green roofs reduce urban heat")
    >>> 
    >>> # Create custom dictionary
    >>> my_dict = EntityDictionary({
    ...     "MEASURE": ["treatment A", "therapy B"],
    ...     "CONDITION": ["disease X", "syndrome Y"]
    ... })
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple


# =============================================================================
# PREDEFINED DICTIONARIES
# =============================================================================

CLIMATE_ENTITIES = {
    # ADAPTATION MEASURES
    "MEASURE": [
        # Green infrastructure
        "green roof", "green roofs", "green_roof",
        "rain garden", "rain gardens", "bioswale", "bioswales",
        "permeable pavement", "permeable paving",
        "green infrastructure", "green_infrastructure",
        "blue-green infrastructure",
        
        # Trees and vegetation
        "street tree", "street trees", "tree planting",
        "urban forest", "urban forestry", "tree canopy",
        "urban greening", "urban vegetation",
        
        # Ecosystem-based
        "nature-based solution", "nature-based solutions", "NbS",
        "ecosystem-based adaptation", "EbA",
        "mangrove restoration", "mangrove planting",
        "wetland restoration", "wetland creation",
        "floodplain restoration", "river restoration",
        "coral restoration", "reef restoration",
        
        # Grey infrastructure
        "sea wall", "seawall", "dike", "dikes", "levee", "levees",
        "storm surge barrier", "flood barrier", "flood wall",
        "drainage system", "stormwater system",
        "retention pond", "detention basin",
        "breakwater", "revetment",
        
        # Warning systems
        "early warning system", "early warning",
        "flood forecasting", "flood forecast",
        "drought monitoring", "drought early warning",
        "heat warning", "heat alert system",
        
        # Agriculture
        "drought-resistant crop", "drought resistant",
        "drought-tolerant", "heat-tolerant",
        "rainwater harvesting", "water harvesting",
        "drip irrigation", "micro-irrigation",
        "conservation agriculture", "climate-smart agriculture",
        "crop diversification", "agroforestry",
        
        # Policy and planning
        "managed retreat", "planned relocation",
        "building code", "building codes",
        "zoning regulation", "land use planning",
        "heat action plan", "adaptation plan",
        "flood risk management",
        
        # Insurance
        "parametric insurance", "index-based insurance",
        "crop insurance", "climate insurance",
        "microinsurance",
        
        # Other
        "cooling center", "cooling shelter",
        "shade structure", "shade provision",
        "reflective pavement", "cool pavement",
        "cool roof", "reflective roof",
    ],
    
    # =========================================================================
    # CLIMATE HAZARDS
    # =========================================================================
    "HAZARD": [
        # Heat
        "heat wave", "heat waves", "heatwave", "heatwaves",
        "extreme heat", "heat stress", "thermal stress",
        "urban heat island", "urban heat",
        "heat dome",
        
        # Flooding
        "flood", "floods", "flooding",
        "flash flood", "flash flooding",
        "coastal flood", "coastal flooding",
        "river flood", "riverine flood",
        "pluvial flood", "surface water flood",
        "storm surge", "tidal surge",
        
        # Drought
        "drought", "droughts",
        "water scarcity", "water stress",
        "water shortage", "aridity",
        "desertification",
        
        # Sea level
        "sea level rise", "rising sea level",
        "coastal erosion", "shoreline retreat",
        "saltwater intrusion", "salinization",
        
        # Storms
        "hurricane", "hurricanes",
        "typhoon", "typhoons",
        "cyclone", "tropical cyclone",
        "tropical storm",
        "extreme weather", "severe weather",
        "DANA",
        
        # Fire
        "wildfire", "wildfires",
        "forest fire", "bushfire",
        "wildland fire",
        
        # Cold
        "cold wave", "cold snap",
        "extreme cold", "frost",
        "freeze", "ice storm",
        
        # Other
        "landslide", "mudslide", "debris flow",
        "avalanche",
        "thunderstorm", "lightning",
        "tornado", "tornadoes",
        
        # General
        "climate change", "global warming",
        "climate risk", "climate hazard",
        "climate impact", "climate threat",
        "extreme event", "extreme climate event",
    ],
    
    # =========================================================================
    # LOCATIONS
    # =========================================================================
    "LOCATION": [
        # Europe
        "rotterdam", "amsterdam", "netherlands",
        "copenhagen", "denmark",
        "london", "uk", "united kingdom",
        "paris", "france",
        "barcelona", "madrid", "spain",
        "berlin", "germany",
        "vienna", "austria",
        
        # Asia
        "singapore",
        "tokyo", "japan",
        "bangkok", "thailand",
        "jakarta", "indonesia",
        "manila", "philippines",
        "ho chi minh", "vietnam",
        "dhaka", "bangladesh",
        "mumbai", "delhi", "india",
        "shanghai", "beijing", "china",
        
        # Americas
        "new york", "miami", "new orleans",
        "los angeles", "san francisco",
        "phoenix", "houston",
        "bogota", "colombia",
        "sao paulo", "rio de janeiro", "brazil",
        "mexico city", "mexico",
        "lima", "peru",
        
        # Africa
        "cape town", "south africa",
        "nairobi", "kenya",
        "lagos", "nigeria",
        "cairo", "egypt",
        "sub-saharan africa",
        
        # Oceania
        "sydney", "melbourne", "australia",
        "auckland", "new zealand",
        
        # Generic
        "coastal area", "coastal zone", "coastline",
        "urban area", "city", "metropolitan",
        "rural area", "countryside",
        "low-lying area", "low elevation",
        "floodplain", "river basin",
        "delta", "estuary",
        "island", "small island",
        "mountain", "highlands",
        "arid region", "semi-arid",
    ],
    
    # =========================================================================
    # IMPACTS
    # =========================================================================
    "IMPACT": [
        # Temperature
        "temperature reduction", "cooling effect",
        "temperature decrease", "thermal comfort",
        
        # Flood
        "flood protection", "flood reduction",
        "flood control", "flood mitigation",
        "runoff reduction", "water retention",
        
        # Economic
        "damage reduction", "loss reduction",
        "cost savings", "economic benefit",
        "cost-benefit", "return on investment",
        "payback period",
        
        # Social
        "mortality reduction", "death prevention",
        "life saving", "casualty reduction",
        "health improvement", "wellbeing",
        "livelihood protection",
        
        # Environmental
        "erosion control", "erosion prevention",
        "habitat creation", "biodiversity",
        "carbon sequestration", "carbon storage",
        "air quality", "water quality",
        
        # Agricultural
        "yield improvement", "crop yield",
        "food security", "agricultural productivity",
        "harvest stability",
        
        # Other
        "energy savings", "energy efficiency",
        "property value", "aesthetic benefit",
        "recreation", "quality of life",
    ],
}


# =============================================================================
# RELATION PATTERNS
# =============================================================================

RELATION_PATTERNS = {
    "MITIGATES": [
        r"(\w+[\w\s]*?)\s+mitigates?\s+(\w+[\w\s]*)",
        r"(\w+[\w\s]*?)\s+reduces?\s+(?:the\s+)?(?:effects?\s+of\s+)?(\w+[\w\s]*)",
        r"(\w+[\w\s]*?)\s+lessens?\s+(\w+[\w\s]*)",
        r"(\w+[\w\s]*?)\s+alleviates?\s+(\w+[\w\s]*)",
        r"(\w+[\w\s]*?)\s+(?:can\s+)?help\s+reduce\s+(\w+[\w\s]*)",
    ],
    
    "CAUSES": [
        r"(\w+[\w\s]*?)\s+causes?\s+(\w+[\w\s]*)",
        r"(\w+[\w\s]*?)\s+leads?\s+to\s+(\w+[\w\s]*)",
        r"(\w+[\w\s]*?)\s+results?\s+in\s+(\w+[\w\s]*)",
        r"(\w+[\w\s]*?)\s+triggers?\s+(\w+[\w\s]*)",
    ],
    
    "PREVENTS": [
        r"(\w+[\w\s]*?)\s+prevents?\s+(\w+[\w\s]*)",
        r"(\w+[\w\s]*?)\s+averts?\s+(\w+[\w\s]*)",
        r"(\w+[\w\s]*?)\s+stops?\s+(\w+[\w\s]*)",
    ],
    
    "PROTECTS": [
        r"(\w+[\w\s]*?)\s+protects?\s+(?:against\s+)?(\w+[\w\s]*)",
        r"(\w+[\w\s]*?)\s+shields?\s+(?:from\s+)?(\w+[\w\s]*)",
        r"(\w+[\w\s]*?)\s+defends?\s+(?:against\s+)?(\w+[\w\s]*)",
    ],
    
    "LOCATED_IN": [
        r"(\w+[\w\s]*?)\s+(?:is\s+)?(?:located\s+)?in\s+(\w+[\w\s]*)",
        r"(\w+[\w\s]*?)\s+implemented\s+in\s+(\w+[\w\s]*)",
    ],
    
    "AFFECTS": [
        r"(\w+[\w\s]*?)\s+affects?\s+(\w+[\w\s]*)",
        r"(\w+[\w\s]*?)\s+impacts?\s+(\w+[\w\s]*)",
        r"(\w+[\w\s]*?)\s+influences?\s+(\w+[\w\s]*)",
    ],
}


# =============================================================================
# ENTITY DICTIONARY CLASS
# =============================================================================

@dataclass
class EntityMatch:
    """
    Represents a matched entity in text.
    
    Attributes:
        text: Matched text
        entity_type: Type/category of entity
        start: Start position in text
        end: End position in text
        confidence: Match confidence (0-1)
    """
    text: str
    entity_type: str
    start: int
    end: int
    confidence: float = 1.0
    normalized: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "text": self.text,
            "type": self.entity_type,
            "start": self.start,
            "end": self.end,
            "confidence": self.confidence,
            "normalized": self.normalized,
        }


class EntityDictionary:
    """
    Configurable dictionary for entity extraction.
    
    Provides methods for finding known entities in text using
    dictionary lookup with fuzzy matching support.
    
    Example:
        >>> # Use predefined dictionary
        >>> dict = EntityDictionary(CLIMATE_ENTITIES)
        >>> 
        >>> # Find entities
        >>> matches = dict.find_entities(
        ...     "Green roofs can mitigate urban heat island effects"
        ... )
        >>> for m in matches:
        ...     print(f"{m.text} -> {m.entity_type}")
        >>> 
        >>> # Custom dictionary
        >>> custom = EntityDictionary({
        ...     "DRUG": ["aspirin", "ibuprofen"],
        ...     "SYMPTOM": ["headache", "fever"]
        ... })
    """
    
    def __init__(self, entities: Dict[str, List[str]]):
        """
        Initialize with entity dictionary.
        
        Args:
            entities: Dict mapping entity type to list of terms
        """
        self.entities = entities
        self._compiled_patterns = self._compile_patterns()
    
    def _compile_patterns(self) -> Dict[str, List[re.Pattern]]:
        """Compile regex patterns for each entity."""
        patterns = {}
        
        for entity_type, terms in self.entities.items():
            patterns[entity_type] = []
            for term in terms:
                variants = {term}
                # Basic plural support (e.g., roof -> roofs)
                if term and term[-1].lower() != "s":
                    variants.add(f"{term}s")
                for variant in variants:
                    escaped = re.escape(variant)
                    pattern = re.compile(rf"\b{escaped}\b", re.IGNORECASE)
                    patterns[entity_type].append((variant, pattern))
        
        return patterns
    
    def find_entities(
        self,
        text: str,
        entity_types: List[str] = None
    ) -> List[EntityMatch]:
        """
        Find all matching entities in text.
        
        Args:
            text: Text to search
            entity_types: Specific types to search (None = all)
            
        Returns:
            List of EntityMatch objects
        """
        matches = []
        text_lower = text.lower()
        
        types_to_search = entity_types or list(self._compiled_patterns.keys())
        
        for entity_type in types_to_search:
            if entity_type not in self._compiled_patterns:
                continue
            
            for term, pattern in self._compiled_patterns[entity_type]:
                for match in pattern.finditer(text):
                    matches.append(EntityMatch(
                        text=match.group(),
                        entity_type=entity_type,
                        start=match.start(),
                        end=match.end(),
                        confidence=1.0,
                        normalized=term.lower().replace(" ", "_"),
                    ))
        
        # Remove overlapping matches (keep longest)
        matches = self._remove_overlaps(matches)
        
        return sorted(matches, key=lambda m: m.start)
    
    def _remove_overlaps(self, matches: List[EntityMatch]) -> List[EntityMatch]:
        """Remove overlapping matches, keeping longest."""
        if not matches:
            return []
        
        # Sort by start position, then by length (descending)
        sorted_matches = sorted(
            matches,
            key=lambda m: (m.start, -(m.end - m.start))
        )
        
        result = []
        last_end = -1
        
        for match in sorted_matches:
            if match.start >= last_end:
                result.append(match)
                last_end = match.end
        
        return result
    
    def get_entity_types(self) -> List[str]:
        """Get list of entity types in dictionary."""
        return list(self.entities.keys())
    
    def get_terms(self, entity_type: str) -> List[str]:
        """Get terms for a specific entity type."""
        return self.entities.get(entity_type, [])
    
    def add_term(self, entity_type: str, term: str) -> None:
        """Add a term to the dictionary."""
        if entity_type not in self.entities:
            self.entities[entity_type] = []
        
        if term not in self.entities[entity_type]:
            self.entities[entity_type].append(term)
            self._compiled_patterns = self._compile_patterns()
    
    def stats(self) -> Dict[str, int]:
        """Get statistics about the dictionary."""
        return {
            entity_type: len(terms)
            for entity_type, terms in self.entities.items()
        }


__all__ = [
    "CLIMATE_ENTITIES",
    "RELATION_PATTERNS",
    "EntityMatch",
    "EntityDictionary",
]
