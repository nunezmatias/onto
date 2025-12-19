"""
Extraction Prompts

This module provides configurable prompts for LLM-based extraction.

Features:
- Default extraction prompt optimized for climate domain
- Customizable entity and relation types
- Templates for different domains

Example:
    >>> from ontorag.extraction.prompts import format_extraction_prompt
    >>> 
    >>> prompt = format_extraction_prompt(
    ...     text="Green roofs reduce urban heat...",
    ...     entity_types=["MEASURE", "HAZARD"],
    ...     relation_types=["MITIGATES", "CAUSES"]
    ... )
"""

from typing import List, Optional


# =============================================================================
# DEFAULT CLIMATE EXTRACTION PROMPT
# =============================================================================

EXTRACTION_PROMPT = """You are an expert in knowledge extraction for climate adaptation research.
Your task is to extract ENTITIES and RELATIONS from the following scientific text.

## ENTITY TYPES TO EXTRACT:
- MEASURE: Adaptation measures and interventions (e.g., green roof, early warning system, mangrove restoration)
- HAZARD: Climate hazards and risks (e.g., flood, drought, heat wave, sea level rise)
- LOCATION: Geographic locations (e.g., Rotterdam, Bangladesh, coastal areas)
- IMPACT: Effects and outcomes (e.g., temperature reduction, flood protection)
- QUANTITY: Numerical values with context (e.g., 2-3°C, 70%, €2.3 billion)

## RELATION TYPES TO EXTRACT:
- MITIGATES: A measure reduces the impact of a hazard
- PROTECTS: A measure protects against a hazard
- PREVENTS: A measure prevents a hazard from occurring
- CAUSES: One hazard causes another
- REDUCES_BY: Quantified reduction (e.g., reduces temperature by 2°C)
- LOCATED_IN: Entity is located in a place
- IMPLEMENTED_IN: Measure is implemented in a location
- AFFECTS: Hazard affects a sector or location

## TEXT TO ANALYZE:
{text}

## INSTRUCTIONS:
1. Extract all relevant entities from the text
2. Identify relations between entities
3. Normalize entity names (lowercase, underscores)
4. Include only relations that are explicitly stated or strongly implied
5. Respond ONLY with valid JSON, no explanations

## RESPONSE FORMAT:
{{
  "entities": [
    {{"text": "entity name", "type": "ENTITY_TYPE"}}
  ],
  "triplets": [
    {{"subject": "entity1", "relation": "RELATION_TYPE", "object": "entity2"}}
  ]
}}

JSON Response:"""


# =============================================================================
# MINIMAL PROMPT
# =============================================================================

MINIMAL_PROMPT = """Extract entities and relations from this text as JSON.

Text: {text}

Response format:
{{"entities": [{{"text": "...", "type": "..."}}], "triplets": [{{"subject": "...", "relation": "...", "object": "..."}}]}}

JSON:"""


# =============================================================================
# BIOMEDICAL PROMPT
# =============================================================================

BIOMEDICAL_PROMPT = """You are a biomedical knowledge extraction expert.
Extract entities and relations from the following text.

## ENTITY TYPES:
- DISEASE: Medical conditions and diseases
- DRUG: Medications and treatments
- GENE: Genes and proteins
- SYMPTOM: Signs and symptoms
- PROCEDURE: Medical procedures
- ANATOMICAL: Body parts and organs

## RELATION TYPES:
- TREATS: Drug treats disease
- CAUSES: Entity causes condition
- ASSOCIATED_WITH: Entities are associated
- INTERACTS_WITH: Drug/gene interactions
- LOCATED_IN: Anatomical location

## TEXT:
{text}

## RESPONSE (JSON only):
{{
  "entities": [{{"text": "...", "type": "..."}}],
  "triplets": [{{"subject": "...", "relation": "...", "object": "..."}}]
}}

JSON:"""


# =============================================================================
# PROMPT FORMATTING FUNCTIONS
# =============================================================================

def format_extraction_prompt(
    text: str,
    entity_types: List[str] = None,
    relation_types: List[str] = None,
    template: str = None,
) -> str:
    """
    Format an extraction prompt with the given text.
    
    Args:
        text: Text to extract from
        entity_types: Custom entity types (uses default if None)
        relation_types: Custom relation types (uses default if None)
        template: Custom prompt template (uses EXTRACTION_PROMPT if None)
    
    Returns:
        Formatted prompt string
    
    Example:
        >>> prompt = format_extraction_prompt(
        ...     "Green roofs reduce heat waves by 2-3°C",
        ...     entity_types=["MEASURE", "HAZARD", "QUANTITY"]
        ... )
    """
    if template:
        return template.format(text=text)
    
    if entity_types or relation_types:
        # Build custom prompt
        prompt_parts = [
            "Extract entities and relations from this text as JSON.\n",
        ]
        
        if entity_types:
            prompt_parts.append(f"\nEntity types: {', '.join(entity_types)}")
        
        if relation_types:
            prompt_parts.append(f"\nRelation types: {', '.join(relation_types)}")
        
        prompt_parts.append(f"\n\nText: {text}")
        prompt_parts.append('\n\nResponse format: {"entities": [...], "triplets": [...]}')
        prompt_parts.append("\n\nJSON:")
        
        return "".join(prompt_parts)
    
    return EXTRACTION_PROMPT.format(text=text)


def get_domain_prompt(domain: str) -> str:
    """
    Get a prompt template for a specific domain.
    
    Args:
        domain: Domain name (climate, biomedical, minimal)
    
    Returns:
        Prompt template string
    
    Example:
        >>> template = get_domain_prompt("biomedical")
        >>> prompt = template.format(text="...")
    """
    prompts = {
        "climate": EXTRACTION_PROMPT,
        "biomedical": BIOMEDICAL_PROMPT,
        "minimal": MINIMAL_PROMPT,
    }
    
    if domain not in prompts:
        raise ValueError(f"Unknown domain: {domain}. Available: {list(prompts.keys())}")
    
    return prompts[domain]


def create_custom_prompt(
    entity_types: List[str],
    relation_types: List[str],
    instructions: str = "",
    examples: List[dict] = None,
) -> str:
    """
    Create a custom extraction prompt.
    
    Args:
        entity_types: List of entity types with descriptions
        relation_types: List of relation types with descriptions
        instructions: Additional instructions
        examples: Optional few-shot examples
    
    Returns:
        Custom prompt template (use .format(text=...) to use)
    
    Example:
        >>> prompt = create_custom_prompt(
        ...     entity_types=["PERSON: A human being", "ORG: An organization"],
        ...     relation_types=["WORKS_FOR: Person works for org"],
        ...     instructions="Focus on employment relationships."
        ... )
    """
    parts = [
        "Extract entities and relations from the following text.\n",
        "\n## ENTITY TYPES:",
    ]
    
    for et in entity_types:
        parts.append(f"\n- {et}")
    
    parts.append("\n\n## RELATION TYPES:")
    
    for rt in relation_types:
        parts.append(f"\n- {rt}")
    
    if instructions:
        parts.append(f"\n\n## INSTRUCTIONS:\n{instructions}")
    
    if examples:
        parts.append("\n\n## EXAMPLES:")
        for ex in examples:
            parts.append(f"\nText: {ex.get('text', '')}")
            parts.append(f"\nOutput: {ex.get('output', '')}")
    
    parts.append("\n\n## TEXT TO ANALYZE:\n{text}")
    parts.append("\n\n## RESPONSE (valid JSON only):")
    parts.append('\n{{"entities": [...], "triplets": [...]}}\n')
    parts.append("\nJSON:")
    
    return "".join(parts)


__all__ = [
    "EXTRACTION_PROMPT",
    "MINIMAL_PROMPT",
    "BIOMEDICAL_PROMPT",
    "format_extraction_prompt",
    "get_domain_prompt",
    "create_custom_prompt",
]
