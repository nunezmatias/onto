"""
LLM-Based Triplet Extraction

This module provides LLM-powered extraction of entities and relations.
Supports multiple LLM providers: OpenAI, Anthropic, HuggingFace, Google.

Features:
- Semantic understanding of text
- Discovery of novel entities and relations
- Configurable prompts
- Multi-provider support

Example:
    >>> from ontorag.extraction import LLMExtractor
    >>> 
    >>> # Using OpenAI
    >>> extractor = LLMExtractor(provider="openai", api_key="sk-...")
    >>> result = extractor.extract(text)
    >>> 
    >>> # Using Anthropic Claude
    >>> extractor = LLMExtractor(provider="anthropic", api_key="...")
"""

from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Union

from .rule_based import ExtractedEntity, ExtractedTriplet, ExtractionResult
from .prompts import EXTRACTION_PROMPT, format_extraction_prompt

logger = logging.getLogger(__name__)


@dataclass
class LLMConfig:
    """
    Configuration for LLM extraction.
    
    Attributes:
        provider: LLM provider name
        model: Model identifier
        temperature: Sampling temperature (0-1)
        max_tokens: Maximum response tokens
        api_key: API key for the provider
        timeout: Request timeout in seconds
        retry_count: Number of retries on failure
        delay_between_calls: Delay in seconds between API calls
    """
    provider: str = "openai"
    model: str = ""
    temperature: float = 0.2
    max_tokens: int = 2048
    api_key: Optional[str] = None
    timeout: int = 60
    retry_count: int = 3
    delay_between_calls: float = 1.0
    
    def __post_init__(self):
        # Set default models per provider
        if not self.model:
            defaults = {
                "openai": "gpt-4o-mini",
                "anthropic": "claude-3-haiku-20240307",
                "huggingface": "meta-llama/Meta-Llama-3-70B-Instruct",
                "google": "gemini-1.5-flash",
            }
            self.model = defaults.get(self.provider, "gpt-4o-mini")


class LLMExtractor:
    """
    Extract entities and relations using Large Language Models.
    
    This extractor uses LLMs to:
    1. Identify entities mentioned in text
    2. Discover relations between entities
    3. Normalize entities to canonical forms
    
    Supported Providers:
    - openai: OpenAI GPT models
    - anthropic: Anthropic Claude models
    - huggingface: HuggingFace Inference API
    - google: Google Gemini models
    
    Example:
        >>> # Basic usage
        >>> extractor = LLMExtractor(
        ...     provider="openai",
        ...     api_key="sk-..."
        ... )
        >>> result = extractor.extract("Green roofs reduce heat by 2-3°C")
        >>> 
        >>> # Custom configuration
        >>> config = LLMConfig(
        ...     provider="anthropic",
        ...     model="claude-3-sonnet-20240229",
        ...     temperature=0.1
        ... )
        >>> extractor = LLMExtractor(config=config, api_key="...")
        >>> 
        >>> # With custom prompt
        >>> extractor = LLMExtractor(
        ...     provider="openai",
        ...     api_key="...",
        ...     custom_prompt="Extract medical entities..."
        ... )
    """
    
    def __init__(
        self,
        provider: str = None,
        api_key: str = None,
        model: str = None,
        config: LLMConfig = None,
        custom_prompt: str = None,
        entity_types: List[str] = None,
        relation_types: List[str] = None,
    ):
        """
        Initialize the LLM extractor.
        
        Args:
            provider: LLM provider (openai, anthropic, huggingface, google)
            api_key: API key for the provider
            model: Model identifier
            config: Full LLMConfig object (overrides other params)
            custom_prompt: Custom extraction prompt template
            entity_types: List of entity types to extract
            relation_types: List of relation types to extract
        """
        if config:
            self.config = config
        else:
            self.config = LLMConfig(
                provider=provider or "openai",
                model=model or "",
                api_key=api_key,
            )
        
        if api_key:
            self.config.api_key = api_key
        
        self.custom_prompt = custom_prompt
        self.entity_types = entity_types
        self.relation_types = relation_types
        
        # Initialize client
        self._client = None
        self._init_client()
        
        # Stats
        self._call_count = 0
        self._total_tokens = 0
    
    def _init_client(self) -> None:
        """Initialize the LLM client based on provider."""
        provider = self.config.provider.lower()
        
        if provider == "openai":
            self._init_openai()
        elif provider == "anthropic":
            self._init_anthropic()
        elif provider == "huggingface":
            self._init_huggingface()
        elif provider == "google":
            self._init_google()
        else:
            raise ValueError(f"Unknown provider: {provider}")
    
    def _init_openai(self) -> None:
        """Initialize OpenAI client."""
        try:
            from openai import OpenAI
            self._client = OpenAI(api_key=self.config.api_key)
            self._call_fn = self._call_openai
        except ImportError:
            raise ImportError("openai package required. Install: pip install openai")
    
    def _init_anthropic(self) -> None:
        """Initialize Anthropic client."""
        try:
            import anthropic
            self._client = anthropic.Anthropic(api_key=self.config.api_key)
            self._call_fn = self._call_anthropic
        except ImportError:
            raise ImportError("anthropic package required. Install: pip install anthropic")
    
    def _init_huggingface(self) -> None:
        """Initialize HuggingFace client."""
        try:
            from huggingface_hub import InferenceClient
            self._client = InferenceClient(token=self.config.api_key)
            self._call_fn = self._call_huggingface
        except ImportError:
            raise ImportError("huggingface_hub required. Install: pip install huggingface_hub")
    
    def _init_google(self) -> None:
        """Initialize Google Gemini client."""
        try:
            import google.generativeai as genai
            genai.configure(api_key=self.config.api_key)
            self._client = genai.GenerativeModel(
                model_name=self.config.model,
                generation_config=genai.GenerationConfig(
                    temperature=self.config.temperature,
                    max_output_tokens=self.config.max_tokens,
                )
            )
            self._call_fn = self._call_google
        except ImportError:
            raise ImportError("google-generativeai required. Install: pip install google-generativeai")
    
    def _call_openai(self, prompt: str) -> str:
        """Call OpenAI API."""
        response = self._client.chat.completions.create(
            model=self.config.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
        )
        return response.choices[0].message.content
    
    def _call_anthropic(self, prompt: str) -> str:
        """Call Anthropic API."""
        response = self._client.messages.create(
            model=self.config.model,
            max_tokens=self.config.max_tokens,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.content[0].text
    
    def _call_huggingface(self, prompt: str) -> str:
        """Call HuggingFace Inference API."""
        response = self._client.chat.completions.create(
            model=self.config.model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=self.config.max_tokens,
            temperature=self.config.temperature,
        )
        return response.choices[0].message.content
    
    def _call_google(self, prompt: str) -> str:
        """Call Google Gemini API."""
        response = self._client.generate_content(prompt)
        return response.text
    
    def extract(self, text: str) -> ExtractionResult:
        """
        Extract entities and triplets from text using LLM.
        
        Args:
            text: Input text to process
            
        Returns:
            ExtractionResult with entities and triplets
        """
        result = ExtractionResult(source_text=text)
        
        if not text or not text.strip():
            return result
        
        # Build prompt
        prompt = self._build_prompt(text)
        
        # Call LLM with retry
        response = None
        for attempt in range(self.config.retry_count):
            try:
                response = self._call_fn(prompt)
                self._call_count += 1
                break
            except Exception as e:
                logger.warning(f"LLM call failed (attempt {attempt + 1}): {e}")
                if attempt < self.config.retry_count - 1:
                    time.sleep(self.config.delay_between_calls * (attempt + 1))
        
        if not response:
            logger.error("All LLM call attempts failed")
            return result
        
        # Parse response
        parsed = self._parse_response(response)
        
        # Convert to result objects
        for entity_data in parsed.get("entities", []):
            result.entities.append(ExtractedEntity(
                text=entity_data.get("text", ""),
                type=entity_data.get("type", "UNKNOWN"),
                normalized=entity_data.get("text", "").lower().replace(" ", "_"),
                confidence=0.8,
                source="llm",
            ))
        
        for triplet_data in parsed.get("triplets", []):
            result.triplets.append(ExtractedTriplet(
                subject=triplet_data.get("subject", ""),
                relation=triplet_data.get("relation", "RELATED_TO"),
                object=triplet_data.get("object", ""),
                confidence=0.8,
                source="llm",
            ))
        
        return result
    
    def extract_batch(
        self,
        texts: List[str],
        show_progress: bool = True
    ) -> List[ExtractionResult]:
        """
        Extract from multiple texts.
        
        Args:
            texts: List of texts to process
            show_progress: Whether to show progress
            
        Returns:
            List of ExtractionResult objects
        """
        results = []
        
        for i, text in enumerate(texts):
            if show_progress:
                logger.info(f"Processing {i + 1}/{len(texts)}...")
            
            result = self.extract(text)
            results.append(result)
            
            # Delay between calls
            if i < len(texts) - 1:
                time.sleep(self.config.delay_between_calls)
        
        return results
    
    def _build_prompt(self, text: str) -> str:
        """Build the extraction prompt."""
        if self.custom_prompt:
            return self.custom_prompt.format(text=text)
        
        return format_extraction_prompt(
            text=text,
            entity_types=self.entity_types,
            relation_types=self.relation_types,
        )
    
    def _parse_response(self, response: str) -> Dict[str, Any]:
        """Parse LLM response into structured data."""
        # Try to extract JSON from response
        try:
            # Look for JSON block
            json_match = re.search(r'\{[\s\S]*\}', response)
            if json_match:
                return json.loads(json_match.group())
        except json.JSONDecodeError:
            pass
        
        # Try to parse as raw JSON
        try:
            # Remove markdown code blocks if present
            cleaned = response.strip()
            if cleaned.startswith("```"):
                cleaned = re.sub(r'^```\w*\n?', '', cleaned)
                cleaned = re.sub(r'\n?```$', '', cleaned)
            return json.loads(cleaned)
        except json.JSONDecodeError:
            pass
        
        logger.warning("Could not parse LLM response as JSON")
        return {"entities": [], "triplets": []}
    
    @property
    def stats(self) -> Dict[str, Any]:
        """Get extraction statistics."""
        return {
            "provider": self.config.provider,
            "model": self.config.model,
            "call_count": self._call_count,
        }


class HybridExtractor:
    """
    Combine rule-based and LLM extraction.
    
    Uses rules first for known entities, then LLM for discovery.
    
    Example:
        >>> extractor = HybridExtractor(
        ...     llm_provider="openai",
        ...     llm_api_key="sk-..."
        ... )
        >>> result = extractor.extract(text)
    """
    
    def __init__(
        self,
        llm_provider: str = None,
        llm_api_key: str = None,
        llm_config: LLMConfig = None,
        use_llm: bool = True,
        entity_dict: Dict[str, List[str]] = None,
    ):
        """
        Initialize hybrid extractor.
        
        Args:
            llm_provider: LLM provider name
            llm_api_key: API key for LLM
            llm_config: Full LLM configuration
            use_llm: Whether to use LLM (False = rules only)
            entity_dict: Custom entity dictionary
        """
        from .rule_based import RuleBasedExtractor
        
        self.rule_extractor = RuleBasedExtractor(entity_dict=entity_dict)
        
        self.llm_extractor = None
        if use_llm and (llm_provider or llm_config):
            try:
                self.llm_extractor = LLMExtractor(
                    provider=llm_provider,
                    api_key=llm_api_key,
                    config=llm_config,
                )
            except Exception as e:
                logger.warning(f"Could not initialize LLM extractor: {e}")
    
    def extract(self, text: str) -> ExtractionResult:
        """
        Extract using both methods and merge results.
        
        Args:
            text: Input text
            
        Returns:
            Merged ExtractionResult
        """
        # Rule-based extraction (always)
        rule_result = self.rule_extractor.extract(text)
        
        # LLM extraction (if available)
        if self.llm_extractor:
            try:
                llm_result = self.llm_extractor.extract(text)
                return self._merge_results(rule_result, llm_result)
            except Exception as e:
                logger.warning(f"LLM extraction failed: {e}")
        
        return rule_result
    
    def _merge_results(
        self,
        rule_result: ExtractionResult,
        llm_result: ExtractionResult
    ) -> ExtractionResult:
        """Merge rule and LLM results."""
        merged = ExtractionResult(source_text=rule_result.source_text)
        
        # Merge entities (prefer rule-based for known entities)
        entity_texts = set()
        
        for entity in rule_result.entities:
            merged.entities.append(entity)
            entity_texts.add(entity.text.lower())
        
        for entity in llm_result.entities:
            if entity.text.lower() not in entity_texts:
                merged.entities.append(entity)
                entity_texts.add(entity.text.lower())
        
        # Merge triplets
        triplet_keys = set()
        
        for triplet in rule_result.triplets:
            key = (triplet.subject, triplet.relation, triplet.object)
            merged.triplets.append(triplet)
            triplet_keys.add(key)
        
        for triplet in llm_result.triplets:
            key = (triplet.subject, triplet.relation, triplet.object)
            if key not in triplet_keys:
                merged.triplets.append(triplet)
                triplet_keys.add(key)
        
        merged.metadata["sources"] = ["rule_based", "llm"]
        
        return merged


__all__ = [
    "LLMConfig",
    "LLMExtractor",
    "HybridExtractor",
]
