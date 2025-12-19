"""
OntoRAG RAG Module

This module provides Retrieval Augmented Generation (RAG) functionality
using knowledge graphs as the retrieval backend.

Key Components:
    - RAGEngine: Main RAG engine with LLM integration
    - RAGContext: Context retrieved from the knowledge graph
    - RAGResponse: Response from the RAG system

Features:
    - Knowledge graph-based retrieval
    - Ontology-aware context building
    - Multiple LLM provider support
    - Configurable prompts

Example:
    >>> from ontorag.rag import RAGEngine
    >>> from ontorag.graph import KnowledgeGraph
    >>> 
    >>> graph = KnowledgeGraph.load("climate_kg.json")
    >>> rag = RAGEngine(graph, llm_provider="openai", api_key="sk-...")
    >>> 
    >>> response = rag.query("What measures can mitigate urban heat?")
    >>> print(response.answer)
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from ..graph import KnowledgeGraph, SearchResult

logger = logging.getLogger(__name__)


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class RAGContext:
    """
    Context retrieved from the knowledge graph for RAG.
    
    Attributes:
        nodes: Retrieved nodes
        edges: Relevant edges
        triplets: Triplets as strings
        text: Formatted context text
        metadata: Additional information
    """
    nodes: List[Dict[str, Any]] = field(default_factory=list)
    edges: List[Dict[str, Any]] = field(default_factory=list)
    triplets: List[str] = field(default_factory=list)
    text: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __len__(self) -> int:
        return len(self.nodes)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "nodes": self.nodes,
            "edges": self.edges,
            "triplets": self.triplets,
            "text": self.text,
            "metadata": self.metadata,
        }


@dataclass
class RAGResponse:
    """
    Response from the RAG system.
    
    Attributes:
        answer: Generated answer
        context: Retrieved context
        query: Original query
        sources: Source references
        confidence: Response confidence
        metadata: Additional information
    """
    answer: str
    context: RAGContext
    query: str = ""
    sources: List[str] = field(default_factory=list)
    confidence: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "answer": self.answer,
            "query": self.query,
            "sources": self.sources,
            "confidence": self.confidence,
            "context": self.context.to_dict(),
            "metadata": self.metadata,
        }


# =============================================================================
# DEFAULT PROMPTS
# =============================================================================

DEFAULT_SYSTEM_PROMPT = """You are a knowledgeable assistant that answers questions based on a knowledge graph about climate adaptation.

Use the provided context from the knowledge graph to answer questions. The context includes:
- Entities and their types (MEASURE, HAZARD, LOCATION, etc.)
- Relationships between entities (MITIGATES, PROTECTS, CAUSES, etc.)
- Relevant triplets in the form (subject, relation, object)

Guidelines:
1. Base your answers on the provided context
2. If the context doesn't contain enough information, say so
3. Be specific and cite the entities and relationships from the context
4. Explain how different measures address different hazards
5. Mention locations and quantitative data when available"""

DEFAULT_QUERY_TEMPLATE = """## KNOWLEDGE GRAPH CONTEXT:

{context}

## QUESTION:
{query}

## ANSWER:
Based on the knowledge graph context provided, """


# =============================================================================
# RAG ENGINE
# =============================================================================

class RAGEngine:
    """
    Retrieval Augmented Generation engine using knowledge graphs.
    
    The RAGEngine combines:
    1. Knowledge graph retrieval for relevant context
    2. Context formatting with ontology awareness
    3. LLM generation for natural language answers
    
    Supported LLM Providers:
    - openai: OpenAI GPT models
    - anthropic: Anthropic Claude models
    - huggingface: HuggingFace models
    - google: Google Gemini models
    - custom: Provide your own callable
    
    Example:
        >>> # Basic usage
        >>> rag = RAGEngine(
        ...     graph=my_graph,
        ...     llm_provider="openai",
        ...     api_key="sk-..."
        ... )
        >>> 
        >>> response = rag.query("How do green roofs help with urban heat?")
        >>> print(response.answer)
        >>> 
        >>> # Custom configuration
        >>> rag = RAGEngine(
        ...     graph=my_graph,
        ...     llm_provider="anthropic",
        ...     model="claude-3-sonnet-20240229",
        ...     api_key="...",
        ...     system_prompt="You are an expert on climate...",
        ...     k=10
        ... )
    """
    
    def __init__(
        self,
        graph: "KnowledgeGraph",
        llm_provider: str = None,
        api_key: str = None,
        model: str = None,
        llm_callable: Callable[[str], str] = None,
        system_prompt: str = None,
        query_template: str = None,
        k: int = 5,
        include_neighbors: bool = True,
        max_context_length: int = 4000,
    ):
        """
        Initialize the RAG engine.
        
        Args:
            graph: Knowledge graph for retrieval
            llm_provider: LLM provider name
            api_key: API key for the LLM provider
            model: Model identifier
            llm_callable: Custom LLM function (overrides provider)
            system_prompt: System prompt for the LLM
            query_template: Template for formatting queries
            k: Number of nodes to retrieve
            include_neighbors: Include neighboring nodes in context
            max_context_length: Maximum context length in characters
        """
        self.graph = graph
        self.k = k
        self.include_neighbors = include_neighbors
        self.max_context_length = max_context_length
        
        self.system_prompt = system_prompt or DEFAULT_SYSTEM_PROMPT
        self.query_template = query_template or DEFAULT_QUERY_TEMPLATE
        
        # Initialize LLM
        self._llm_fn = None
        
        if llm_callable:
            self._llm_fn = llm_callable
        elif llm_provider:
            self._init_llm(llm_provider, api_key, model)
    
    def _init_llm(
        self,
        provider: str,
        api_key: str,
        model: str = None,
    ) -> None:
        """Initialize LLM based on provider."""
        provider = provider.lower()
        
        if provider == "openai":
            self._init_openai(api_key, model)
        elif provider == "anthropic":
            self._init_anthropic(api_key, model)
        elif provider == "huggingface":
            self._init_huggingface(api_key, model)
        elif provider == "google":
            self._init_google(api_key, model)
        else:
            raise ValueError(f"Unknown LLM provider: {provider}")
    
    def _init_openai(self, api_key: str, model: str = None) -> None:
        """Initialize OpenAI."""
        try:
            from openai import OpenAI
            client = OpenAI(api_key=api_key)
            model = model or "gpt-4o-mini"
            
            def call_openai(prompt: str) -> str:
                response = client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": self.system_prompt},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.3,
                    max_tokens=1024,
                )
                return response.choices[0].message.content
            
            self._llm_fn = call_openai
            
        except ImportError:
            raise ImportError("openai package required. Install: pip install openai")
    
    def _init_anthropic(self, api_key: str, model: str = None) -> None:
        """Initialize Anthropic."""
        try:
            import anthropic
            client = anthropic.Anthropic(api_key=api_key)
            model = model or "claude-3-haiku-20240307"
            
            def call_anthropic(prompt: str) -> str:
                response = client.messages.create(
                    model=model,
                    max_tokens=1024,
                    system=self.system_prompt,
                    messages=[{"role": "user", "content": prompt}],
                )
                return response.content[0].text
            
            self._llm_fn = call_anthropic
            
        except ImportError:
            raise ImportError("anthropic package required. Install: pip install anthropic")
    
    def _init_huggingface(self, api_key: str, model: str = None) -> None:
        """Initialize HuggingFace."""
        try:
            from huggingface_hub import InferenceClient
            client = InferenceClient(token=api_key)
            model = model or "meta-llama/Meta-Llama-3-70B-Instruct"
            
            def call_hf(prompt: str) -> str:
                full_prompt = f"{self.system_prompt}\n\n{prompt}"
                response = client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": full_prompt}],
                    max_tokens=1024,
                )
                return response.choices[0].message.content
            
            self._llm_fn = call_hf
            
        except ImportError:
            raise ImportError("huggingface_hub required. Install: pip install huggingface_hub")
    
    def _init_google(self, api_key: str, model: str = None) -> None:
        """Initialize Google Gemini."""
        try:
            import google.generativeai as genai
            genai.configure(api_key=api_key)
            model = model or "gemini-1.5-flash"
            
            client = genai.GenerativeModel(
                model_name=model,
                system_instruction=self.system_prompt,
            )
            
            def call_gemini(prompt: str) -> str:
                response = client.generate_content(prompt)
                return response.text
            
            self._llm_fn = call_gemini
            
        except ImportError:
            raise ImportError("google-generativeai required. Install: pip install google-generativeai")
    
    def query(
        self,
        query: str,
        k: int = None,
        return_context_only: bool = False,
    ) -> RAGResponse:
        """
        Query the RAG system.
        
        Args:
            query: User question
            k: Number of nodes to retrieve (overrides default)
            return_context_only: Only return context, don't generate answer
            
        Returns:
            RAGResponse with answer and context
        """
        k = k or self.k
        
        # Step 1: Retrieve relevant nodes
        search_results = self.graph.search(
            query=query,
            k=k,
            search_type="hybrid",
        )
        
        # Step 2: Build context
        context = self._build_context(search_results)
        
        # Step 3: Return early if only context requested
        if return_context_only or not self._llm_fn:
            return RAGResponse(
                answer="[LLM not configured - returning context only]",
                context=context,
                query=query,
                sources=[r.node_id for r in search_results if r.node],
            )
        
        # Step 4: Generate answer
        prompt = self.query_template.format(
            context=context.text,
            query=query,
        )
        
        try:
            answer = self._llm_fn(prompt)
        except Exception as e:
            logger.error(f"LLM generation failed: {e}")
            answer = f"Error generating response: {e}"
        
        return RAGResponse(
            answer=answer,
            context=context,
            query=query,
            sources=[r.node_id for r in search_results if r.node],
            confidence=sum(r.score for r in search_results) / len(search_results) if search_results else 0,
        )
    
    def _build_context(self, search_results: List["SearchResult"]) -> RAGContext:
        """Build context from search results."""
        context = RAGContext()
        
        text_parts = []
        seen_nodes = set()
        seen_triplets = set()
        
        for result in search_results:
            if not result.node:
                continue
            
            node = result.node
            
            # Add node
            if node.id not in seen_nodes:
                context.nodes.append(node.to_dict())
                seen_nodes.add(node.id)
                
                text_parts.append(
                    f"- {node.label} (type: {node.node_type})"
                )
                if node.properties:
                    for k, v in node.properties.items():
                        text_parts.append(f"  • {k}: {v}")
            
            # Add neighbors if enabled
            if self.include_neighbors:
                neighbors = self.graph.get_neighbors(node.id)
                
                for relation, direction, neighbor in neighbors[:5]:
                    if neighbor.id not in seen_nodes:
                        context.nodes.append(neighbor.to_dict())
                        seen_nodes.add(neighbor.id)
                    
                    # Build triplet
                    if direction == "out":
                        triplet = f"({node.label}, {relation}, {neighbor.label})"
                    else:
                        triplet = f"({neighbor.label}, {relation}, {node.label})"
                    
                    if triplet not in seen_triplets:
                        context.triplets.append(triplet)
                        seen_triplets.add(triplet)
                        text_parts.append(f"  → {triplet}")
        
        # Add edges
        for edge in self.graph.edges:
            if edge.source in seen_nodes and edge.target in seen_nodes:
                context.edges.append(edge.to_dict())
        
        # Build text
        context.text = "\n".join(text_parts[:50])  # Limit length
        
        # Truncate if too long
        if len(context.text) > self.max_context_length:
            context.text = context.text[:self.max_context_length] + "\n..."
        
        return context
    
    def get_related(
        self,
        query: str,
        k: int = 5,
    ) -> List[Dict[str, Any]]:
        """
        Get related entities without generating an answer.
        
        Args:
            query: Search query
            k: Number of results
            
        Returns:
            List of related entity dictionaries
        """
        results = self.graph.search(query=query, k=k)
        return [
            {
                "id": r.node_id,
                "label": r.node.label if r.node else r.node_id,
                "type": r.node.node_type if r.node else "unknown",
                "score": r.score,
            }
            for r in results
        ]


__all__ = [
    "RAGContext",
    "RAGResponse",
    "RAGEngine",
]
