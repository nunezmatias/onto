"""
OntoRAG: Ontology-First Retrieval Augmented Generation

OntoRAG is a Python library for building semantically rich RAG systems
using knowledge graphs with explicit ontological structure.

Unlike generic RAG libraries, OntoRAG treats ontology as a first-class
citizen, enabling more precise knowledge representation and retrieval
through explicit semantic relationships.

Key Features:
    - Ontology-aware knowledge graphs
    - Multi-source corpus collection (arXiv, Semantic Scholar, OpenAlex)
    - Rule-based and LLM-powered triplet extraction
    - Hybrid search (semantic + structural)
    - Multiple LLM provider support

Quick Start:
    >>> from ontorag import Ontology, KnowledgeGraph, RAGEngine
    >>> from ontorag import create_climate_ontology
    >>> 
    >>> # 1. Load or create ontology
    >>> onto = create_climate_ontology()
    >>> 
    >>> # 2. Build knowledge graph
    >>> graph = KnowledgeGraph(ontology=onto)
    >>> graph.add_node("green_roof", node_type="green_infrastructure")
    >>> graph.add_node("urban_heat", node_type="heat_wave")
    >>> graph.add_edge("green_roof", "MITIGATES", "urban_heat")
    >>> 
    >>> # 3. Query with RAG
    >>> rag = RAGEngine(graph, llm_provider="openai", api_key="sk-...")
    >>> response = rag.query("How can we cool cities?")
    >>> print(response.answer)

Installation:
    $ pip install ontorag                    # Core only
    $ pip install ontorag[corpus]            # + Corpus collection
    $ pip install ontorag[llm]               # + All LLM providers
    $ pip install ontorag[all]               # Everything

Documentation:
    https://ontorag.readthedocs.io

GitHub:
    https://github.com/ontorag/ontorag
"""

__version__ = "0.2.0"
__author__ = "OntoRAG Team"

# =============================================================================
# CORE IMPORTS
# =============================================================================

from .core import (
    OntoRAGConfig,
    config,
    get_config,
    set_config,
    logger,
)

# =============================================================================
# ONTOLOGY
# =============================================================================

from .ontology import (
    # Classes
    OntologyClass,
    OntologyRelation,
    RelationType,
    ValidationResult,
    Ontology,
    
    # Functions
    load_ontology_from_json,
    create_ontology_from_dict,
    create_climate_ontology,
    create_climate_adaptation_ontology,
)

# =============================================================================
# GRAPH
# =============================================================================

from .graph import (
    Node,
    Edge,
    SearchResult,
    KnowledgeGraph,
    GraphBuilder,
)

# =============================================================================
# SEARCH
# =============================================================================

from .search import (
    SearchSource,
    ItemType,
    HybridSearcher,
    SemanticSearcher,
    StructuralSearcher,
)

# =============================================================================
# RAG
# =============================================================================

from .rag import (
    RAGContext,
    RAGResponse,
    RAGEngine,
)

# =============================================================================
# EXTRACTION
# =============================================================================

from .extraction import (
    # Data classes
    ExtractedEntity,
    ExtractedTriplet,
    ExtractionResult,
    EntityMatch,
    EntityDictionary,
    
    # Extractors
    RuleBasedExtractor,
    TripletExtractor,  # Alias
    
    # Dictionaries
    CLIMATE_ENTITIES,
    
    # Prompts
    EXTRACTION_PROMPT,
    format_extraction_prompt,
)

# LLM extraction (optional)
try:
    from .extraction import (
        LLMConfig,
        LLMExtractor,
        HybridExtractor,
    )
except ImportError:
    LLMConfig = None
    LLMExtractor = None
    HybridExtractor = None

# =============================================================================
# CORPUS (optional)
# =============================================================================

try:
    from .corpus import (
        Document,
        BaseSource,
        SearchQuery,
        Corpus,
        CorpusAnalyzer,
        ArxivSource,
        SemanticScholarSource,
        OpenAlexSource,
    )
    _CORPUS_AVAILABLE = True
except ImportError:
    _CORPUS_AVAILABLE = False
    Document = None
    Corpus = None
    ArxivSource = None
    SemanticScholarSource = None
    OpenAlexSource = None

# =============================================================================
# ANALYTICS
# =============================================================================

from .analytics import (
    GraphMetrics,
    OntologyMetrics,
    GraphAnalytics,
    OntologyAnalytics,
)

# =============================================================================
# VISUALIZATION
# =============================================================================

from .visualization import (
    GraphVisualizer,
    OntologyVisualizer,
)

# =============================================================================
# PUBLIC API
# =============================================================================

__all__ = [
    # Version
    "__version__",
    
    # Config
    "OntoRAGConfig",
    "config",
    "get_config",
    "set_config",
    "logger",
    
    # Ontology
    "Ontology",
    "OntologyClass",
    "OntologyRelation",
    "RelationType",
    "ValidationResult",
    "load_ontology_from_json",
    "create_ontology_from_dict",
    "create_climate_ontology",
    "create_climate_adaptation_ontology",
    
    # Graph
    "Node",
    "Edge",
    "SearchResult",
    "KnowledgeGraph",
    "GraphBuilder",
    
    # Search
    "SearchSource",
    "ItemType",
    "HybridSearcher",
    "SemanticSearcher",
    "StructuralSearcher",
    
    # RAG
    "RAGContext",
    "RAGResponse",
    "RAGEngine",
    
    # Extraction
    "ExtractedEntity",
    "ExtractedTriplet",
    "ExtractionResult",
    "EntityMatch",
    "EntityDictionary",
    "RuleBasedExtractor",
    "TripletExtractor",
    "LLMConfig",
    "LLMExtractor",
    "HybridExtractor",
    "CLIMATE_ENTITIES",
    "EXTRACTION_PROMPT",
    "format_extraction_prompt",
    
    # Corpus
    "Document",
    "BaseSource",
    "SearchQuery",
    "Corpus",
    "CorpusAnalyzer",
    "ArxivSource",
    "SemanticScholarSource",
    "OpenAlexSource",
    
    # Analytics
    "GraphMetrics",
    "OntologyMetrics",
    "GraphAnalytics",
    "OntologyAnalytics",
    
    # Visualization
    "GraphVisualizer",
    "OntologyVisualizer",
]
