# OntoRAG

**Ontology-First Retrieval Augmented Generation**

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

OntoRAG is a Python library for building semantically rich RAG (Retrieval Augmented Generation) systems using knowledge graphs with explicit ontological structure.

Unlike generic RAG libraries that treat relationships as simple connections, OntoRAG treats **ontology as a first-class citizen**, enabling more precise knowledge representation and retrieval through:

- 🔷 **Explicit semantic relationships** (MITIGATES, CAUSES, PROTECTS vs generic "relates to")
- 🏛️ **Hierarchical class structures** with IS-A inheritance
- ✅ **Domain-specific validation** ensuring data quality
- 🔍 **Hybrid search** combining semantic similarity with structural graph exploration

## Features

### Core Capabilities

| Feature | Description |
|---------|-------------|
| **Ontology Management** | Define classes, relations, synonyms, and constraints. Load from JSON or create programmatically. |
| **Knowledge Graphs** | Build graphs with NetworkX + ChromaDB for hybrid semantic/structural search. |
| **Multi-Source Corpus** | Collect documents from arXiv, Semantic Scholar, and OpenAlex. |
| **Triplet Extraction** | Rule-based and LLM-powered extraction of entities and relations. |
| **RAG Engine** | Query your knowledge graph with natural language using any LLM provider. |
| **Analytics** | Graph metrics, ontology coverage analysis, and visualization. |

### Supported LLM Providers

- OpenAI (GPT-4, GPT-4o, etc.)
- Anthropic (Claude 3)
- Google (Gemini)
- HuggingFace (Llama, Mistral, etc.)

## Installation

```bash
# Core only (ontology + graph + basic extraction)
pip install ontorag

# With corpus collection (arXiv, Semantic Scholar, OpenAlex)
pip install ontorag[corpus]

# With all LLM providers
pip install ontorag[llm]

# With visualization
pip install ontorag[viz]

# Everything
pip install ontorag[all]
```

## Quick Start

### 1. Create an Ontology

```python
from ontorag import create_climate_ontology, Ontology, OntologyClass, OntologyRelation

# Use predefined ontology
onto = create_climate_ontology()

# Or create your own
onto = Ontology("my_domain", "My Domain Ontology")
onto.add_class(OntologyClass(id="entity", name="Entity"))
onto.add_class(OntologyClass(
    id="measure", 
    name="Measure",
    parent="entity",
    synonyms=["intervention", "action"]
))
onto.add_relation(OntologyRelation(
    id="MITIGATES",
    name="mitigates",
    domain=["measure"],
    range=["hazard"]
))

# Or load from JSON
onto = Ontology.load("my_ontology.json")
```

### 2. Build a Knowledge Graph

```python
from ontorag import KnowledgeGraph

# Create graph with ontology validation
graph = KnowledgeGraph(ontology=onto)

# Add entities
graph.add_node("green_roof", node_type="green_infrastructure",
               label="Green Roof", properties={"cooling": "2-3°C"})
graph.add_node("urban_heat", node_type="heat_wave")

# Add relationships
graph.add_edge("green_roof", "MITIGATES", "urban_heat")

# Search
results = graph.search("cooling solutions for cities", k=5)
for r in results:
    print(f"{r.node.label}: {r.score:.2f}")
```

### 3. Extract Knowledge from Text

```python
from ontorag import RuleBasedExtractor, LLMExtractor

# Rule-based (no API needed)
extractor = RuleBasedExtractor()
result = extractor.extract("Green roofs reduce urban heat by 2-3°C")

print(result.entities)  # Found entities
print(result.triplets)  # (green_roof, MITIGATES, urban_heat)

# LLM-based (more comprehensive)
llm_extractor = LLMExtractor(provider="openai", api_key="sk-...")
result = llm_extractor.extract(text)
```

### 4. Collect Academic Papers

```python
from ontorag.corpus import Corpus

corpus = Corpus(name="climate_research")

# Search multiple sources
corpus.search_all(
    "climate adaptation flood mitigation",
    sources=["arxiv", "semantic_scholar", "openalex"],
    max_per_source=100
)

print(corpus.summary())
corpus.save("corpus.json")
```

### 5. Query with RAG

```python
from ontorag import RAGEngine

rag = RAGEngine(
    graph=graph,
    llm_provider="openai",
    api_key="sk-..."
)

response = rag.query("What measures can reduce urban flooding?")
print(response.answer)
print(response.sources)  # Referenced entities
```

## JSON Ontology Format

Define ontologies in JSON for easy sharing and version control:

```json
{
  "name": "climate_adaptation",
  "description": "Climate adaptation domain ontology",
  "version": "1.0.0",
  "classes": {
    "climate_hazard": {
      "name": "Climate Hazard",
      "description": "Natural hazards worsened by climate change",
      "synonyms": ["hazard", "risk", "threat"]
    },
    "flood": {
      "name": "Flood",
      "parent": "climate_hazard",
      "synonyms": ["flooding", "inundation"]
    }
  },
  "relations": {
    "MITIGATES": {
      "name": "mitigates",
      "description": "Subject reduces impact of object",
      "relation_type": "mitigative",
      "domain": ["adaptation_measure"],
      "range": ["climate_hazard"]
    }
  }
}
```

Load it:

```python
from ontorag import load_ontology_from_json
onto = load_ontology_from_json("my_ontology.json")
```

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        OntoRAG                               │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐              │
│  │ Ontology │───▶│  Graph   │◀───│ Corpus   │              │
│  │ Module   │    │ Module   │    │ Module   │              │
│  └──────────┘    └──────────┘    └──────────┘              │
│       │              │  ▲            │                      │
│       │              │  │            │                      │
│       ▼              ▼  │            ▼                      │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐              │
│  │ Validation│   │  Search  │    │Extraction│              │
│  │   Rules   │   │  Engine  │    │  Module  │              │
│  └──────────┘    └──────────┘    └──────────┘              │
│                       │                                      │
│                       ▼                                      │
│               ┌──────────────┐                              │
│               │  RAG Engine  │                              │
│               │  (+ LLM)     │                              │
│               └──────────────┘                              │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

## Examples

See the `examples/` directory for complete workflows:

1. **01_quick_start.py** - Basic ontology and graph creation
2. **02_ontology_from_json.py** - Loading and validating ontologies
3. **03_corpus_collection.py** - Collecting papers from academic sources
4. **04_triplet_extraction.py** - Extracting knowledge from text
5. **05_build_knowledge_graph.py** - Building graphs from extracted data
6. **06_rag_queries.py** - Querying with RAG

## Comparison with Other Libraries

| Feature | OntoRAG | LlamaIndex | LangChain |
|---------|---------|------------|-----------|
| Explicit ontology | ✅ First-class | ❌ No | ❌ No |
| Typed relations | ✅ Semantic types | ❌ Generic | ❌ Generic |
| Class hierarchies | ✅ IS-A inheritance | ❌ No | ❌ No |
| Domain validation | ✅ Built-in | ❌ No | ❌ No |
| Academic sources | ✅ arXiv, S2, OA | ❌ Limited | ❌ Limited |
| Rule-based extraction | ✅ Yes | ❌ No | ❌ No |
| Hybrid graph search | ✅ Semantic+Structural | ⚠️ Partial | ⚠️ Partial |

## Documentation

- [User Guide](docs/user_guide.md)
- [API Reference](docs/api_reference.md)
- [Examples](examples/)

## Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## License

MIT License - see [LICENSE](LICENSE) for details.

## Citation

If you use OntoRAG in your research, please cite:

```bibtex
@software{ontorag2024,
  title = {OntoRAG: Ontology-First Retrieval Augmented Generation},
  year = {2024},
  url = {https://github.com/ontorag/ontorag}
}
```
