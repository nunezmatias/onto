# OntoRAG User Guide

This guide covers the main features and workflows of OntoRAG.

## Table of Contents

1. [Installation](#installation)
2. [Core Concepts](#core-concepts)
3. [Ontology Module](#ontology-module)
4. [Corpus Module](#corpus-module)
5. [Extraction Module](#extraction-module)
6. [Graph Module](#graph-module)
7. [RAG Module](#rag-module)
8. [Best Practices](#best-practices)

---

## Installation

### Basic Installation

```bash
pip install ontorag
```

### With Optional Features

```bash
# Corpus collection (arXiv, Semantic Scholar, OpenAlex)
pip install ontorag[corpus]

# LLM providers (OpenAI, Anthropic, HuggingFace, Google)
pip install ontorag[llm]

# Visualization (matplotlib, wordcloud)
pip install ontorag[viz]

# Everything
pip install ontorag[all]
```

### From Source

```bash
git clone https://github.com/ontorag/ontorag.git
cd ontorag
pip install -e ".[all]"
```

---

## Core Concepts

### What is OntoRAG?

OntoRAG is a library for building **ontology-aware knowledge graphs** and using them for **Retrieval Augmented Generation (RAG)**.

### Why Ontology-First?

Traditional RAG systems treat all relationships equally. OntoRAG recognizes that relationships have different meanings:

- `MITIGATES` implies reduction of a hazard
- `CAUSES` implies a causal chain
- `PART_OF` implies composition and inheritance

This semantic richness enables more precise retrieval and reasoning.

### Key Components

1. **Ontology**: Defines the vocabulary and rules for your domain
2. **Corpus**: Collection of documents from various sources
3. **Extraction**: Extract entities and relations from text
4. **Graph**: Store and query knowledge as a graph
5. **RAG**: Answer questions using the knowledge graph

---

## Ontology Module

### Creating an Ontology Programmatically

```python
from ontorag import Ontology, OntologyClass, OntologyRelation

# Create empty ontology
onto = Ontology("healthcare", "Healthcare Domain Ontology")

# Add classes
onto.add_class(OntologyClass(
    id="entity",
    name="Entity",
    description="Root class for all entities"
))

onto.add_class(OntologyClass(
    id="drug",
    name="Drug",
    description="Pharmaceutical compound",
    parent="entity",
    synonyms=["medication", "medicine", "pharmaceutical"]
))

onto.add_class(OntologyClass(
    id="disease",
    name="Disease",
    parent="entity",
    synonyms=["condition", "disorder"]
))

# Add relations
onto.add_relation(OntologyRelation(
    id="TREATS",
    name="treats",
    description="Drug treats disease",
    relation_type="functional",
    domain=["drug"],
    range=["disease"],
    synonyms=["cures", "alleviates"]
))
```

### Loading from JSON

```python
from ontorag import load_ontology_from_json

onto = load_ontology_from_json("my_ontology.json")
```

### JSON Schema

```json
{
  "name": "domain_name",
  "description": "Description of the domain",
  "version": "1.0.0",
  "classes": {
    "class_id": {
      "name": "Human Readable Name",
      "description": "Description",
      "parent": "parent_class_id",
      "synonyms": ["alt1", "alt2"]
    }
  },
  "relations": {
    "RELATION_ID": {
      "name": "relation name",
      "description": "What this relation means",
      "relation_type": "causal|mitigative|spatial|...",
      "domain": ["class_id"],
      "range": ["class_id"],
      "synonyms": ["alt1"]
    }
  }
}
```

### Validation

```python
result = onto.validate()

if not result.is_valid:
    for error in result.errors:
        print(f"Error: {error}")
        
for warning in result.warnings:
    print(f"Warning: {warning}")
```

### Normalization

```python
# Normalize entity mentions to canonical form
onto.normalize_entity("flooding")  # Returns "flood"
onto.normalize_entity("heatwave")  # Returns "heat_wave"

# Normalize relations
onto.normalize_relation("reduces")  # Returns "MITIGATES"
```

---

## Corpus Module

### Searching Academic Sources

```python
from ontorag.corpus import Corpus, ArxivSource, SemanticScholarSource

# Create corpus
corpus = Corpus(name="my_research")

# Search specific source
arxiv = ArxivSource()
docs = arxiv.search(
    'abs:"climate change" AND abs:"adaptation"',
    max_results=50
)

# Or search through corpus manager
corpus.search(
    "flood mitigation green infrastructure",
    sources=["arxiv", "semantic_scholar"],
    max_results=100
)
```

### Multi-Source Search

```python
corpus.search_all(
    "urban heat island cooling",
    sources=["arxiv", "semantic_scholar", "openalex"],
    max_per_source=50
)
```

### Filtering

```python
# Filter by year
recent = corpus.filter(year_min=2020, year_max=2024)

# Filter by source
arxiv_only = corpus.filter(sources=["arxiv"])

# Filter by content
with_abstracts = corpus.filter(has_abstract=True)
cited = corpus.filter(min_citations=10)
```

### Analysis

```python
from ontorag.corpus import CorpusAnalyzer

analyzer = CorpusAnalyzer(corpus)

# Term frequency
top_terms = analyzer.term_frequency(n=20)

# Year distribution
years = analyzer.year_distribution()

# Generate word cloud
analyzer.generate_wordcloud("cloud.png")
```

### Export

```python
corpus.save("corpus.json")
corpus.to_csv("corpus.csv")

# Load back
loaded = Corpus.load("corpus.json")
```

---

## Extraction Module

### Rule-Based Extraction

```python
from ontorag import RuleBasedExtractor

extractor = RuleBasedExtractor()

result = extractor.extract(
    "Green roofs can reduce urban temperatures by 2-3°C, "
    "providing effective mitigation against heat waves."
)

# Entities found
for entity in result.entities:
    print(f"{entity.text} -> {entity.type}")

# Triplets extracted
for triplet in result.triplets:
    print(f"({triplet.subject}, {triplet.relation}, {triplet.object})")
```

### Custom Dictionary

```python
my_entities = {
    "SYMPTOM": ["fever", "cough", "fatigue"],
    "TREATMENT": ["rest", "medication", "surgery"]
}

extractor = RuleBasedExtractor(entity_dict=my_entities)
```

### LLM-Based Extraction

```python
from ontorag import LLMExtractor

extractor = LLMExtractor(
    provider="openai",
    api_key="sk-...",
    model="gpt-4o-mini"
)

result = extractor.extract(text)
```

### Hybrid Extraction

```python
from ontorag import HybridExtractor

extractor = HybridExtractor(
    llm_provider="openai",
    llm_api_key="sk-...",
    entity_dict=my_custom_dict
)

# Uses rules first, then LLM for discovery
result = extractor.extract(text)
```

---

## Graph Module

### Creating a Graph

```python
from ontorag import KnowledgeGraph

# With ontology validation
graph = KnowledgeGraph(ontology=onto)

# Or without (no validation)
graph = KnowledgeGraph()
```

### Adding Nodes

```python
graph.add_node(
    "aspirin",
    node_type="drug",
    label="Aspirin",
    properties={"dosage": "500mg"}
)
```

### Adding Edges

```python
graph.add_edge(
    source="aspirin",
    relation="TREATS",
    target="headache",
    properties={"effectiveness": "high"}
)
```

### Bulk Import

```python
# From extraction results
from ontorag import GraphBuilder

builder = GraphBuilder(ontology=onto)

for text in documents:
    result = extractor.extract(text)
    builder.add_extraction_result(result)

graph = builder.build()
```

### Search

```python
# Semantic search
results = graph.search("pain relief", k=5, search_type="semantic")

# Structural search
results = graph.search("aspirin", k=5, search_type="structural")

# Hybrid (recommended)
results = graph.search("headache treatment", k=5, search_type="hybrid")
```

### Navigation

```python
# Get neighbors
neighbors = graph.get_neighbors("aspirin")
for relation, direction, node in neighbors:
    print(f"{relation} -> {node.label}")

# Find path
path = graph.get_path("aspirin", "headache")

# Get subgraph
subgraph = graph.get_subgraph("aspirin", depth=2)
```

### Persistence

```python
graph.save("knowledge_graph.json")
loaded = KnowledgeGraph.load("knowledge_graph.json", ontology=onto)
```

---

## RAG Module

### Basic Setup

```python
from ontorag import RAGEngine

rag = RAGEngine(
    graph=graph,
    llm_provider="openai",
    api_key="sk-..."
)
```

### Querying

```python
response = rag.query("What treats headaches?")

print(response.answer)
print(response.sources)  # Entities used
print(response.context)  # Retrieved context
```

### Custom Prompts

```python
rag = RAGEngine(
    graph=graph,
    llm_provider="openai",
    api_key="sk-...",
    system_prompt="You are a medical expert...",
    query_template="Context: {context}\n\nQuestion: {query}"
)
```

### Different Providers

```python
# OpenAI
rag = RAGEngine(graph, llm_provider="openai", api_key="sk-...")

# Anthropic
rag = RAGEngine(graph, llm_provider="anthropic", api_key="...")

# Google Gemini
rag = RAGEngine(graph, llm_provider="google", api_key="...")

# HuggingFace
rag = RAGEngine(graph, llm_provider="huggingface", api_key="...")
```

---

## Best Practices

### Ontology Design

1. **Start with clear class hierarchies** - Use IS-A relationships properly
2. **Use specific relations** - "MITIGATES" is better than "AFFECTS"
3. **Include synonyms** - Help match varied terminology
4. **Define domain/range** - Constrain what makes sense
5. **Document everything** - Add descriptions to classes and relations

### Extraction

1. **Combine rule-based and LLM** - Rules for known entities, LLM for discovery
2. **Validate against ontology** - Normalize entities to canonical forms
3. **Set confidence thresholds** - Filter low-confidence extractions
4. **Review and refine** - Iterate on dictionaries and prompts

### Graph Building

1. **Start small** - Begin with core entities and relations
2. **Add incrementally** - Grow the graph over time
3. **Validate regularly** - Check against ontology constraints
4. **Monitor statistics** - Watch for anomalies in graph metrics

### RAG Queries

1. **Tune k parameter** - More context isn't always better
2. **Use hybrid search** - Combines semantic and structural signals
3. **Customize prompts** - Tailor to your domain
4. **Include sources** - Show provenance in answers

---

## Getting Help

- **Documentation**: https://ontorag.readthedocs.io
- **Issues**: https://github.com/ontorag/ontorag/issues
- **Discussions**: https://github.com/ontorag/ontorag/discussions
