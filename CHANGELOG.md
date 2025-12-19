# Changelog

All notable changes to OntoRAG will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.0] - 2024-12-19

### Added

- **Corpus Module**: Multi-source academic paper collection
  - ArxivSource: Search and retrieve arXiv papers
  - SemanticScholarSource: Integration with Semantic Scholar API
  - OpenAlexSource: Access to OpenAlex academic database
  - Corpus class for managing document collections
  - CorpusAnalyzer for statistics and visualization

- **Enhanced Extraction Module**:
  - LLMExtractor for LLM-powered entity and relation extraction
  - Support for OpenAI, Anthropic, HuggingFace, and Google providers
  - HybridExtractor combining rule-based and LLM approaches
  - Configurable extraction prompts
  - Batch processing with progress tracking

- **JSON Ontology Support**:
  - Load ontologies from JSON files
  - Predefined climate adaptation ontology template
  - Comprehensive validation on load

- **Improved Documentation**:
  - Complete user guide
  - API reference structure
  - Six comprehensive examples
  - Inline code documentation

### Changed

- Reorganized package structure for better modularity
- Improved error handling across all modules
- Enhanced type hints for IDE support

### Fixed

- Entity normalization edge cases
- Graph serialization with special characters
- Memory leak in corpus collection

## [0.1.0] - 2024-12-01

### Added

- Initial release
- Ontology module with classes and relations
- KnowledgeGraph with NetworkX + ChromaDB backend
- Rule-based triplet extraction
- Basic RAG engine
- Climate adaptation ontology
- HTML graph visualization
