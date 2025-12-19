"""
Colab smoke test for OntoRAG + Gemini on this branch.

Usage in Google Colab (single notebook cell):

```bash
!git clone -b onto2 https://github.com/nunezmatias/onto.git
%cd /content/onto
!pip install -e .
!pip install networkx chromadb sentence-transformers google-generativeai
!python examples/colab_gemini_smoke_test.py
```

Notes:
- Expects a `GOOGLE_API_KEY` value set in Colab secrets (`google.colab.userdata`)
  or as an environment variable. If missing, the script still runs and returns
  context-only answers.
- Uses the rule-based extractor and builds a graph from a simulated corpus,
  then runs a RAG query with a Gemini LLM callable.
"""

from __future__ import annotations

import os
import random
from typing import Callable, Dict, List, Optional

from ontorag import create_climate_ontology
from ontorag.extraction import RuleBasedExtractor
from ontorag.graph import KnowledgeGraph
from ontorag.rag import RAGEngine


def _get_google_api_key() -> Optional[str]:
    """Retrieve Google API key from Colab secrets or environment."""
    api_key = os.environ.get("GOOGLE_API_KEY")
    if api_key:
        return api_key

    try:
        from google.colab import userdata  # type: ignore

        api_key = userdata.get("GOOGLE_API_KEY")
    except Exception:
        api_key = None
    return api_key


def configure_gemini_llm(
    model: str = "gemini-1.5-flash", temperature: float = 0.2
) -> Optional[Callable[[str], str]]:
    """
    Create a Gemini-powered callable compatible with RAGEngine.

    Returns None if no API key is found so the script can still execute and
    return context-only responses.
    """
    api_key = _get_google_api_key()
    if not api_key:
        print("⚠️ GOOGLE_API_KEY not found; RAG answers will use context-only mode.")
        return None

    import google.generativeai as genai

    genai.configure(api_key=api_key)
    client = genai.GenerativeModel(model)

    def call_gemini(prompt: str) -> str:
        response = client.generate_content(
            prompt, generation_config={"temperature": temperature}
        )
        return response.text

    return call_gemini


def build_graph(documents: List[Dict[str, str]]) -> KnowledgeGraph:
    """Create a KnowledgeGraph from a list of document dicts."""
    ontology = create_climate_ontology()
    extractor = RuleBasedExtractor(ontology=ontology)
    kg = KnowledgeGraph(ontology=ontology)

    node_accumulator: Dict[str, Dict] = {}
    edge_accumulator: Dict[tuple, Dict] = {}

    for doc in documents:
        result = extractor.extract(doc["text"])

        for entity in result.entities:
            node_id = (entity.normalized or entity.text).lower().replace(" ", "_")
            entry = node_accumulator.setdefault(
                node_id,
                {
                    "type": entity.type or "entity",
                    "mention_count": 0,
                    "source_docs": set(),
                },
            )
            entry["mention_count"] += 1
            entry["source_docs"].add(doc["id"])

        for triplet in result.triplets:
            key = (triplet.subject.lower(), triplet.relation.upper(), triplet.object.lower())
            entry = edge_accumulator.setdefault(
                key, {"mention_count": 0, "source_docs": set()}
            )
            entry["mention_count"] += 1
            entry["source_docs"].add(doc["id"])

    for node_id, data in node_accumulator.items():
        kg.add_node(
            node_id=node_id,
            node_type=data["type"],
            properties={
                "mention_count": data["mention_count"],
                "source_docs": sorted(data["source_docs"]),
            },
        )

    for (source, relation, target), data in edge_accumulator.items():
        kg.add_edge(
            source=source,
            relation=relation,
            target=target,
            properties={
                "mention_count": data["mention_count"],
                "source_docs": sorted(data["source_docs"]),
            },
            create_nodes=True,
        )

    return kg


def main() -> None:
    base_paras = [
        "Los techos verdes mitigan la isla de calor urbana y reducen la escorrentía.",
        "Las inundaciones causan daños severos en zonas costeras y afectan a la agricultura.",
        "Los sistemas de alerta temprana protegen a la población ante tormentas intensas.",
        "Los humedales artificiales previenen inundaciones y complementan infraestructuras grises.",
    ]

    documents = [
        {"id": f"doc_{i:04d}", "text": random.choice(base_paras)} for i in range(200)
    ]

    kg = build_graph(documents)
    print(f"📈 Grafo listo -> Nodos: {kg.num_nodes}, Aristas: {kg.num_edges}")
    print("Resumen:", kg.stats())

    gemini_llm = configure_gemini_llm()
    rag = RAGEngine(kg, llm_callable=gemini_llm, k=15)

    question = "¿Qué medidas mitigan inundaciones y calor urbano?"
    response = rag.query(question, k=15)

    print("\n🤖 Respuesta del RAG:")
    print(response.answer)
    print("\n🔗 Fuentes:", response.sources)


if __name__ == "__main__":
    main()
