"""
Minimal command-line interface for OntoRAG.

Provides a quick sanity check that core components work together without
requiring additional configuration.
"""

from __future__ import annotations

import argparse
import sys

from ontorag import (
    __version__,
    GraphAnalytics,
    KnowledgeGraph,
    Node,
)
from ontorag.ontology import create_climate_ontology


def run_quickstart() -> None:
    """Build a tiny demo graph and print basic metrics."""
    ontology = create_climate_ontology()
    kg = KnowledgeGraph(ontology=ontology, name="ontorag_cli_demo", verbose=False)

    kg.add_nodes(
        [
            Node(
                id="demo_green_roof",
                name="Techo Verde de Demostración",
                node_type="green_roof",
                description="Ejemplo de infraestructura verde para demostrar OntoRAG.",
                text_content="Los techos verdes ayudan a mitigar el efecto isla de calor.",
            ),
            Node(
                id="demo_heat_wave",
                name="Ola de Calor de Demostración",
                node_type="heat_wave",
                description="Ejemplo de peligro climático.",
                text_content="Las olas de calor afectan a áreas urbanas densas.",
            ),
        ]
    )

    analytics = GraphAnalytics(kg)
    summary = analytics.summary(compute_centrality=False)

    print("OntoRAG quickstart\n")
    print(f"Versión: {__version__}")
    print(summary)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="OntoRAG - Ontology-First Graph RAG framework"
    )
    parser.add_argument(
        "--version", action="store_true", help="Mostrar la versión instalada"
    )
    parser.add_argument(
        "--quickstart",
        action="store_true",
        help="Construir un grafo mínimo y mostrar métricas",
    )

    args = parser.parse_args(argv)

    if args.version:
        print(__version__)
        return 0

    if args.quickstart:
        run_quickstart()
        return 0

    parser.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
