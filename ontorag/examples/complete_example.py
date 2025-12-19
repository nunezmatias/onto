#!/usr/bin/env python3
"""
═══════════════════════════════════════════════════════════════════════════════
EJEMPLO COMPLETO: OntoRAG End-to-End
═══════════════════════════════════════════════════════════════════════════════

Este ejemplo demuestra el flujo completo de OntoRAG:

1. Crear ontología de adaptación climática
2. Construir grafo de conocimiento
3. Visualizar el grafo (HTML interactivo)
4. Analizar estadísticas
5. Consultar con RAG
6. Extraer tripletas de texto nuevo

Para ejecutar:
    python examples/complete_example.py

Requisitos:
    pip install ontorag[all]
"""

import os
import sys

# Agregar el path del proyecto
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ontorag import (
    Ontology,
    OntologyClass,
    OntologyRelation,
    RelationType,
    KnowledgeGraph,
    Node,
    Edge,
    GraphBuilder,
    RAGEngine,
    GraphAnalytics,
    OntologyAnalytics,
    GraphVisualizer,
    TripletExtractor,
)
from ontorag.rag import create_mock_llm
from ontorag.ontology import create_climate_ontology


def main():
    print("=" * 70)
    print(" 🔷 OntoRAG - Ejemplo Completo")
    print("=" * 70)
    
    # ═══════════════════════════════════════════════════════════════════════
    # PASO 1: Crear o cargar ontología
    # ═══════════════════════════════════════════════════════════════════════
    
    print("\n📚 PASO 1: Creando ontología de adaptación climática...")
    
    # Usar la ontología predefinida
    onto = create_climate_ontology()
    
    # O crear una personalizada:
    # onto = Ontology("mi_dominio", "Mi ontología personalizada")
    # onto.add_class(OntologyClass(...))
    # onto.add_relation(OntologyRelation(...))
    
    print(onto.summary())
    
    # Validar ontología
    print("\n🔍 Validando ontología...")
    validation = onto.validate()
    print(validation)
    
    # ═══════════════════════════════════════════════════════════════════════
    # PASO 2: Construir grafo de conocimiento
    # ═══════════════════════════════════════════════════════════════════════
    
    print("\n📊 PASO 2: Construyendo grafo de conocimiento...")
    
    # Método 1: Construcción directa
    kg = KnowledgeGraph(
        ontology=onto,
        name="climate_adaptation_graph",
        verbose=True
    )
    
    # Agregar nodos
    nodes = [
        Node(
            id="green_roof_bcn",
            name="Techos Verdes de Barcelona",
            node_type="green_roof",
            description="Programa de instalación de techos verdes en edificios de Barcelona",
            text_content="Barcelona ha implementado un programa de techos verdes que cubre más de 10,000 m² de superficie en el distrito del Eixample.",
            source_docs=["doc_001"]
        ),
        Node(
            id="urban_heat_bcn",
            name="Isla de Calor Urbana Barcelona",
            node_type="heat_wave",
            description="Efecto de isla de calor en el área metropolitana de Barcelona",
            text_content="El efecto isla de calor en Barcelona puede aumentar la temperatura hasta 4°C en comparación con áreas rurales circundantes.",
            source_docs=["doc_002"]
        ),
        Node(
            id="rain_garden_madrid",
            name="Jardines de Lluvia Madrid",
            node_type="rain_garden",
            description="Sistema de jardines de lluvia en Madrid para gestión de escorrentía",
            text_content="Madrid ha instalado más de 50 jardines de lluvia en parques públicos para reducir inundaciones urbanas.",
            source_docs=["doc_003"]
        ),
        Node(
            id="flood_risk_madrid",
            name="Riesgo de Inundación Madrid",
            node_type="flood",
            description="Zonas de riesgo de inundación en Madrid",
            text_content="Las zonas bajas de Madrid Río presentan riesgo de inundación durante eventos de precipitación extrema.",
            source_docs=["doc_003"]
        ),
        Node(
            id="early_warning_valencia",
            name="Sistema Alerta Temprana Valencia",
            node_type="early_warning_system",
            description="Sistema de alerta temprana para DANA en Valencia",
            text_content="Valencia cuenta con un sistema de alerta temprana que monitorea en tiempo real condiciones atmosféricas.",
            source_docs=["doc_004"]
        ),
        Node(
            id="storm_mediterranean",
            name="DANA Mediterráneo",
            node_type="storm",
            description="Depresión Aislada en Niveles Altos que afecta el Mediterráneo",
            text_content="Las DANAs son fenómenos recurrentes que causan lluvias torrenciales en la costa mediterránea española.",
            source_docs=["doc_004"]
        ),
        Node(
            id="agriculture_spain",
            name="Sector Agrícola España",
            node_type="agriculture_sector",
            description="Agricultura española vulnerable al cambio climático",
            source_docs=["doc_005"]
        ),
        Node(
            id="drought_spain",
            name="Sequía España",
            node_type="drought",
            description="Episodios de sequía en la Península Ibérica",
            text_content="España experimenta sequías cada vez más frecuentes que afectan principalmente a Andalucía y Murcia.",
            source_docs=["doc_005"]
        ),
    ]
    
    kg.add_nodes(nodes)
    
    # Agregar aristas con relaciones semánticas ricas
    edges = [
        Edge(
            source="green_roof_bcn",
            target="urban_heat_bcn",
            relation="MITIGATES",
            text_context="Los techos verdes reducen el efecto isla de calor mediante evapotranspiración",
            confidence=0.92
        ),
        Edge(
            source="rain_garden_madrid",
            target="flood_risk_madrid",
            relation="MITIGATES",
            text_context="Los jardines de lluvia absorben escorrentía y reducen riesgo de inundación",
            confidence=0.88
        ),
        Edge(
            source="rain_garden_madrid",
            target="flood_risk_madrid",
            relation="PREVENTS",
            text_context="Sistema preventivo de gestión de aguas pluviales",
            confidence=0.75
        ),
        Edge(
            source="early_warning_valencia",
            target="storm_mediterranean",
            relation="PROTECTS",
            text_context="El sistema de alerta protege a la población ante DANAs",
            confidence=0.95
        ),
        Edge(
            source="storm_mediterranean",
            target="flood_risk_madrid",
            relation="CAUSES",
            text_context="Las DANAs pueden causar inundaciones en zonas interiores",
            confidence=0.70
        ),
        Edge(
            source="drought_spain",
            target="agriculture_spain",
            relation="AFFECTS",
            text_context="Las sequías afectan severamente la producción agrícola",
            confidence=0.90
        ),
        Edge(
            source="urban_heat_bcn",
            target="health_sector",
            relation="AFFECTS",
            text_context="Las olas de calor urbano aumentan mortalidad en grupos vulnerables",
            confidence=0.85
        ),
    ]
    
    # Agregar nodo adicional para la última arista
    kg.add_node(Node(
        id="health_sector",
        name="Sector Salud",
        node_type="health_sector",
        description="Sistema de salud pública"
    ))
    
    kg.add_edges(edges)
    
    # Mostrar resumen del grafo
    print("\n" + kg.summary())
    
    # ═══════════════════════════════════════════════════════════════════════
    # PASO 3: Analizar estadísticas
    # ═══════════════════════════════════════════════════════════════════════
    
    print("\n📈 PASO 3: Analizando estadísticas del grafo...")
    
    analytics = GraphAnalytics(kg)
    print(analytics.summary())
    
    # Análisis de ontología
    onto_analytics = OntologyAnalytics(onto)
    print(onto_analytics.summary())
    
    # Cobertura de ontología
    coverage = onto_analytics.check_coverage(kg)
    print(f"\n📊 Cobertura de la ontología:")
    print(f"   Tipos de nodo: {coverage['node_types']['coverage_pct']:.1f}%")
    print(f"   Relaciones: {coverage['relations']['coverage_pct']:.1f}%")
    
    if coverage['node_types']['uncovered']:
        print(f"   ⚠️ Tipos no cubiertos: {coverage['node_types']['uncovered']}")
    
    # ═══════════════════════════════════════════════════════════════════════
    # PASO 4: Visualizar el grafo
    # ═══════════════════════════════════════════════════════════════════════
    
    print("\n🎨 PASO 4: Generando visualizaciones...")
    
    viz = GraphVisualizer(kg)
    
    # Generar grafo interactivo
    graph_path = viz.generate_html(
        "output/grafo_interactivo.html",
        title="Grafo de Adaptación Climática"
    )
    print(f"   ✅ Grafo interactivo: {graph_path}")
    
    # Generar dashboard
    dashboard_path = viz.generate_dashboard(
        "output/dashboard.html",
        title="OntoRAG Dashboard - Adaptación Climática"
    )
    print(f"   ✅ Dashboard: {dashboard_path}")
    
    # Visualizar ontología
    from ontorag.visualization import OntologyVisualizer
    onto_viz = OntologyVisualizer(onto)
    onto_path = onto_viz.generate_html(
        "output/ontologia.html",
        title="Ontología de Adaptación Climática"
    )
    print(f"   ✅ Ontología: {onto_path}")
    
    # ═══════════════════════════════════════════════════════════════════════
    # PASO 5: Consultar con RAG
    # ═══════════════════════════════════════════════════════════════════════
    
    print("\n🤖 PASO 5: Consultando con RAG...")
    
    # Usar LLM mock para el ejemplo (reemplazar con OpenAI/Anthropic en producción)
    llm = create_mock_llm()
    
    # Para usar OpenAI:
    # from ontorag.rag import create_openai_llm
    # llm = create_openai_llm(api_key="sk-...", model="gpt-4o-mini")
    
    rag = RAGEngine(
        knowledge_graph=kg,
        llm_function=llm,
        verbose=True
    )
    
    # Hacer consultas
    queries = [
        "¿Qué medidas mitigan el efecto isla de calor?",
        "¿Cómo se protege Valencia contra las tormentas?",
        "¿Qué impactos tiene la sequía en España?",
    ]
    
    for query in queries:
        print(f"\n{'─' * 60}")
        response = rag.query(query, k_nodes=5, k_edges=10)
        print(f"\n📝 Respuesta:\n{response.answer[:500]}...")
        print(f"\n📚 Fuentes: {response.citations}")
    
    # Explicar proceso de búsqueda
    print("\n🔍 Explicación del proceso de búsqueda:")
    explanation = rag.explain_retrieval("medidas contra inundaciones")
    print(explanation)
    
    # ═══════════════════════════════════════════════════════════════════════
    # PASO 6: Extraer tripletas de texto nuevo
    # ═══════════════════════════════════════════════════════════════════════
    
    print("\n📝 PASO 6: Extrayendo tripletas de texto nuevo...")
    
    nuevo_texto = """
    Sevilla ha implementado un innovador programa de bosques urbanos que mitiga 
    significativamente el efecto isla de calor en el casco histórico. Los árboles 
    proporcionan sombra y reducen la temperatura ambiente hasta 3 grados.
    
    Además, el nuevo sistema de humedales artificiales en el Guadalquivir previene 
    inundaciones durante las crecidas del río y protege las zonas residenciales 
    cercanas.
    """
    
    # Extractor basado en reglas (sin costo de API)
    from ontorag.extraction import RuleBasedExtractor
    
    rule_extractor = RuleBasedExtractor(ontology=onto)
    result = rule_extractor.extract(nuevo_texto)
    
    print(result.summary())
    
    # Convertir a elementos del grafo
    from ontorag.extraction import extraction_to_graph_elements
    new_nodes, new_edges = extraction_to_graph_elements(result, onto)
    
    print(f"\n   Nuevos nodos: {len(new_nodes)}")
    print(f"   Nuevas aristas: {len(new_edges)}")
    
    # ═══════════════════════════════════════════════════════════════════════
    # PASO 7: Guardar todo
    # ═══════════════════════════════════════════════════════════════════════
    
    print("\n💾 PASO 7: Guardando resultados...")
    
    # Guardar ontología
    onto.save("output/ontologia.json")
    print("   ✅ Ontología guardada: output/ontologia.json")
    
    # Guardar grafo
    kg.save("output/grafo/")
    print("   ✅ Grafo guardado: output/grafo/")
    
    # Exportar historial de consultas
    rag.export_history("output/historial_consultas.json")
    print("   ✅ Historial guardado: output/historial_consultas.json")
    
    # ═══════════════════════════════════════════════════════════════════════
    # RESUMEN FINAL
    # ═══════════════════════════════════════════════════════════════════════
    
    print("\n" + "=" * 70)
    print(" ✅ EJEMPLO COMPLETADO")
    print("=" * 70)
    print(f"""
Archivos generados:
  📊 output/grafo_interactivo.html  - Visualización interactiva del grafo
  📈 output/dashboard.html          - Dashboard con estadísticas
  📚 output/ontologia.html          - Visualización de la ontología
  💾 output/ontologia.json          - Ontología serializada
  📁 output/grafo/                  - Grafo completo persistido
  📝 output/historial_consultas.json - Historial de consultas RAG

Estadísticas finales:
  • Clases en ontología: {len(onto.classes)}
  • Relaciones en ontología: {len(onto.relations)}
  • Nodos en grafo: {kg.num_nodes}
  • Aristas en grafo: {kg.num_edges}
  • Consultas RAG realizadas: {len(rag.query_history)}

Para usar con un LLM real, reemplaza create_mock_llm() con:
  from ontorag.rag import create_openai_llm
  llm = create_openai_llm(api_key="sk-...", model="gpt-4o-mini")
""")


if __name__ == "__main__":
    # Crear directorio de output
    os.makedirs("output", exist_ok=True)
    main()
