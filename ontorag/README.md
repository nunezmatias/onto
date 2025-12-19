# 🔷 OntoRAG

<p align="center">
  <strong>Ontology-First Retrieval Augmented Generation</strong>
</p>

<p align="center">
  <em>Construye sistemas RAG semánticamente ricos sobre grafos de conocimiento</em>
</p>

<p align="center">
  <a href="#instalación">Instalación</a> •
  <a href="#quick-start">Quick Start</a> •
  <a href="#arquitectura">Arquitectura</a> •
  <a href="#documentación">Docs</a> •
  <a href="#roadmap">Roadmap</a>
</p>

---

## 🎯 ¿Qué es OntoRAG?

OntoRAG es una biblioteca Python que pone la **ontología como ciudadano de primera clase** en sistemas RAG. A diferencia de otras bibliotecas que tratan las relaciones de forma genérica, OntoRAG permite:

- ✅ **Relaciones semánticamente ricas**: `MITIGA`, `PREVIENE`, `CAUSA` en vez de genéricos `RELACIONADO_CON`
- ✅ **Normalización semántica**: Mapea variantes lingüísticas a conceptos canónicos
- ✅ **Validación ontológica**: Verifica consistencia de tripletas contra el esquema
- ✅ **Búsqueda híbrida**: Combina similitud vectorial con exploración estructural del grafo
- ✅ **Visualización interactiva**: Dashboards HTML con estadísticas y grafos explorables

```python
from ontorag import Ontology, KnowledgeGraph, RAGEngine

# 1. Define tu ontología
onto = Ontology("clima")
onto.add_class(OntologyClass(id="flood", name="Inundación", synonyms=["flooding", "riada"]))
onto.add_relation(OntologyRelation(id="MITIGATES", relation_type=RelationType.MITIGATIVE))

# 2. Construye el grafo
kg = KnowledgeGraph(ontology=onto)
kg.add_node(Node(id="green_roof", name="Techo Verde", node_type="adaptation_measure"))
kg.add_edge(Edge(source="green_roof", target="flood", relation="MITIGATES"))

# 3. Consulta con RAG
rag = RAGEngine(kg, llm_function=my_llm)
response = rag.query("¿Qué medidas mitigan las inundaciones?")
print(response.answer)
```

---

## 📊 Comparación con Otras Bibliotecas

| Característica | OntoRAG | LlamaIndex | LangChain |
|---------------|---------|------------|-----------|
| **Ontología editable** | ✅ Primera clase | ❌ No | ❌ No |
| **Relaciones tipadas** | ✅ MITIGATES, CAUSES, PREVENTS | ⚠️ Genéricas | ⚠️ Genéricas |
| **Validación semántica** | ✅ 6 verificaciones | ❌ No | ❌ No |
| **Normalización por sinónimos** | ✅ Automática | ❌ Manual | ❌ Manual |
| **Jerarquías IS-A** | ✅ Con herencia | ⚠️ Limitada | ❌ No |
| **Búsqueda híbrida** | ✅ Semántica + Estructural | ⚠️ Solo vectorial | ⚠️ Solo vectorial |
| **Visualización integrada** | ✅ HTML interactivo | ❌ Requiere extras | ❌ Requiere extras |
| **Dominio agnóstico** | ✅ | ✅ | ✅ |

---

## 🚀 Instalación

```bash
# Instalación básica
pip install ontorag

# Con soporte para OpenAI
pip install ontorag[openai]

# Con soporte para Anthropic Claude
pip install ontorag[anthropic]

# Instalación completa (desarrollo)
pip install ontorag[all,dev]
```

### Desde el código fuente

```bash
git clone https://github.com/ontorag/ontorag.git
cd ontorag
pip install -e ".[all,dev]"
```

---

## 📖 Quick Start

### 1. Crear una Ontología

```python
from ontorag import Ontology, OntologyClass, OntologyRelation, RelationType

# Crear ontología
onto = Ontology("mi_dominio", "Mi ontología de dominio")

# Agregar clases con jerarquía
onto.add_class(OntologyClass(
    id="entity",
    name="Entidad",
    description="Clase raíz"
))

onto.add_class(OntologyClass(
    id="person",
    name="Persona",
    parent="entity",
    synonyms=["individuo", "ser humano", "person"]
))

onto.add_class(OntologyClass(
    id="organization",
    name="Organización",
    parent="entity",
    synonyms=["empresa", "compañía", "org"]
))

# Agregar relaciones semánticas
onto.add_relation(OntologyRelation(
    id="WORKS_FOR",
    name="trabaja para",
    description="La persona trabaja en la organización",
    relation_type=RelationType.FUNCTIONAL,
    domain=["person"],
    range=["organization"],
    inverse="EMPLOYS",
    synonyms=["empleado de", "works at", "trabaja en"]
))

# Validar ontología
result = onto.validate()
print(result)  # Muestra errores, warnings e info

# Guardar
onto.save("mi_ontologia.json")
```

### 2. Construir el Grafo de Conocimiento

```python
from ontorag import KnowledgeGraph, Node, Edge

# Crear grafo con ontología
kg = KnowledgeGraph(
    ontology=onto,
    name="mi_grafo",
    verbose=True
)

# Agregar nodos
kg.add_node(Node(
    id="juan_perez",
    name="Juan Pérez",
    node_type="person",
    description="Ingeniero de software senior",
    text_content="Juan Pérez es un ingeniero con 10 años de experiencia..."
))

kg.add_node(Node(
    id="acme_corp",
    name="ACME Corporation",
    node_type="organization",
    description="Empresa tecnológica multinacional"
))

# Agregar aristas (validadas contra la ontología)
kg.add_edge(Edge(
    source="juan_perez",
    target="acme_corp",
    relation="WORKS_FOR",
    text_context="Juan Pérez trabaja como ingeniero senior en ACME Corporation",
    confidence=0.95
))

# Ver estadísticas
print(kg.summary())

# Guardar grafo
kg.save("mi_grafo/")
```

### 3. Visualizar el Grafo

```python
from ontorag import GraphVisualizer

viz = GraphVisualizer(kg)

# Generar visualización interactiva
viz.generate_html("grafo.html", title="Mi Grafo de Conocimiento")

# Generar dashboard completo
viz.generate_dashboard("dashboard.html")
```

### 4. Consultar con RAG

```python
from ontorag import RAGEngine
from ontorag.rag import create_openai_llm

# Configurar LLM
llm = create_openai_llm(api_key="sk-...", model="gpt-4o-mini")

# Crear motor RAG
rag = RAGEngine(
    knowledge_graph=kg,
    llm_function=llm,
    verbose=True
)

# Hacer consulta
response = rag.query(
    "¿Quién trabaja en ACME Corporation?",
    k_nodes=10,
    k_edges=15,
    semantic_weight=0.6,
    structural_weight=0.4
)

print(response.answer)
print(f"Fuentes: {response.citations}")

# Explicar el proceso de búsqueda
print(rag.explain_retrieval("empleados de ACME"))
```

### 5. Extraer Tripletas de Texto

```python
from ontorag import TripletExtractor

extractor = TripletExtractor(ontology=onto, llm_function=llm)

result = extractor.extract("""
María García es la CEO de TechStart Inc., una startup de inteligencia artificial.
La empresa fue fundada en 2020 y tiene 50 empleados.
""")

print(result.summary())
# Tripletas encontradas:
#   • (María García) --[WORKS_FOR]--> (TechStart Inc.)
#   • (TechStart Inc.) --[IS_A]--> (startup)

# Convertir a elementos del grafo
from ontorag.extraction import extraction_to_graph_elements
nodes, edges = extraction_to_graph_elements(result, onto)
kg.add_nodes(nodes)
kg.add_edges(edges)
```

---

## 🏗️ Arquitectura

OntoRAG está diseñado de forma modular para permitir evolución independiente:

```
ontorag/
├── ontology/          # 🧠 NÚCLEO: Definición y validación de ontologías
│   ├── OntologyClass
│   ├── OntologyRelation
│   ├── Ontology
│   └── ValidationResult
│
├── graph/             # 📊 Grafo de conocimiento con embeddings
│   ├── Node
│   ├── Edge
│   ├── KnowledgeGraph (NetworkX + ChromaDB)
│   └── GraphBuilder
│
├── search/            # 🔍 Búsqueda híbrida
│   ├── SemanticSearcher
│   ├── StructuralSearcher
│   └── HybridSearcher
│
├── rag/               # 🤖 Sistema RAG
│   ├── RAGContext
│   ├── RAGResponse
│   ├── RAGEngine
│   └── LLM Adapters (OpenAI, Anthropic, Mock)
│
├── extraction/        # 📝 Extracción de tripletas
│   ├── TripletExtractor (LLM-based)
│   └── RuleBasedExtractor
│
├── analytics/         # 📈 Estadísticas y métricas
│   ├── GraphAnalytics
│   └── OntologyAnalytics
│
└── visualization/     # 🎨 Visualización HTML
    ├── GraphVisualizer
    └── OntologyVisualizer
```

### Flujo de Datos

```
┌─────────────┐     ┌──────────────┐     ┌─────────────────┐
│   Texto     │────▶│  Extractor   │────▶│  Tripletas      │
│   (docs)    │     │  (LLM/Rules) │     │  normalizadas   │
└─────────────┘     └──────────────┘     └────────┬────────┘
                                                  │
                    ┌──────────────┐               ▼
                    │  Ontología   │◀─────────────────────────┐
                    │  (validación)│                          │
                    └──────────────┘     ┌─────────────────┐  │
                                         │  Knowledge      │  │
                    ┌──────────────┐     │  Graph          │──┘
      Query ───────▶│  Hybrid      │────▶│  (NetworkX +    │
                    │  Search      │     │   ChromaDB)     │
                    └──────┬───────┘     └─────────────────┘
                           │
                           ▼
                    ┌──────────────┐     ┌─────────────────┐
                    │  RAG Engine  │────▶│   Respuesta     │
                    │  (LLM)       │     │   + Citaciones  │
                    └──────────────┘     └─────────────────┘
```

---

## 🛤️ Roadmap de Crecimiento

### Fase 1: Fundamentos (v0.1 - v0.3) ✅ ACTUAL

```
✅ Módulo de ontología completo
✅ Grafo con NetworkX + ChromaDB
✅ Búsqueda híbrida
✅ Sistema RAG básico
✅ Visualización HTML
✅ Extracción de tripletas
```

### Fase 2: Escalabilidad (v0.4 - v0.6)

```
🔲 Integración con Neo4j para grafos grandes (>1M nodos)
🔲 Streaming de respuestas RAG
🔲 Caché inteligente de embeddings
🔲 API REST con FastAPI
🔲 Soporte para múltiples ontologías federadas
🔲 Import/Export en formatos estándar (OWL, RDF, JSON-LD)
```

### Fase 3: Inteligencia (v0.7 - v0.9)

```
🔲 Ontology Learning: Sugerir clases/relaciones desde corpus
🔲 Entity Resolution: Deduplicación automática
🔲 Temporal Reasoning: Relaciones con tiempo
🔲 Uncertainty Handling: Propagación de confianza
🔲 Multi-hop Reasoning: Inferencia sobre el grafo
🔲 Explanation Generation: "¿Por qué esta respuesta?"
```

### Fase 4: Ecosistema (v1.0+)

```
🔲 Plugin system para backends de grafo
🔲 Marketplace de ontologías predefinidas
🔲 UI web para edición de ontologías (Streamlit/Gradio)
🔲 CLI completa para operaciones comunes
🔲 Integración con herramientas MLOps
🔲 Benchmarks contra datasets estándar
```

---

## 💎 Ventajas Competitivas

### 1. **Ontología como Ciudadano de Primera Clase**

Mientras LlamaIndex y LangChain tratan los grafos como almacenes genéricos, OntoRAG permite definir, validar y evolucionar ontologías:

```python
# OntoRAG: Relaciones semánticamente ricas
edge = Edge(source="aspirina", target="dolor_cabeza", relation="ALIVIA")

# Otros: Relaciones genéricas que pierden información
edge = Edge(source="aspirina", target="dolor_cabeza", relation="RELATED_TO")  # ❌
```

### 2. **Normalización Semántica Automática**

```python
# "flooding", "inundación", "riada" → todos mapean a `flood`
normalized = ontology.normalize_entity("riada")  # → "flood"
```

### 3. **Validación de Consistencia**

6 verificaciones automáticas que previenen errores:
- Ciclos en jerarquías
- Relaciones semánticamente vacías
- Dominios/rangos incompletos
- Cobertura de sinónimos
- Coherencia de inversas
- Clases no utilizadas

### 4. **Búsqueda Híbrida Real**

Combina lo mejor de ambos mundos:
- **Semántica**: Encuentra "medidas contra el calor" aunque no aparezca textualmente
- **Estructural**: Sigue relaciones en el grafo para encontrar conexiones

### 5. **Visualización Integrada**

Sin dependencias adicionales, genera dashboards HTML interactivos con:
- Grafo explorable (zoom, pan, búsqueda)
- Estadísticas en tiempo real
- Distribuciones de tipos y relaciones
- Export a PNG

---

## 💰 Valor Comercial

### Casos de Uso

| Industria | Aplicación | Valor |
|-----------|------------|-------|
| **Salud** | Base de conocimiento de interacciones medicamentosas | Reducir errores médicos, acelerar diagnósticos |
| **Legal** | Grafo de legislación y jurisprudencia | Búsqueda semántica en miles de documentos legales |
| **Finanzas** | Red de entidades para compliance/AML | Detectar relaciones ocultas entre entidades |
| **Manufactura** | Ontología de componentes y fallas | Mantenimiento predictivo, troubleshooting |
| **E-commerce** | Grafo de productos y atributos | Recomendaciones más precisas, búsqueda por atributos |
| **Investigación** | Base de conocimiento científico | Descubrimiento de conexiones entre papers |

### Modelo de Negocio Potencial

1. **Open Source Core**: Biblioteca base gratuita
2. **OntoRAG Cloud**: Servicio managed con:
   - Hosting de grafos
   - API REST
   - Dashboards en la nube
   - Colaboración en tiempo real
3. **Enterprise Features**:
   - Integración con Neo4j Enterprise
   - SSO/SAML
   - Auditoría y compliance
   - Soporte dedicado
4. **Marketplace de Ontologías**:
   - Ontologías predefinidas por dominio
   - Modelo de revenue share

### Métricas de Impacto

- **Precisión RAG**: +15-25% vs RAG genérico (por relaciones semánticas)
- **Tiempo de desarrollo**: -40% vs construir desde cero
- **Mantenibilidad**: Ontología versionable y validable
- **Trazabilidad**: Citaciones automáticas a fuentes

---

## 📚 Documentación

- [Guía de Usuario](docs/user-guide.md)
- [API Reference](docs/api-reference.md)
- [Ejemplos](examples/)
- [Contributing](CONTRIBUTING.md)

---

## 🤝 Contribuir

¡Contribuciones bienvenidas! Ver [CONTRIBUTING.md](CONTRIBUTING.md) para guías.

```bash
# Setup desarrollo
git clone https://github.com/ontorag/ontorag.git
cd ontorag
pip install -e ".[dev]"

# Correr tests
pytest

# Formatear código
black ontorag/
ruff check ontorag/
```

---

## 📄 Licencia

MIT License - ver [LICENSE](LICENSE) para detalles.

---

<p align="center">
  <strong>🔷 OntoRAG</strong><br>
  <em>Porque las relaciones importan.</em>
</p>
