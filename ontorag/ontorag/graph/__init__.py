"""
╔═══════════════════════════════════════════════════════════════════════════════╗
║                         ONTORAG - MÓDULO GRAPH                                ║
║                                                                               ║
║  Construcción y gestión de Grafos de Conocimiento.                           ║
║                                                                               ║
║  Características:                                                             ║
║  - Validación automática contra ontología                                    ║
║  - Embeddings con Sentence Transformers                                      ║
║  - Búsqueda vectorial con ChromaDB                                           ║
║  - Estadísticas completas del grafo                                          ║
║  - Persistencia completa                                                     ║
╚═══════════════════════════════════════════════════════════════════════════════╝
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Any, Tuple, Iterator
from pathlib import Path
from datetime import datetime
import json
import hashlib
import pickle

# Dependencias
try:
    import networkx as nx
except ImportError:
    nx = None

try:
    import chromadb
except ImportError:
    chromadb = None

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    SentenceTransformer = None

from ontorag.ontology import Ontology


# ═══════════════════════════════════════════════════════════════════════════════
# NODE
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class Node:
    """
    Nodo en el grafo de conocimiento.
    
    Attributes:
        id: Identificador único
        name: Etiqueta legible
        node_type: Tipo según la ontología
        description: Descripción del nodo
        properties: Propiedades adicionales
        text_content: Contenido textual para embeddings
        source_docs: IDs de documentos fuente
    """
    id: str
    name: str
    node_type: str
    description: str = ""
    properties: Dict[str, Any] = field(default_factory=dict)
    text_content: str = ""
    source_docs: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        self.id = self._normalize_id(self.id)
    
    @staticmethod
    def _normalize_id(text: str) -> str:
        import re
        text = re.sub(r'[^\w\s-]', '', str(text))
        return re.sub(r'\s+', '_', text.strip()).lower()[:100]

    @property
    def label(self) -> str:
        """Alias de compatibilidad para 'name'."""
        return self.name

    @label.setter
    def label(self, value: str):
        self.name = value
    
    def to_embedding_text(self) -> str:
        """Texto para calcular embedding."""
        parts = [f"[{self.node_type}]", self.name]
        if self.description:
            parts.append(self.description)
        if self.text_content:
            parts.append(self.text_content[:500])
        return " ".join(parts)
    
    def to_dict(self) -> Dict:
        base = {k: v for k, v in self.__dict__.items()}
        base["label"] = self.name  # Compatibilidad con versiones previas/visualizaciones
        base["name"] = self.name
        return base
    
    @classmethod
    def from_dict(cls, d: Dict) -> "Node":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


# ═══════════════════════════════════════════════════════════════════════════════
# EDGE
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class Edge:
    """
    Arista en el grafo de conocimiento.
    
    Attributes:
        source: ID del nodo origen
        target: ID del nodo destino
        relation: Tipo de relación (según ontología)
        text_context: Contexto textual de donde se extrajo
        confidence: Nivel de confianza (0-1)
        source_docs: IDs de documentos fuente
    """
    source: str
    target: str
    relation: str
    text_context: str = ""
    confidence: float = 1.0
    properties: Dict[str, Any] = field(default_factory=dict)
    source_docs: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        self.relation = self.relation.upper().replace(" ", "_")
        self.confidence = max(0.0, min(1.0, self.confidence))
    
    @property
    def id(self) -> str:
        return f"{self.source}__{self.relation}__{self.target}"
    
    def to_embedding_text(self) -> str:
        parts = [self.source, self.relation, self.target]
        if self.text_context:
            parts.append(self.text_context[:300])
        return " ".join(parts)
    
    def to_dict(self) -> Dict:
        return {"id": self.id, **{k: v for k, v in self.__dict__.items()}}
    
    @classmethod
    def from_dict(cls, d: Dict) -> "Edge":
        d = {k: v for k, v in d.items() if k in cls.__dataclass_fields__}
        return cls(**d)


# ═══════════════════════════════════════════════════════════════════════════════
# GRAPH STATISTICS
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class GraphStatistics:
    """
    Estadísticas completas del grafo de conocimiento.
    """
    # Básicas
    num_nodes: int = 0
    num_edges: int = 0
    density: float = 0.0
    avg_degree: float = 0.0
    
    # Conectividad
    num_components: int = 0
    largest_component_size: int = 0
    is_connected: bool = False
    
    # Distribuciones
    nodes_by_type: Dict[str, int] = field(default_factory=dict)
    edges_by_relation: Dict[str, int] = field(default_factory=dict)
    degree_distribution: Dict[int, int] = field(default_factory=dict)
    
    # Centralidad (top nodos)
    top_degree: List[Dict] = field(default_factory=list)
    top_betweenness: List[Dict] = field(default_factory=list)
    top_pagerank: List[Dict] = field(default_factory=list)
    
    # Cobertura ontología
    ontology_coverage: Dict[str, float] = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        return {k: v for k, v in self.__dict__.items()}
    
    def summary(self) -> str:
        lines = [
            f"\n{'═'*70}",
            f"{'ESTADÍSTICAS DEL GRAFO':^70}",
            f"{'═'*70}",
            f"",
            f"  📊 MÉTRICAS BÁSICAS",
            f"  {'─'*40}",
            f"  Nodos: {self.num_nodes:,}",
            f"  Aristas: {self.num_edges:,}",
            f"  Densidad: {self.density:.4f}",
            f"  Grado promedio: {self.avg_degree:.2f}",
            f"",
            f"  🔗 CONECTIVIDAD",
            f"  {'─'*40}",
            f"  Componentes: {self.num_components}",
            f"  Mayor componente: {self.largest_component_size} nodos",
            f"",
            f"  📈 DISTRIBUCIÓN DE NODOS",
            f"  {'─'*40}",
        ]
        
        for ntype, count in sorted(self.nodes_by_type.items(), key=lambda x: -x[1])[:8]:
            pct = count / self.num_nodes * 100 if self.num_nodes else 0
            bar = "█" * min(int(pct/2), 20)
            lines.append(f"  {ntype[:25]:25} {count:>5} ({pct:>5.1f}%) {bar}")
        
        lines.extend([
            f"",
            f"  📈 DISTRIBUCIÓN DE RELACIONES",
            f"  {'─'*40}",
        ])
        
        for rel, count in sorted(self.edges_by_relation.items(), key=lambda x: -x[1])[:8]:
            pct = count / self.num_edges * 100 if self.num_edges else 0
            bar = "█" * min(int(pct/2), 20)
            lines.append(f"  {rel[:25]:25} {count:>5} ({pct:>5.1f}%) {bar}")
        
        if self.top_degree:
            lines.extend([
                f"",
                f"  🏆 NODOS MÁS CONECTADOS (Top 5)",
                f"  {'─'*40}",
            ])
            for i, node in enumerate(self.top_degree[:5], 1):
                lines.append(f"  {i}. {node['label'][:30]:30} (grado: {node['degree']})")
        
        lines.append(f"{'═'*70}")
        return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════════════
# KNOWLEDGE GRAPH
# ═══════════════════════════════════════════════════════════════════════════════

class KnowledgeGraph:
    """
    Grafo de Conocimiento con búsqueda semántica.
    
    Combina:
    - NetworkX para estructura y consultas de grafo
    - ChromaDB para búsqueda vectorial
    - Ontología para validación semántica
    
    Ejemplo:
        >>> from ontorag import KnowledgeGraph, Ontology, Node, Edge
        >>> 
        >>> onto = Ontology.load("climate.json")
        >>> kg = KnowledgeGraph(ontology=onto)
        >>> 
        >>> kg.add_node(Node("n1", "Techo Verde BCN", "green_roof"))
        >>> kg.add_node(Node("n2", "Isla de Calor", "heat_wave"))
        >>> kg.add_edge(Edge("n1", "n2", "MITIGATES"))
        >>> 
        >>> # Estadísticas
        >>> stats = kg.get_statistics()
        >>> print(stats.summary())
    """
    
    DEFAULT_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    
    def __init__(
        self,
        ontology: Optional[Ontology] = None,
        name: str = "knowledge_graph",
        embedding_model: str = None,
        persist_path: str = None,
        verbose: bool = True
    ):
        self.name = name
        self.ontology = ontology
        self.verbose = verbose
        self.persist_path = persist_path
        self.created_at = datetime.now().isoformat()

        if nx is None:
            raise ImportError("Instala networkx: pip install networkx")
        
        self._log(f"🔷 Inicializando KnowledgeGraph: {name}")
        
        # NetworkX
        self.graph = nx.MultiDiGraph()
        
        # Embeddings
        self.embedding_model_name = embedding_model or self.DEFAULT_MODEL
        self._embedding_model = None
        self._embedding_cache: Dict[str, List[float]] = {}
        
        # ChromaDB
        self._chroma_client = None
        self._nodes_collection = None
        self._edges_collection = None
        
        if persist_path:
            Path(persist_path).mkdir(parents=True, exist_ok=True)
        
        self._log(f"   ✓ Grafo inicializado")
    
    def _log(self, msg: str):
        if self.verbose:
            print(msg)
    
    def _get_embedding_model(self):
        """Carga modelo de embeddings (lazy loading)."""
        if self._embedding_model is None:
            if SentenceTransformer is None:
                raise ImportError("pip install sentence-transformers")
            self._log(f"🔷 Cargando modelo: {self.embedding_model_name}")
            self._embedding_model = SentenceTransformer(self.embedding_model_name)
        return self._embedding_model
    
    def _get_chroma(self):
        """Inicializa ChromaDB (lazy loading)."""
        if self._chroma_client is None:
            if chromadb is None:
                raise ImportError("pip install chromadb")
            if self.persist_path:
                self._chroma_client = chromadb.PersistentClient(
                    path=str(Path(self.persist_path) / "chroma"))
            else:
                self._chroma_client = chromadb.Client()
            
            self._nodes_collection = self._chroma_client.get_or_create_collection(
                f"{self.name}_nodes")
            self._edges_collection = self._chroma_client.get_or_create_collection(
                f"{self.name}_edges")
        return self._chroma_client
    
    def _get_embedding(self, text: str) -> List[float]:
        """Calcula embedding con caché."""
        text_hash = hashlib.md5(text.encode()).hexdigest()[:16]
        if text_hash not in self._embedding_cache:
            model = self._get_embedding_model()
            emb = model.encode(text, convert_to_numpy=True).tolist()
            self._embedding_cache[text_hash] = emb
        return self._embedding_cache[text_hash]
    
    # ═══════════════════════════════════════════════════════════════════════════
    # GESTIÓN DE NODOS
    # ═══════════════════════════════════════════════════════════════════════════
    
    def add_node(self, node: Node, validate: bool = True) -> "KnowledgeGraph":
        """Agrega un nodo al grafo."""
        # Validar y normalizar tipo
        if validate and self.ontology:
            normalized = self.ontology.normalize_entity(node.node_type)
            if normalized:
                node.node_type = normalized
        
        # Agregar a NetworkX
        self.graph.add_node(node.id, **node.to_dict())
        
        # Agregar a ChromaDB
        self._get_chroma()
        emb_text = node.to_embedding_text()
        embedding = self._get_embedding(emb_text)
        
        self._nodes_collection.upsert(
            ids=[node.id],
            documents=[emb_text],
            embeddings=[embedding],
            metadatas=[{"label": node.label, "type": node.node_type}]
        )
        
        return self
    
    def add_nodes(self, nodes: List[Node], validate: bool = True) -> "KnowledgeGraph":
        """Agrega múltiples nodos (batch)."""
        self._log(f"🔷 Agregando {len(nodes)} nodos...")
        
        ids, docs, metas = [], [], []
        
        for node in nodes:
            if validate and self.ontology:
                normalized = self.ontology.normalize_entity(node.node_type)
                if normalized:
                    node.node_type = normalized
            
            self.graph.add_node(node.id, **node.to_dict())
            
            ids.append(node.id)
            docs.append(node.to_embedding_text())
            metas.append({"label": node.label, "type": node.node_type})
        
        # Batch embeddings
        model = self._get_embedding_model()
        embeddings = model.encode(docs, convert_to_numpy=True, 
                                  show_progress_bar=self.verbose).tolist()
        
        self._get_chroma()
        self._nodes_collection.upsert(ids=ids, documents=docs, 
                                      embeddings=embeddings, metadatas=metas)
        
        self._log(f"   ✓ {len(nodes)} nodos agregados")
        return self
    
    def get_node(self, node_id: str) -> Optional[Dict]:
        if node_id in self.graph:
            return dict(self.graph.nodes[node_id])
        return None
    
    def has_node(self, node_id: str) -> bool:
        return node_id in self.graph
    
    # ═══════════════════════════════════════════════════════════════════════════
    # GESTIÓN DE ARISTAS
    # ═══════════════════════════════════════════════════════════════════════════
    
    def add_edge(self, edge: Edge, validate: bool = True) -> "KnowledgeGraph":
        """Agrega una arista al grafo."""
        if edge.source not in self.graph:
            self._log(f"   ⚠️ Nodo origen '{edge.source}' no existe")
            return self
        if edge.target not in self.graph:
            self._log(f"   ⚠️ Nodo destino '{edge.target}' no existe")
            return self
        
        # Validar relación
        if validate and self.ontology:
            normalized = self.ontology.normalize_relation(edge.relation)
            if normalized:
                edge.relation = normalized
            
            # Validar dominio/rango
            relation = self.ontology.get_relation(edge.relation)
            if relation:
                source_type = self.graph.nodes[edge.source].get("node_type", "")
                target_type = self.graph.nodes[edge.target].get("node_type", "")
                hierarchies = self.ontology.get_all_hierarchies()
                valid, msg = relation.validate_triple(source_type, target_type, hierarchies)
                if not valid:
                    self._log(f"   ⚠️ {msg}")
        
        # Agregar a NetworkX
        self.graph.add_edge(edge.source, edge.target, key=edge.id, **edge.to_dict())
        
        # Agregar a ChromaDB
        self._get_chroma()
        emb_text = edge.to_embedding_text()
        embedding = self._get_embedding(emb_text)
        
        self._edges_collection.upsert(
            ids=[edge.id],
            documents=[emb_text],
            embeddings=[embedding],
            metadatas={"source": edge.source, "target": edge.target, 
                       "relation": edge.relation}
        )
        
        return self
    
    def add_edges(self, edges: List[Edge], validate: bool = True) -> "KnowledgeGraph":
        """Agrega múltiples aristas (batch)."""
        self._log(f"🔷 Agregando {len(edges)} aristas...")
        
        ids, docs, metas = [], [], []
        valid_edges = []
        
        for edge in edges:
            if edge.source not in self.graph or edge.target not in self.graph:
                continue
            
            if validate and self.ontology:
                normalized = self.ontology.normalize_relation(edge.relation)
                if normalized:
                    edge.relation = normalized
            
            self.graph.add_edge(edge.source, edge.target, key=edge.id, **edge.to_dict())
            
            ids.append(edge.id)
            docs.append(edge.to_embedding_text())
            metas.append({"source": edge.source, "target": edge.target,
                         "relation": edge.relation})
            valid_edges.append(edge)
        
        if ids:
            model = self._get_embedding_model()
            embeddings = model.encode(docs, convert_to_numpy=True,
                                     show_progress_bar=self.verbose).tolist()
            
            self._get_chroma()
            self._edges_collection.upsert(ids=ids, documents=docs,
                                         embeddings=embeddings, metadatas=metas)
        
        self._log(f"   ✓ {len(valid_edges)} aristas agregadas")
        return self
    
    def get_edges(self, source: str = None, target: str = None, 
                  relation: str = None) -> List[Dict]:
        """Obtiene aristas filtradas."""
        edges = []
        for u, v, data in self.graph.edges(data=True):
            if source and u != source:
                continue
            if target and v != target:
                continue
            if relation and data.get("relation") != relation:
                continue
            edges.append({"source": u, "target": v, **data})
        return edges
    
    # ═══════════════════════════════════════════════════════════════════════════
    # PROPIEDADES
    # ═══════════════════════════════════════════════════════════════════════════
    
    @property
    def num_nodes(self) -> int:
        return self.graph.number_of_nodes()
    
    @property
    def num_edges(self) -> int:
        return self.graph.number_of_edges()
    
    # ═══════════════════════════════════════════════════════════════════════════
    # ESTADÍSTICAS
    # ═══════════════════════════════════════════════════════════════════════════
    
    def get_statistics(self, compute_centrality: bool = True) -> GraphStatistics:
        """
        Calcula estadísticas completas del grafo.
        
        Args:
            compute_centrality: Si calcular centralidad (puede ser lento)
        """
        stats = GraphStatistics()
        stats.num_nodes = self.num_nodes
        stats.num_edges = self.num_edges
        
        if stats.num_nodes == 0:
            return stats
        
        # Densidad y grado
        stats.density = nx.density(self.graph)
        degrees = dict(self.graph.degree())
        stats.avg_degree = sum(degrees.values()) / stats.num_nodes
        
        # Distribución de grados
        for deg in degrees.values():
            stats.degree_distribution[deg] = stats.degree_distribution.get(deg, 0) + 1
        
        # Conectividad
        try:
            stats.num_components = nx.number_weakly_connected_components(self.graph)
            stats.is_connected = nx.is_weakly_connected(self.graph)
            components = list(nx.weakly_connected_components(self.graph))
            if components:
                stats.largest_component_size = max(len(c) for c in components)
        except:
            pass
        
        # Distribución por tipo
        for nid in self.graph.nodes():
            ntype = self.graph.nodes[nid].get("node_type", "unknown")
            stats.nodes_by_type[ntype] = stats.nodes_by_type.get(ntype, 0) + 1
        
        # Distribución por relación
        for u, v, data in self.graph.edges(data=True):
            rel = data.get("relation", "unknown")
            stats.edges_by_relation[rel] = stats.edges_by_relation.get(rel, 0) + 1
        
        # Centralidad
        if compute_centrality and stats.num_nodes > 1:
            self._compute_centrality(stats)
        
        # Cobertura ontología
        if self.ontology:
            stats.ontology_coverage = self._compute_ontology_coverage()
        
        return stats
    
    def _compute_centrality(self, stats: GraphStatistics, top_k: int = 10):
        """Calcula métricas de centralidad."""
        # Grado
        degrees = dict(self.graph.degree())
        sorted_deg = sorted(degrees.items(), key=lambda x: -x[1])
        stats.top_degree = [
            {"id": nid, "label": self.graph.nodes[nid].get("label", nid),
             "type": self.graph.nodes[nid].get("node_type", ""), "degree": deg}
            for nid, deg in sorted_deg[:top_k]
        ]
        
        # PageRank
        try:
            pr = nx.pagerank(self.graph, alpha=0.85)
            sorted_pr = sorted(pr.items(), key=lambda x: -x[1])
            stats.top_pagerank = [
                {"id": nid, "label": self.graph.nodes[nid].get("label", nid),
                 "pagerank": score}
                for nid, score in sorted_pr[:top_k]
            ]
        except:
            pass
        
        # Betweenness (solo si el grafo no es muy grande)
        if self.num_nodes < 500:
            try:
                bc = nx.betweenness_centrality(self.graph)
                sorted_bc = sorted(bc.items(), key=lambda x: -x[1])
                stats.top_betweenness = [
                    {"id": nid, "label": self.graph.nodes[nid].get("label", nid),
                     "betweenness": score}
                    for nid, score in sorted_bc[:top_k]
                ]
            except:
                pass
    
    def _compute_ontology_coverage(self) -> Dict[str, float]:
        """Calcula cobertura de la ontología en el grafo."""
        graph_types = set()
        for nid in self.graph.nodes():
            ntype = self.graph.nodes[nid].get("node_type", "")
            if ntype:
                graph_types.add(ntype)
        
        graph_rels = set()
        for u, v, data in self.graph.edges(data=True):
            rel = data.get("relation", "")
            if rel:
                graph_rels.add(rel)
        
        onto_classes = set(self.ontology.classes.keys())
        onto_rels = set(self.ontology.relations.keys())
        
        return {
            "types_in_graph": len(graph_types),
            "types_in_ontology": len(onto_classes),
            "types_coverage": len(graph_types & onto_classes) / len(graph_types) * 100 
                             if graph_types else 100,
            "relations_in_graph": len(graph_rels),
            "relations_in_ontology": len(onto_rels),
            "relations_coverage": len(graph_rels & onto_rels) / len(graph_rels) * 100
                                 if graph_rels else 100,
        }
    
    # ═══════════════════════════════════════════════════════════════════════════
    # CONSULTAS DE GRAFO
    # ═══════════════════════════════════════════════════════════════════════════
    
    def get_neighbors(self, node_id: str, direction: str = "both") -> List[str]:
        """Obtiene vecinos de un nodo."""
        neighbors = set()
        if direction in ("out", "both"):
            neighbors.update(self.graph.successors(node_id))
        if direction in ("in", "both"):
            neighbors.update(self.graph.predecessors(node_id))
        return list(neighbors)
    
    def find_paths(self, source: str, target: str, max_length: int = 4) -> List[List[str]]:
        """Encuentra caminos entre dos nodos."""
        try:
            paths = list(nx.all_simple_paths(self.graph, source, target, cutoff=max_length))
            return paths[:10]
        except:
            return []
    
    def get_subgraph(self, node_ids: List[str]) -> nx.MultiDiGraph:
        """Extrae subgrafo con los nodos dados."""
        return self.graph.subgraph(node_ids).copy()
    
    # ═══════════════════════════════════════════════════════════════════════════
    # PERSISTENCIA
    # ═══════════════════════════════════════════════════════════════════════════
    
    def save(self, path: str = None):
        """Guarda el grafo completo."""
        path = Path(path or self.persist_path or f"{self.name}_data")
        path.mkdir(parents=True, exist_ok=True)
        
        self._log(f"💾 Guardando grafo en {path}...")
        
        # Guardar NetworkX
        with open(path / "graph.pickle", "wb") as f:
            pickle.dump(self.graph, f)
        
        # Metadata
        meta = {
            "name": self.name,
            "created_at": self.created_at,
            "embedding_model": self.embedding_model_name,
            "num_nodes": self.num_nodes,
            "num_edges": self.num_edges,
        }
        with open(path / "metadata.json", "w") as f:
            json.dump(meta, f, indent=2)
        
        # Ontología
        if self.ontology:
            self.ontology.save(str(path / "ontology.json"))
        
        self._log(f"   ✓ Guardado completado")
    
    @classmethod
    def load(cls, path: str, verbose: bool = True) -> "KnowledgeGraph":
        """Carga un grafo desde disco."""
        path = Path(path)
        
        with open(path / "metadata.json") as f:
            meta = json.load(f)
        
        ontology = None
        if (path / "ontology.json").exists():
            ontology = Ontology.load(str(path / "ontology.json"))
        
        kg = cls(
            ontology=ontology,
            name=meta.get("name", "loaded_graph"),
            embedding_model=meta.get("embedding_model"),
            persist_path=str(path),
            verbose=verbose
        )
        
        with open(path / "graph.pickle", "rb") as f:
            kg.graph = pickle.load(f)
        
        kg.created_at = meta.get("created_at", kg.created_at)
        
        if verbose:
            print(f"✓ Cargado: {kg.num_nodes} nodos, {kg.num_edges} aristas")
        
        return kg
    
    def __repr__(self):
        return f"KnowledgeGraph('{self.name}', nodes={self.num_nodes}, edges={self.num_edges})"


# ═══════════════════════════════════════════════════════════════════════════════
# GRAPH BUILDER (Patrón Builder)
# ═══════════════════════════════════════════════════════════════════════════════

class GraphBuilder:
    """
    Builder para construcción fluida de grafos.
    
    Ejemplo:
        >>> kg = (GraphBuilder("my_graph")
        ...     .with_ontology(onto)
        ...     .add_node("n1", "Nodo 1", "type_a")
        ...     .add_edge("n1", "n2", "RELATES_TO")
        ...     .build())
    """
    
    def __init__(self, name: str = "graph"):
        self.name = name
        self.ontology = None
        self.nodes: List[Node] = []
        self.edges: List[Edge] = []
        self.verbose = True
    
    def with_ontology(self, onto: Ontology) -> "GraphBuilder":
        self.ontology = onto
        return self
    
    def with_verbose(self, v: bool) -> "GraphBuilder":
        self.verbose = v
        return self
    
    def add_node(self, id: str, label: str, node_type: str, **kwargs) -> "GraphBuilder":
        self.nodes.append(Node(id=id, label=label, node_type=node_type, **kwargs))
        return self
    
    def add_edge(self, source: str, target: str, relation: str, **kwargs) -> "GraphBuilder":
        self.edges.append(Edge(source=source, target=target, relation=relation, **kwargs))
        return self
    
    def build(self) -> KnowledgeGraph:
        kg = KnowledgeGraph(ontology=self.ontology, name=self.name, verbose=self.verbose)
        if self.nodes:
            kg.add_nodes(self.nodes)
        if self.edges:
            kg.add_edges(self.edges)
        return kg
