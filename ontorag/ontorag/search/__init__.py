"""
╔═══════════════════════════════════════════════════════════════════════════════╗
║                         ONTORAG - MÓDULO SEARCH                               ║
║                                                                               ║
║  Búsqueda HÍBRIDA: Semántica + Estructural                                   ║
║                                                                               ║
║  ¿Por qué híbrida?                                                           ║
║  - Semántica: Encuentra "calor urbano" → "urban heat island"                 ║
║  - Estructural: Encuentra green_roof → MITIGATES → heat_wave                 ║
║  - Combinación: Lo mejor de ambos mundos                                     ║
╚═══════════════════════════════════════════════════════════════════════════════╝
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional, TYPE_CHECKING
from enum import Enum

if TYPE_CHECKING:
    from ontorag.graph import KnowledgeGraph


class SearchSource(Enum):
    SEMANTIC = "semantic"
    STRUCTURAL = "structural"
    HYBRID = "hybrid"


class ItemType(Enum):
    NODE = "node"
    EDGE = "edge"


@dataclass
class SearchResult:
    """Resultado de búsqueda unificado."""
    item_id: str
    item_type: ItemType
    score: float
    source: SearchSource
    data: Dict = field(default_factory=dict)
    explanation: str = ""
    
    def __post_init__(self):
        # Asegurar que el score esté en [0, 1]
        self.score = max(0.0, min(1.0, float(self.score)))
    
    def to_dict(self) -> Dict:
        return {
            "item_id": self.item_id,
            "item_type": self.item_type.value,
            "score": self.score,
            "source": self.source.value,
            "data": self.data,
            "explanation": self.explanation
        }


class HybridSearcher:
    """
    Buscador híbrido: semántico + estructural.
    
    Proceso:
    1. Búsqueda semántica con embeddings
    2. Usa resultados como semillas para exploración estructural
    3. Combina scores con pesos configurables
    
    Ejemplo:
        >>> searcher = HybridSearcher(kg)
        >>> results = searcher.search(
        ...     "medidas contra inundaciones",
        ...     semantic_weight=0.6,
        ...     structural_weight=0.4
        ... )
    """
    
    def __init__(self, knowledge_graph: "KnowledgeGraph"):
        self.kg = knowledge_graph
    
    def search_semantic(
        self,
        query: str,
        k: int = 10,
        node_types: List[str] = None,
        min_score: float = 0.0
    ) -> List[SearchResult]:
        """Búsqueda semántica con embeddings."""
        results = []
        
        query_embedding = self.kg._get_embedding(query)
        
        # Buscar nodos
        try:
            where = {"type": {"$in": node_types}} if node_types else None
            node_results = self.kg._nodes_collection.query(
                query_embeddings=[query_embedding],
                n_results=k,
                where=where
            )
            
            if node_results["ids"] and node_results["ids"][0]:
                for i, nid in enumerate(node_results["ids"][0]):
                    dist = node_results["distances"][0][i] if node_results["distances"] else 0
                    score = max(0, 1.0 - dist)
                    
                    if score >= min_score:
                        node_data = self.kg.get_node(nid) or {}
                        results.append(SearchResult(
                            item_id=nid,
                            item_type=ItemType.NODE,
                            score=score,
                            source=SearchSource.SEMANTIC,
                            data=node_data,
                            explanation=f"Similitud semántica: {score:.1%}"
                        ))
        except Exception as e:
            pass
        
        # Buscar aristas
        try:
            edge_results = self.kg._edges_collection.query(
                query_embeddings=[query_embedding],
                n_results=k
            )
            
            if edge_results["ids"] and edge_results["ids"][0]:
                for i, eid in enumerate(edge_results["ids"][0]):
                    dist = edge_results["distances"][0][i] if edge_results["distances"] else 0
                    score = max(0, 1.0 - dist)
                    
                    if score >= min_score:
                        meta = edge_results["metadatas"][0][i] if edge_results["metadatas"] else {}
                        results.append(SearchResult(
                            item_id=eid,
                            item_type=ItemType.EDGE,
                            score=score,
                            source=SearchSource.SEMANTIC,
                            data=meta,
                            explanation=f"Relación relevante: {score:.1%}"
                        ))
        except:
            pass
        
        results.sort(key=lambda x: x.score, reverse=True)
        return results
    
    def search_structural(
        self,
        seed_nodes: List[str],
        max_depth: int = 2,
        max_neighbors: int = 10,
        decay: float = 0.7
    ) -> List[SearchResult]:
        """Búsqueda estructural desde nodos semilla."""
        results = []
        visited = set()
        
        def explore(node_id: str, depth: int, base_score: float):
            if depth > max_depth or node_id in visited:
                return
            visited.add(node_id)
            
            node_data = self.kg.get_node(node_id)
            if node_data:
                results.append(SearchResult(
                    item_id=node_id,
                    item_type=ItemType.NODE,
                    score=base_score,
                    source=SearchSource.STRUCTURAL,
                    data=node_data,
                    explanation=f"Conexión a profundidad {depth}"
                ))
            
            # Explorar vecinos
            neighbors = 0
            for _, neighbor, data in self.kg.graph.out_edges(node_id, data=True):
                if neighbors >= max_neighbors:
                    break
                neighbors += 1
                
                # Agregar arista
                results.append(SearchResult(
                    item_id=data.get("id", f"{node_id}_{neighbor}"),
                    item_type=ItemType.EDGE,
                    score=base_score * 0.9,
                    source=SearchSource.STRUCTURAL,
                    data={"source": node_id, "target": neighbor, 
                          "relation": data.get("relation", "")},
                    explanation=f"Relación desde {node_id}"
                ))
                
                explore(neighbor, depth + 1, base_score * decay)
            
            for predecessor, _, data in self.kg.graph.in_edges(node_id, data=True):
                if neighbors >= max_neighbors:
                    break
                neighbors += 1
                explore(predecessor, depth + 1, base_score * decay)
        
        for seed in seed_nodes:
            if seed in self.kg.graph:
                explore(seed, 0, 1.0)
        
        return results
    
    def search(
        self,
        query: str,
        k: int = 20,
        semantic_weight: float = 0.6,
        structural_weight: float = 0.4,
        max_depth: int = 2,
        num_seeds: int = 5
    ) -> List[SearchResult]:
        """
        Búsqueda híbrida combinada.
        
        Args:
            query: Texto de consulta
            k: Número de resultados
            semantic_weight: Peso semántico (0-1)
            structural_weight: Peso estructural (0-1)
            max_depth: Profundidad de exploración
            num_seeds: Nodos semilla para estructural
        """
        # Normalizar pesos
        total = semantic_weight + structural_weight
        if total > 0:
            semantic_weight /= total
            structural_weight /= total
        
        # Semántica
        semantic_results = self.search_semantic(query, k=k*2)
        
        # Semillas para estructural
        seeds = [r.item_id for r in semantic_results 
                 if r.item_type == ItemType.NODE][:num_seeds]
        
        # Estructural
        structural_results = []
        if seeds and structural_weight > 0:
            structural_results = self.search_structural(seeds, max_depth)
        
        # Combinar
        combined: Dict[str, float] = {}
        all_results: Dict[str, SearchResult] = {}
        
        for r in semantic_results:
            key = f"{r.item_type.value}:{r.item_id}"
            combined[key] = r.score * semantic_weight
            all_results[key] = r
        
        for r in structural_results:
            key = f"{r.item_type.value}:{r.item_id}"
            if key in combined:
                combined[key] += r.score * structural_weight
            else:
                combined[key] = r.score * structural_weight
                all_results[key] = r
        
        # Ordenar y retornar
        sorted_keys = sorted(combined.keys(), key=lambda x: combined[x], reverse=True)
        
        final = []
        for key in sorted_keys[:k]:
            result = all_results[key]
            result.score = combined[key]
            result.source = SearchSource.HYBRID
            final.append(result)
        
        return final
