"""
ontorag.core - Componentes core de la biblioteca
"""

from dataclasses import dataclass
from typing import Dict, Any, Optional
from enum import Enum


class LogLevel(Enum):
    """Niveles de logging."""
    DEBUG = 0
    INFO = 1
    WARNING = 2
    ERROR = 3
    SILENT = 4


@dataclass
class OntoRAGConfig:
    """
    Configuración global de OntoRAG.
    
    Example:
        >>> from ontorag.core import OntoRAGConfig
        >>> config = OntoRAGConfig(
        ...     embedding_model="sentence-transformers/all-MiniLM-L6-v2",
        ...     log_level=LogLevel.INFO
        ... )
    """
    # Modelo de embeddings por defecto
    embedding_model: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    
    # Dimensiones del embedding (se auto-detecta)
    embedding_dimension: Optional[int] = None
    
    # Configuración de búsqueda
    default_k_nodes: int = 10
    default_k_edges: int = 15
    default_k_docs: int = 5
    default_semantic_weight: float = 0.6
    default_structural_weight: float = 0.4
    default_max_depth: int = 2
    
    # Logging
    log_level: LogLevel = LogLevel.INFO
    
    # Idioma por defecto para respuestas
    default_language: str = "español"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convierte a diccionario."""
        return {
            "embedding_model": self.embedding_model,
            "embedding_dimension": self.embedding_dimension,
            "default_k_nodes": self.default_k_nodes,
            "default_k_edges": self.default_k_edges,
            "default_k_docs": self.default_k_docs,
            "default_semantic_weight": self.default_semantic_weight,
            "default_structural_weight": self.default_structural_weight,
            "default_max_depth": self.default_max_depth,
            "log_level": self.log_level.name,
            "default_language": self.default_language,
        }


# Configuración global singleton
_global_config: Optional[OntoRAGConfig] = None


def get_config() -> OntoRAGConfig:
    """Obtiene la configuración global."""
    global _global_config
    if _global_config is None:
        _global_config = OntoRAGConfig()
    return _global_config


def set_config(config: OntoRAGConfig):
    """Establece la configuración global."""
    global _global_config
    _global_config = config
