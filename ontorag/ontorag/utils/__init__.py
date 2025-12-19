"""
ontorag.utils - Utilidades y funciones auxiliares
"""

import re
import hashlib
from typing import List, Dict, Any
from pathlib import Path
import json


def normalize_text(text: str) -> str:
    """
    Normaliza texto para comparación.
    
    - Convierte a minúsculas
    - Elimina caracteres especiales
    - Normaliza espacios
    """
    text = text.lower().strip()
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text


def text_to_id(text: str, max_length: int = 50) -> str:
    """
    Convierte texto a un ID válido.
    
    Args:
        text: Texto a convertir
        max_length: Longitud máxima del ID
        
    Returns:
        ID normalizado
    """
    text = normalize_text(text)
    text = text.replace(' ', '_')
    return text[:max_length]


def hash_text(text: str) -> str:
    """Genera hash MD5 de un texto."""
    return hashlib.md5(text.encode()).hexdigest()


def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 100) -> List[str]:
    """
    Divide texto en chunks con overlap.
    
    Args:
        text: Texto a dividir
        chunk_size: Tamaño de cada chunk
        overlap: Solapamiento entre chunks
        
    Returns:
        Lista de chunks
    """
    if len(text) <= chunk_size:
        return [text]
    
    chunks = []
    start = 0
    
    while start < len(text):
        end = start + chunk_size
        
        # Intentar cortar en un espacio
        if end < len(text):
            space_pos = text.rfind(' ', start, end)
            if space_pos > start:
                end = space_pos
        
        chunks.append(text[start:end])
        start = end - overlap
    
    return chunks


def merge_dicts(base: Dict, override: Dict) -> Dict:
    """Combina dos diccionarios recursivamente."""
    result = base.copy()
    
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = merge_dicts(result[key], value)
        else:
            result[key] = value
    
    return result


def load_json(path: str) -> Dict[str, Any]:
    """Carga un archivo JSON."""
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def save_json(data: Any, path: str, indent: int = 2):
    """Guarda datos en un archivo JSON."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=indent, ensure_ascii=False)


class ProgressBar:
    """
    Barra de progreso simple para loops.
    
    Example:
        >>> for item in ProgressBar(items, desc="Procesando"):
        ...     process(item)
    """
    
    def __init__(self, iterable, desc: str = "", total: int = None):
        self.iterable = iterable
        self.desc = desc
        self.total = total or len(iterable)
        self.current = 0
    
    def __iter__(self):
        for item in self.iterable:
            self.current += 1
            self._update()
            yield item
        print()  # Nueva línea al final
    
    def _update(self):
        pct = self.current / self.total * 100
        bar_len = 30
        filled = int(bar_len * self.current / self.total)
        bar = '█' * filled + '░' * (bar_len - filled)
        print(f'\r{self.desc}: |{bar}| {pct:.1f}% ({self.current}/{self.total})', end='', flush=True)
