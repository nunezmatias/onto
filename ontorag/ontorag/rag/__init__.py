"""
ontorag.rag - Sistema RAG sobre Grafos de Conocimiento

Proporciona:
- Búsqueda híbrida (semántica + estructural)
- Construcción de contexto inteligente
- Generación de respuestas con LLM
- Trazabilidad y citación de fuentes
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Callable
from datetime import datetime
import json

from ontorag.graph import KnowledgeGraph
from ontorag.search import SearchResult


@dataclass
class RAGContext:
    """
    Contexto construido para el LLM.
    
    Contiene:
    - Nodos relevantes encontrados
    - Aristas/relaciones relevantes
    - Documentos fuente
    - Metadatos de la búsqueda
    """
    nodes: List[Dict[str, Any]] = field(default_factory=list)
    edges: List[Dict[str, Any]] = field(default_factory=list)
    documents: List[Dict[str, Any]] = field(default_factory=list)
    search_results: List[SearchResult] = field(default_factory=list)
    
    def to_text(self, max_length: int = 4000) -> str:
        """
        Convierte el contexto a texto para el prompt del LLM.
        
        Args:
            max_length: Longitud máxima del contexto
            
        Returns:
            Texto formateado del contexto
        """
        sections = []
        
        # Sección de entidades/nodos
        if self.nodes:
            node_lines = ["## Entidades Relevantes"]
            for node in self.nodes[:10]:  # Limitar a 10 nodos
                node_lines.append(
                    f"- **{node.get('name', node.get('id'))}** ({node.get('type', 'unknown')})"
                )
                if node.get('text_content'):
                    node_lines.append(f"  {node['text_content'][:200]}...")
            sections.append("\n".join(node_lines))
        
        # Sección de relaciones
        if self.edges:
            edge_lines = ["## Relaciones Encontradas"]
            for edge in self.edges[:15]:  # Limitar a 15 aristas
                edge_lines.append(
                    f"- {edge.get('source')} **{edge.get('relation')}** {edge.get('target')}"
                )
                if edge.get('text_context'):
                    edge_lines.append(f"  Contexto: {edge['text_context'][:150]}...")
            sections.append("\n".join(edge_lines))
        
        # Sección de documentos fuente
        if self.documents:
            doc_lines = ["## Fuentes Documentales"]
            for doc in self.documents[:5]:
                doc_lines.append(f"- [{doc.get('id', 'Doc')}]: {doc.get('text', '')[:200]}...")
            sections.append("\n".join(doc_lines))
        
        context_text = "\n\n".join(sections)
        
        # Truncar si es necesario
        if len(context_text) > max_length:
            context_text = context_text[:max_length] + "\n\n[Contexto truncado...]"
        
        return context_text
    
    def get_source_citations(self) -> List[str]:
        """Obtiene lista de citaciones de fuentes."""
        citations = set()
        
        for node in self.nodes:
            for doc_id in node.get('source_docs', []):
                citations.add(doc_id)
        
        for edge in self.edges:
            for doc_id in edge.get('source_docs', []):
                citations.add(doc_id)
        
        return list(citations)


@dataclass
class RAGResponse:
    """
    Respuesta completa del sistema RAG.
    
    Incluye:
    - La respuesta generada
    - El contexto usado
    - Metadatos de la consulta
    - Fuentes citadas
    """
    query: str
    answer: str
    context: RAGContext
    citations: List[str]
    prompt_used: str = ""
    system_prompt: str = ""
    user_prompt: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    
    def to_dict(self) -> Dict[str, Any]:
        """Convierte a diccionario para serialización."""
        return {
            "query": self.query,
            "answer": self.answer,
            "citations": self.citations,
            "metadata": self.metadata,
            "timestamp": self.timestamp,
            "system_prompt": self.system_prompt,
            "user_prompt": self.user_prompt,
            "prompt_used": self.prompt_used,
            "context": {
                "num_nodes": len(self.context.nodes),
                "num_edges": len(self.context.edges),
                "num_documents": len(self.context.documents)
            }
        }


# Type alias para función generadora de LLM
LLMFunction = Callable[[str, str], str]


def create_openai_llm(
    api_key: str = None,
    model: str = "gpt-4o-mini",
    temperature: float = 0.1
) -> LLMFunction:
    """
    Crea una función LLM usando OpenAI.
    
    Args:
        api_key: API key de OpenAI (o usa OPENAI_API_KEY)
        model: Modelo a usar
        temperature: Temperatura para generación
        
    Returns:
        Función que toma (system_prompt, user_prompt) y retorna respuesta
    """
    try:
        from openai import OpenAI
    except ImportError:
        raise ImportError("Instala openai: pip install openai")
    
    import os
    api_key = api_key or os.getenv("OPENAI_API_KEY")
    client = OpenAI(api_key=api_key)
    
    def generate(system_prompt: str, user_prompt: str) -> str:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=temperature
        )
        return response.choices[0].message.content
    
    return generate


def create_anthropic_llm(
    api_key: str = None,
    model: str = "claude-3-haiku-20240307",
    temperature: float = 0.1
) -> LLMFunction:
    """
    Crea una función LLM usando Anthropic Claude.
    
    Args:
        api_key: API key de Anthropic (o usa ANTHROPIC_API_KEY)
        model: Modelo a usar
        temperature: Temperatura para generación
        
    Returns:
        Función que toma (system_prompt, user_prompt) y retorna respuesta
    """
    try:
        import anthropic
    except ImportError:
        raise ImportError("Instala anthropic: pip install anthropic")
    
    import os
    api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
    client = anthropic.Anthropic(api_key=api_key)
    
    def generate(system_prompt: str, user_prompt: str) -> str:
        response = client.messages.create(
            model=model,
            max_tokens=1024,
            system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}],
            temperature=temperature
        )
        return response.content[0].text
    
    return generate


def create_mock_llm() -> LLMFunction:
    """
    Crea un LLM mock para pruebas (no requiere API key).
    
    Útil para desarrollo y testing sin costos de API.
    """
    def generate(system_prompt: str, user_prompt: str) -> str:
        return f"""[Respuesta Mock]

Basándome en el contexto proporcionado, aquí está mi análisis:

El contexto incluye información relevante sobre la consulta. 
Para una respuesta real, configura un LLM con:
- create_openai_llm()
- create_anthropic_llm()

Consulta recibida: {user_prompt[:200]}...
"""
    
    return generate


class RAGEngine:
    """
    Motor RAG para consultas sobre el grafo de conocimiento.
    
    Combina:
    - Búsqueda híbrida (semántica + estructural)
    - Construcción inteligente de contexto
    - Generación de respuestas con LLM
    - Trazabilidad de fuentes
    
    Example:
        >>> from ontorag import KnowledgeGraph, RAGEngine
        >>> from ontorag.rag import create_openai_llm
        >>> 
        >>> kg = KnowledgeGraph.load("my_graph/")
        >>> llm = create_openai_llm(api_key="sk-...")
        >>> 
        >>> rag = RAGEngine(kg, llm_function=llm)
        >>> response = rag.query("¿Qué medidas mitigan las inundaciones?")
        >>> print(response.answer)
    """
    
    DEFAULT_SYSTEM_PROMPT = """Eres un asistente experto que responde preguntas basándose en un grafo de conocimiento.

INSTRUCCIONES:
1. Basa tu respuesta EXCLUSIVAMENTE en el contexto proporcionado
2. Si el contexto no contiene información suficiente, indícalo claramente
3. Cita las fuentes cuando sea posible
4. Organiza la respuesta de forma clara y estructurada
5. Responde en el mismo idioma que la pregunta

FORMATO DE RESPUESTA:
- Comienza con una respuesta directa a la pregunta
- Luego proporciona detalles y explicaciones
- Finaliza mencionando las fuentes si las hay
"""
    
    def __init__(
        self,
        knowledge_graph: KnowledgeGraph,
        llm_function: LLMFunction = None,
        system_prompt: str = None,
        verbose: bool = True
    ):
        """
        Inicializa el motor RAG.
        
        Args:
            knowledge_graph: Grafo de conocimiento para búsqueda
            llm_function: Función para generar respuestas (default: mock)
            system_prompt: Prompt de sistema personalizado
            verbose: Mostrar información de depuración
        """
        self.kg = knowledge_graph
        self.llm = llm_function or create_mock_llm()
        self.system_prompt = system_prompt or self.DEFAULT_SYSTEM_PROMPT
        self.verbose = verbose
        self.query_history: List[RAGResponse] = []
    
    def _log(self, message: str):
        """Imprime mensaje si verbose está activo."""
        if self.verbose:
            print(message)
    
    def build_context(
        self,
        query: str,
        k_nodes: int = 10,
        k_edges: int = 15,
        k_docs: int = 5,
        semantic_weight: float = 0.6,
        structural_weight: float = 0.4,
        max_depth: int = 2
    ) -> RAGContext:
        """
        Construye el contexto para una consulta.
        
        Args:
            query: La pregunta del usuario
            k_nodes: Número máximo de nodos
            k_edges: Número máximo de aristas
            k_docs: Número máximo de documentos
            semantic_weight: Peso búsqueda semántica
            structural_weight: Peso búsqueda estructural
            max_depth: Profundidad de exploración
            
        Returns:
            RAGContext con la información recuperada
        """
        self._log(f"🔍 Buscando contexto para: '{query[:50]}...'")
        
        # Búsqueda híbrida
        search_results = self.kg.search_hybrid(
            query=query,
            k=k_nodes + k_edges + k_docs,
            semantic_weight=semantic_weight,
            structural_weight=structural_weight,
            max_depth=max_depth
        )
        
        self._log(f"   📊 {len(search_results)} resultados encontrados")
        
        # Separar resultados por tipo
        context = RAGContext(search_results=search_results)
        
        for result in search_results:
            if result.item_type == "node" and len(context.nodes) < k_nodes:
                context.nodes.append(result.data)
            elif result.item_type == "edge" and len(context.edges) < k_edges:
                context.edges.append(result.data)
            elif result.item_type == "document" and len(context.documents) < k_docs:
                context.documents.append({"id": result.item_id, **result.data})
        
        self._log(f"   ✅ Contexto: {len(context.nodes)} nodos, {len(context.edges)} aristas")
        
        return context
    
    def generate_response(
        self,
        query: str,
        context: RAGContext,
        response_language: str = "español"
    ) -> str:
        """
        Genera una respuesta usando el LLM.
        
        Args:
            query: La pregunta del usuario
            context: Contexto construido
            response_language: Idioma de la respuesta
            
        Returns:
            Texto de la respuesta generada
        """
        # Construir prompt de usuario
        context_text = context.to_text()
        
        user_prompt = f"""## CONTEXTO DEL GRAFO DE CONOCIMIENTO

{context_text}

---

## PREGUNTA DEL USUARIO

{query}

---

Por favor, responde en {response_language} basándote únicamente en el contexto proporcionado.
"""
        
        self._log("🤖 Generando respuesta...")
        
        # Llamar al LLM
        response = self.llm(self.system_prompt, user_prompt)
        
        self._log("   ✅ Respuesta generada")
        
        return response
    
    def query(
        self,
        question: str,
        k_nodes: int = 10,
        k_edges: int = 15,
        k_docs: int = 5,
        semantic_weight: float = 0.6,
        structural_weight: float = 0.4,
        max_depth: int = 2,
        response_language: str = "español",
        save_to_history: bool = True
    ) -> RAGResponse:
        """
        Ejecuta una consulta RAG completa.
        
        Este es el método principal que:
        1. Busca información relevante en el grafo
        2. Construye el contexto
        3. Genera la respuesta con el LLM
        4. Retorna una respuesta completa con citaciones
        
        Args:
            question: La pregunta del usuario
            k_nodes: Número máximo de nodos a recuperar
            k_edges: Número máximo de aristas a recuperar
            k_docs: Número máximo de documentos a recuperar
            semantic_weight: Peso de la búsqueda semántica (0-1)
            structural_weight: Peso de la búsqueda estructural (0-1)
            max_depth: Profundidad máxima de exploración del grafo
            response_language: Idioma para la respuesta
            save_to_history: Si guardar en el historial
            
        Returns:
            RAGResponse con la respuesta completa
            
        Example:
            >>> response = rag.query("¿Cómo mitigan los techos verdes el calor urbano?")
            >>> print(response.answer)
            >>> print(f"Fuentes: {response.citations}")
        """
        import time
        start_time = time.time()
        
        self._log(f"\n{'='*60}")
        self._log(f"📝 CONSULTA: {question}")
        self._log(f"{'='*60}")
        
        # Paso 1: Construir contexto
        context = self.build_context(
            query=question,
            k_nodes=k_nodes,
            k_edges=k_edges,
            k_docs=k_docs,
            semantic_weight=semantic_weight,
            structural_weight=structural_weight,
            max_depth=max_depth
        )
        
        # Paso 2: Generar respuesta
        context_text = context.to_text()
        user_prompt = f"""## CONTEXTO DEL GRAFO DE CONOCIMIENTO

{context_text}

---

## PREGUNTA DEL USUARIO

{question}

---

Por favor, responde en {response_language} basándote únicamente en el contexto proporcionado.
"""
        
        answer = self.generate_response(
            query=question,
            context=context,
            response_language=response_language
        )
        
        # Paso 3: Construir respuesta
        elapsed_time = time.time() - start_time
        
        response = RAGResponse(
            query=question,
            answer=answer,
            context=context,
            system_prompt=self.system_prompt,
            user_prompt=user_prompt,
            prompt_used=user_prompt,
            citations=context.get_source_citations(),
            metadata={
                "elapsed_time": elapsed_time,
                "k_nodes": k_nodes,
                "k_edges": k_edges,
                "k_docs": k_docs,
                "semantic_weight": semantic_weight,
                "structural_weight": structural_weight,
                "max_depth": max_depth,
                "num_search_results": len(context.search_results)
            }
        )
        
        # Guardar en historial
        if save_to_history:
            self.query_history.append(response)
        
        self._log(f"\n⏱️ Tiempo total: {elapsed_time:.2f}s")
        self._log(f"{'='*60}\n")
        
        return response
    
    def query_with_ontology_hints(
        self,
        question: str,
        focus_types: List[str] = None,
        focus_relations: List[str] = None,
        **kwargs
    ) -> RAGResponse:
        """
        Consulta con hints ontológicos para mejor precisión.
        
        Útil cuando sabes qué tipos de entidades o relaciones
        son más relevantes para la pregunta.
        
        Args:
            question: La pregunta
            focus_types: Tipos de nodo a priorizar
            focus_relations: Tipos de relación a priorizar
            **kwargs: Argumentos adicionales para query()
            
        Returns:
            RAGResponse
        """
        # Modificar el query para incluir hints ontológicos
        enhanced_query = question
        
        if self.kg.ontology and focus_types:
            # Agregar sinónimos de los tipos al query
            for type_id in focus_types:
                cls = self.kg.ontology.get_class(type_id)
                if cls:
                    enhanced_query += f" {cls.name}"
        
        return self.query(enhanced_query, **kwargs)
    
    def explain_retrieval(
        self,
        question: str,
        k: int = 10
    ) -> str:
        """
        Explica qué información se recuperaría para una pregunta.
        
        Útil para depuración y comprensión del sistema.
        
        Args:
            question: La pregunta a analizar
            k: Número de resultados a mostrar
            
        Returns:
            Explicación textual del proceso de retrieval
        """
        context = self.build_context(question, k_nodes=k, k_edges=k, k_docs=k//2)
        
        lines = [
            f"🔍 ANÁLISIS DE RETRIEVAL",
            f"="*50,
            f"Pregunta: {question}",
            f"",
            f"📊 RESULTADOS DE BÚSQUEDA:",
            f""
        ]
        
        for i, result in enumerate(context.search_results[:k], 1):
            lines.append(f"{i}. [{result.item_type.upper()}] {result.item_id}")
            lines.append(f"   Score: {result.score:.4f} | Fuente: {result.source}")
            if result.item_type == "node":
                lines.append(f"   Tipo: {result.data.get('type')}")
            elif result.item_type == "edge":
                lines.append(f"   Relación: {result.data.get('source')} -> {result.data.get('target')}")
            lines.append("")
        
        lines.extend([
            f"📝 CONTEXTO CONSTRUIDO:",
            f"   • Nodos: {len(context.nodes)}",
            f"   • Aristas: {len(context.edges)}",
            f"   • Documentos: {len(context.documents)}",
            f"   • Citaciones: {len(context.get_source_citations())}",
        ])
        
        return "\n".join(lines)
    
    def export_history(self, path: str):
        """Exporta el historial de consultas a JSON."""
        history_data = [r.to_dict() for r in self.query_history]
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(history_data, f, indent=2, ensure_ascii=False)
    
    def __repr__(self) -> str:
        return f"RAGEngine(kg={self.kg}, queries={len(self.query_history)})"
