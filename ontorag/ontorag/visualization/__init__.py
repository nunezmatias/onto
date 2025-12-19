"""
═══════════════════════════════════════════════════════════════════════════════
MÓDULO: ontorag.visualization
═══════════════════════════════════════════════════════════════════════════════

Visualización interactiva de grafos de conocimiento y ontologías.

Componentes:
- GraphVisualizer: Genera visualizaciones HTML interactivas con D3.js/vis.js
- OntologyVisualizer: Visualiza jerarquías de ontología
- ReportGenerator: Genera reportes HTML completos

Características:
- Visualización interactiva (zoom, pan, drag)
- Coloreado por tipo de nodo
- Filtros por relación
- Tooltips con información
- Export a PNG/SVG
- Dashboard con estadísticas
"""

from __future__ import annotations
from typing import Dict, List, Optional, Any, TYPE_CHECKING
from pathlib import Path
from datetime import datetime
import json
import html

if TYPE_CHECKING:
    from ontorag.graph import KnowledgeGraph
    from ontorag.ontology import Ontology
    from ontorag.analytics import GraphMetrics


# ═══════════════════════════════════════════════════════════════════════════════
# PALETA DE COLORES
# ═══════════════════════════════════════════════════════════════════════════════

DEFAULT_COLORS = [
    "#4E79A7",  # Azul
    "#F28E2B",  # Naranja
    "#E15759",  # Rojo
    "#76B7B2",  # Turquesa
    "#59A14F",  # Verde
    "#EDC948",  # Amarillo
    "#B07AA1",  # Púrpura
    "#FF9DA7",  # Rosa
    "#9C755F",  # Marrón
    "#BAB0AC",  # Gris
]

RELATION_COLORS = {
    "MITIGATES": "#59A14F",
    "PREVENTS": "#4E79A7",
    "CAUSES": "#E15759",
    "AFFECTS": "#F28E2B",
    "PROTECTS": "#76B7B2",
    "IMPLEMENTED_IN": "#9C755F",
    "LOCATED_IN": "#BAB0AC",
    "PART_OF": "#B07AA1",
}


# ═══════════════════════════════════════════════════════════════════════════════
# GRAPH VISUALIZER
# ═══════════════════════════════════════════════════════════════════════════════

class GraphVisualizer:
    """
    Genera visualizaciones HTML interactivas del grafo de conocimiento.
    
    Usa vis.js para crear grafos interactivos con:
    - Nodos coloreados por tipo
    - Aristas etiquetadas con relación
    - Zoom, pan, y drag
    - Panel de información
    - Filtros interactivos
    
    Example:
        >>> viz = GraphVisualizer(knowledge_graph)
        >>> viz.generate_html("graph.html")
        >>> viz.generate_dashboard("dashboard.html")
    """
    
    def __init__(
        self,
        knowledge_graph: "KnowledgeGraph",
        colors: List[str] = None,
    ):
        self.kg = knowledge_graph
        self.colors = colors or DEFAULT_COLORS
        self._type_colors: Dict[str, str] = {}
    
    def _get_color_for_type(self, node_type: str) -> str:
        """Asigna color consistente a cada tipo."""
        if node_type not in self._type_colors:
            idx = len(self._type_colors) % len(self.colors)
            self._type_colors[node_type] = self.colors[idx]
        return self._type_colors[node_type]
    
    def _prepare_graph_data(self, max_nodes: int = 500) -> Dict[str, Any]:
        """Prepara datos del grafo para visualización."""
        nodes = []
        edges = []
        
        # Limitar nodos si es necesario
        node_ids = list(self.kg.graph.nodes())[:max_nodes]
        node_set = set(node_ids)
        
        for node_id in node_ids:
            node_data = self.kg.graph.nodes[node_id]
            node_type = node_data.get("type", "unknown")
            
            nodes.append({
                "id": node_id,
                "label": node_data.get("name", node_id)[:30],
                "title": f"<b>{node_data.get('name', node_id)}</b><br>Tipo: {node_type}<br>{node_data.get('description', '')[:100]}",
                "color": self._get_color_for_type(node_type),
                "group": node_type,
            })
        
        edge_id = 0
        for source, target, data in self.kg.graph.edges(data=True):
            if source in node_set and target in node_set:
                relation = data.get("relation", "RELATED")
                edges.append({
                    "id": edge_id,
                    "from": source,
                    "to": target,
                    "label": relation,
                    "title": f"{relation}<br>{data.get('text_context', '')[:100]}",
                    "color": RELATION_COLORS.get(relation, "#999999"),
                    "arrows": "to",
                })
                edge_id += 1
        
        return {"nodes": nodes, "edges": edges}
    
    def generate_html(
        self,
        output_path: str,
        title: str = "Knowledge Graph Visualization",
        max_nodes: int = 500,
        width: str = "100%",
        height: str = "800px",
    ) -> str:
        """
        Genera archivo HTML con visualización interactiva.
        
        Args:
            output_path: Ruta del archivo de salida
            title: Título de la página
            max_nodes: Máximo de nodos a mostrar
            width: Ancho del canvas
            height: Alto del canvas
            
        Returns:
            Ruta del archivo generado
        """
        graph_data = self._prepare_graph_data(max_nodes)
        
        # Calcular estadísticas
        stats = self.kg.get_statistics()
        
        # Generar leyenda de tipos
        legend_items = []
        for node_type, color in self._type_colors.items():
            count = stats["nodes"]["by_type"].get(node_type, 0)
            legend_items.append(f'<div class="legend-item"><span class="legend-color" style="background:{color}"></span>{node_type} ({count})</div>')
        
        html_content = f'''<!DOCTYPE html>
<html lang="es">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{html.escape(title)}</title>
    <script src="https://unpkg.com/vis-network/standalone/umd/vis-network.min.js"></script>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{ 
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, sans-serif;
            background: #f5f5f5;
        }}
        .header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 20px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }}
        .header h1 {{ font-size: 24px; font-weight: 600; }}
        .header .stats {{
            display: flex;
            gap: 30px;
            margin-top: 10px;
            font-size: 14px;
            opacity: 0.9;
        }}
        .container {{
            display: flex;
            height: calc(100vh - 80px);
        }}
        .sidebar {{
            width: 280px;
            background: white;
            padding: 20px;
            overflow-y: auto;
            border-right: 1px solid #e0e0e0;
        }}
        .sidebar h3 {{
            font-size: 14px;
            text-transform: uppercase;
            color: #666;
            margin-bottom: 15px;
            letter-spacing: 0.5px;
        }}
        .legend-item {{
            display: flex;
            align-items: center;
            padding: 8px 0;
            font-size: 13px;
            border-bottom: 1px solid #f0f0f0;
        }}
        .legend-color {{
            width: 16px;
            height: 16px;
            border-radius: 4px;
            margin-right: 10px;
        }}
        .graph-container {{
            flex: 1;
            background: white;
        }}
        #graph {{
            width: 100%;
            height: 100%;
        }}
        .controls {{
            padding: 15px;
            border-bottom: 1px solid #e0e0e0;
        }}
        .controls input {{
            width: 100%;
            padding: 10px;
            border: 1px solid #ddd;
            border-radius: 6px;
            font-size: 14px;
        }}
        .controls input:focus {{
            outline: none;
            border-color: #667eea;
        }}
        .info-panel {{
            padding: 15px;
            background: #f9f9f9;
            border-radius: 8px;
            margin-top: 15px;
            font-size: 13px;
            display: none;
        }}
        .info-panel.active {{ display: block; }}
        .info-panel h4 {{ color: #333; margin-bottom: 10px; }}
        .info-panel p {{ color: #666; line-height: 1.5; }}
        .btn {{
            display: inline-block;
            padding: 8px 16px;
            background: #667eea;
            color: white;
            border: none;
            border-radius: 6px;
            cursor: pointer;
            font-size: 13px;
            margin-top: 10px;
        }}
        .btn:hover {{ background: #5a6fd6; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>🔷 {html.escape(title)}</h1>
        <div class="stats">
            <span>📊 {stats['nodes']['total']} nodos</span>
            <span>🔗 {stats['edges']['total']} aristas</span>
            <span>📁 {len(stats['nodes']['by_type'])} tipos</span>
            <span>⚡ {len(stats['edges']['by_relation'])} relaciones</span>
        </div>
    </div>
    
    <div class="container">
        <div class="sidebar">
            <div class="controls">
                <input type="text" id="searchInput" placeholder="🔍 Buscar nodo...">
            </div>
            
            <h3>Tipos de Nodos</h3>
            {''.join(legend_items)}
            
            <div id="nodeInfo" class="info-panel">
                <h4 id="nodeTitle">Selecciona un nodo</h4>
                <p id="nodeDetails"></p>
            </div>
            
            <button class="btn" onclick="resetView()">🔄 Restablecer Vista</button>
            <button class="btn" onclick="exportPNG()">📷 Exportar PNG</button>
        </div>
        
        <div class="graph-container">
            <div id="graph"></div>
        </div>
    </div>
    
    <script>
        // Datos del grafo
        const graphData = {json.dumps(graph_data)};
        
        // Crear dataset
        const nodes = new vis.DataSet(graphData.nodes);
        const edges = new vis.DataSet(graphData.edges);
        
        // Opciones del grafo
        const options = {{
            nodes: {{
                shape: 'dot',
                size: 20,
                font: {{ size: 12, color: '#333' }},
                borderWidth: 2,
                shadow: true,
            }},
            edges: {{
                font: {{ size: 10, align: 'middle' }},
                width: 1.5,
                smooth: {{ type: 'continuous' }},
            }},
            physics: {{
                solver: 'forceAtlas2Based',
                forceAtlas2Based: {{
                    gravitationalConstant: -50,
                    centralGravity: 0.01,
                    springLength: 100,
                    springConstant: 0.08,
                }},
                stabilization: {{ iterations: 150 }},
            }},
            interaction: {{
                hover: true,
                tooltipDelay: 200,
                hideEdgesOnDrag: true,
            }},
        }};
        
        // Crear red
        const container = document.getElementById('graph');
        const network = new vis.Network(container, {{ nodes, edges }}, options);
        
        // Buscar nodos
        document.getElementById('searchInput').addEventListener('input', function(e) {{
            const query = e.target.value.toLowerCase();
            if (query.length < 2) {{
                nodes.forEach(node => nodes.update({{ id: node.id, hidden: false }}));
                return;
            }}
            
            nodes.forEach(node => {{
                const match = node.label.toLowerCase().includes(query);
                nodes.update({{ id: node.id, hidden: !match, opacity: match ? 1 : 0.2 }});
            }});
        }});
        
        // Mostrar info del nodo seleccionado
        network.on('click', function(params) {{
            if (params.nodes.length > 0) {{
                const nodeId = params.nodes[0];
                const node = nodes.get(nodeId);
                
                document.getElementById('nodeInfo').classList.add('active');
                document.getElementById('nodeTitle').textContent = node.label;
                document.getElementById('nodeDetails').innerHTML = node.title;
            }}
        }});
        
        function resetView() {{
            network.fit();
            nodes.forEach(node => nodes.update({{ id: node.id, hidden: false }}));
            document.getElementById('searchInput').value = '';
        }}
        
        function exportPNG() {{
            const canvas = container.getElementsByTagName('canvas')[0];
            const link = document.createElement('a');
            link.download = 'knowledge_graph.png';
            link.href = canvas.toDataURL('image/png');
            link.click();
        }}
    </script>
</body>
</html>'''
        
        # Guardar archivo
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(html_content, encoding='utf-8')
        
        return str(output_path)
    
    def generate_dashboard(
        self,
        output_path: str,
        title: str = "OntoRAG Dashboard",
    ) -> str:
        """
        Genera dashboard HTML completo con estadísticas y visualización.
        
        Args:
            output_path: Ruta del archivo de salida
            title: Título del dashboard
            
        Returns:
            Ruta del archivo generado
        """
        stats = self.kg.get_statistics()
        graph_data = self._prepare_graph_data(max_nodes=300)
        
        # Preparar datos para gráficos
        node_type_data = json.dumps([
            {"type": t, "count": c} 
            for t, c in sorted(stats["nodes"]["by_type"].items(), key=lambda x: -x[1])[:10]
        ])
        
        relation_data = json.dumps([
            {"relation": r, "count": c}
            for r, c in sorted(stats["edges"]["by_relation"].items(), key=lambda x: -x[1])[:10]
        ])
        
        # Top nodos
        metrics = stats.get("graph_metrics", {})
        top_nodes = metrics.get("top_connected_nodes", [])
        
        top_nodes_html = ""
        for i, node in enumerate(top_nodes[:10], 1):
            top_nodes_html += f'''
            <tr>
                <td>{i}</td>
                <td>{html.escape(node.get('name', node.get('id', '?')))[:30]}</td>
                <td>{node.get('degree', 0)}</td>
            </tr>'''
        
        html_content = f'''<!DOCTYPE html>
<html lang="es">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{html.escape(title)}</title>
    <script src="https://unpkg.com/vis-network/standalone/umd/vis-network.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{ 
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: #0f0f23;
            color: #cccccc;
            min-height: 100vh;
        }}
        .header {{
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
            padding: 30px;
            border-bottom: 1px solid #333;
        }}
        .header h1 {{ 
            font-size: 28px; 
            color: #00d4ff;
            display: flex;
            align-items: center;
            gap: 15px;
        }}
        .header .subtitle {{ color: #888; margin-top: 8px; }}
        
        .metrics {{
            display: grid;
            grid-template-columns: repeat(4, 1fr);
            gap: 20px;
            padding: 20px 30px;
        }}
        .metric-card {{
            background: linear-gradient(145deg, #1a1a2e, #16213e);
            border-radius: 12px;
            padding: 25px;
            border: 1px solid #333;
        }}
        .metric-card .value {{
            font-size: 36px;
            font-weight: 700;
            color: #00d4ff;
        }}
        .metric-card .label {{
            font-size: 14px;
            color: #888;
            margin-top: 5px;
        }}
        
        .dashboard {{
            display: grid;
            grid-template-columns: 2fr 1fr;
            gap: 20px;
            padding: 0 30px 30px;
        }}
        .card {{
            background: #1a1a2e;
            border-radius: 12px;
            padding: 25px;
            border: 1px solid #333;
        }}
        .card h3 {{
            font-size: 16px;
            color: #00d4ff;
            margin-bottom: 20px;
            display: flex;
            align-items: center;
            gap: 10px;
        }}
        
        #miniGraph {{ height: 400px; }}
        
        .chart-container {{
            position: relative;
            height: 250px;
        }}
        
        table {{
            width: 100%;
            border-collapse: collapse;
        }}
        th, td {{
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #333;
        }}
        th {{ color: #888; font-weight: 500; }}
        tr:hover {{ background: rgba(0,212,255,0.05); }}
        
        .tag {{
            display: inline-block;
            padding: 4px 10px;
            background: rgba(0,212,255,0.1);
            color: #00d4ff;
            border-radius: 4px;
            font-size: 12px;
            margin: 2px;
        }}
        
        .ontology-info {{
            margin-top: 20px;
        }}
        .ontology-info h4 {{
            color: #fff;
            margin-bottom: 10px;
        }}
        
        @media (max-width: 1200px) {{
            .metrics {{ grid-template-columns: repeat(2, 1fr); }}
            .dashboard {{ grid-template-columns: 1fr; }}
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>
            <svg width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="#00d4ff" stroke-width="2">
                <circle cx="12" cy="12" r="10"/>
                <circle cx="12" cy="12" r="4"/>
                <line x1="12" y1="2" x2="12" y2="6"/>
                <line x1="12" y1="18" x2="12" y2="22"/>
                <line x1="2" y1="12" x2="6" y2="12"/>
                <line x1="18" y1="12" x2="22" y2="12"/>
            </svg>
            {html.escape(title)}
        </h1>
        <p class="subtitle">Knowledge Graph: {html.escape(self.kg.name)} | Generado: {datetime.now().strftime('%Y-%m-%d %H:%M')}</p>
    </div>
    
    <div class="metrics">
        <div class="metric-card">
            <div class="value">{stats['nodes']['total']:,}</div>
            <div class="label">Total Nodos</div>
        </div>
        <div class="metric-card">
            <div class="value">{stats['edges']['total']:,}</div>
            <div class="label">Total Aristas</div>
        </div>
        <div class="metric-card">
            <div class="value">{len(stats['nodes']['by_type'])}</div>
            <div class="label">Tipos de Nodos</div>
        </div>
        <div class="metric-card">
            <div class="value">{len(stats['edges']['by_relation'])}</div>
            <div class="label">Tipos de Relaciones</div>
        </div>
    </div>
    
    <div class="dashboard">
        <div class="card">
            <h3>🔷 Vista del Grafo</h3>
            <div id="miniGraph"></div>
        </div>
        
        <div class="card">
            <h3>📊 Distribución de Nodos</h3>
            <div class="chart-container">
                <canvas id="nodesChart"></canvas>
            </div>
        </div>
        
        <div class="card">
            <h3>🏆 Nodos Más Conectados</h3>
            <table>
                <thead>
                    <tr><th>#</th><th>Nodo</th><th>Conexiones</th></tr>
                </thead>
                <tbody>
                    {top_nodes_html}
                </tbody>
            </table>
        </div>
        
        <div class="card">
            <h3>🔗 Distribución de Relaciones</h3>
            <div class="chart-container">
                <canvas id="relationsChart"></canvas>
            </div>
        </div>
    </div>
    
    <script>
        // Datos
        const graphData = {json.dumps(graph_data)};
        const nodeTypeData = {node_type_data};
        const relationData = {relation_data};
        
        // Mini grafo
        const nodes = new vis.DataSet(graphData.nodes.slice(0, 100));
        const edges = new vis.DataSet(graphData.edges.filter(e => 
            graphData.nodes.slice(0, 100).find(n => n.id === e.from) &&
            graphData.nodes.slice(0, 100).find(n => n.id === e.to)
        ));
        
        new vis.Network(document.getElementById('miniGraph'), 
            {{ nodes, edges }}, 
            {{
                nodes: {{ shape: 'dot', size: 10, font: {{ size: 10 }} }},
                edges: {{ width: 1, arrows: 'to' }},
                physics: {{ solver: 'forceAtlas2Based', stabilization: {{ iterations: 100 }} }},
            }}
        );
        
        // Gráfico de nodos
        new Chart(document.getElementById('nodesChart'), {{
            type: 'doughnut',
            data: {{
                labels: nodeTypeData.map(d => d.type),
                datasets: [{{
                    data: nodeTypeData.map(d => d.count),
                    backgroundColor: {json.dumps(DEFAULT_COLORS)},
                }}]
            }},
            options: {{
                responsive: true,
                maintainAspectRatio: false,
                plugins: {{ legend: {{ position: 'right', labels: {{ color: '#ccc' }} }} }}
            }}
        }});
        
        // Gráfico de relaciones
        new Chart(document.getElementById('relationsChart'), {{
            type: 'bar',
            data: {{
                labels: relationData.map(d => d.relation),
                datasets: [{{
                    label: 'Cantidad',
                    data: relationData.map(d => d.count),
                    backgroundColor: '#00d4ff',
                }}]
            }},
            options: {{
                responsive: true,
                maintainAspectRatio: false,
                indexAxis: 'y',
                plugins: {{ legend: {{ display: false }} }},
                scales: {{
                    x: {{ ticks: {{ color: '#888' }}, grid: {{ color: '#333' }} }},
                    y: {{ ticks: {{ color: '#888' }}, grid: {{ display: false }} }}
                }}
            }}
        }});
    </script>
</body>
</html>'''
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(html_content, encoding='utf-8')
        
        return str(output_path)


# ═══════════════════════════════════════════════════════════════════════════════
# ONTOLOGY VISUALIZER
# ═══════════════════════════════════════════════════════════════════════════════

class OntologyVisualizer:
    """
    Visualiza la ontología como árbol jerárquico.
    """
    
    def __init__(self, ontology: "Ontology"):
        self.ontology = ontology
    
    def generate_html(self, output_path: str, title: str = "Ontology Structure") -> str:
        """Genera HTML con visualización de la ontología."""
        
        # Construir árbol de clases
        def build_tree(class_id: str, depth: int = 0) -> str:
            cls = self.ontology.classes.get(class_id)
            if not cls:
                return ""
            
            children = [c.id for c in self.ontology.classes.values() if c.parent == class_id]
            
            children_html = ""
            if children:
                children_items = "".join(build_tree(c, depth + 1) for c in children)
                children_html = f'<ul class="children">{children_items}</ul>'
            
            return f'''
            <li class="class-item" data-depth="{depth}">
                <div class="class-header">
                    <span class="class-name">{html.escape(cls.name)}</span>
                    <span class="class-id">({cls.id})</span>
                </div>
                <div class="class-desc">{html.escape(cls.description[:100]) if cls.description else ''}</div>
                <div class="synonyms">
                    {''.join(f'<span class="syn">{html.escape(s)}</span>' for s in cls.synonyms[:5])}
                </div>
                {children_html}
            </li>'''
        
        # Encontrar clases raíz
        root_classes = [c.id for c in self.ontology.classes.values() if not c.parent]
        tree_html = "".join(build_tree(rc) for rc in root_classes)
        
        # Relaciones
        relations_html = ""
        for rel in self.ontology.relations.values():
            relations_html += f'''
            <div class="relation-card">
                <div class="rel-header">
                    <span class="rel-name">{rel.id}</span>
                    <span class="rel-type">{rel.relation_type.value}</span>
                </div>
                <div class="rel-desc">{html.escape(rel.description)}</div>
                <div class="rel-constraints">
                    <span>Dominio: {', '.join(rel.domain) or 'Cualquiera'}</span>
                    <span>Rango: {', '.join(rel.range) or 'Cualquiera'}</span>
                </div>
            </div>'''
        
        html_content = f'''<!DOCTYPE html>
<html lang="es">
<head>
    <meta charset="UTF-8">
    <title>{html.escape(title)}</title>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{ 
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: #f8f9fa;
            padding: 40px;
        }}
        h1 {{ color: #333; margin-bottom: 30px; }}
        h2 {{ color: #555; margin: 30px 0 20px; font-size: 20px; }}
        
        .container {{ display: grid; grid-template-columns: 1fr 1fr; gap: 40px; }}
        
        .class-tree {{ background: white; padding: 30px; border-radius: 12px; box-shadow: 0 2px 10px rgba(0,0,0,0.05); }}
        .class-tree ul {{ list-style: none; padding-left: 25px; border-left: 2px solid #e0e0e0; }}
        .class-tree > ul {{ padding-left: 0; border-left: none; }}
        
        .class-item {{ margin: 15px 0; }}
        .class-header {{ display: flex; align-items: center; gap: 10px; }}
        .class-name {{ font-weight: 600; color: #333; font-size: 16px; }}
        .class-id {{ color: #888; font-size: 12px; }}
        .class-desc {{ color: #666; font-size: 13px; margin: 5px 0; }}
        .synonyms {{ display: flex; flex-wrap: wrap; gap: 5px; }}
        .syn {{ 
            background: #e3f2fd; 
            color: #1976d2; 
            padding: 2px 8px; 
            border-radius: 4px; 
            font-size: 11px; 
        }}
        
        .relations {{ background: white; padding: 30px; border-radius: 12px; box-shadow: 0 2px 10px rgba(0,0,0,0.05); }}
        .relation-card {{ 
            border: 1px solid #e0e0e0; 
            border-radius: 8px; 
            padding: 15px; 
            margin-bottom: 15px; 
        }}
        .rel-header {{ display: flex; justify-content: space-between; align-items: center; }}
        .rel-name {{ font-weight: 600; color: #333; }}
        .rel-type {{ 
            background: #e8f5e9; 
            color: #2e7d32; 
            padding: 3px 10px; 
            border-radius: 4px; 
            font-size: 12px; 
        }}
        .rel-desc {{ color: #666; font-size: 13px; margin: 10px 0; }}
        .rel-constraints {{ font-size: 12px; color: #888; display: flex; gap: 20px; }}
    </style>
</head>
<body>
    <h1>📚 {html.escape(self.ontology.name)}</h1>
    <p style="color:#666;margin-bottom:30px;">{html.escape(self.ontology.description)}</p>
    
    <div class="container">
        <div class="class-tree">
            <h2>🏛️ Jerarquía de Clases</h2>
            <ul>{tree_html}</ul>
        </div>
        
        <div class="relations">
            <h2>🔗 Relaciones</h2>
            {relations_html}
        </div>
    </div>
</body>
</html>'''
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(html_content, encoding='utf-8')
        
        return str(output_path)
