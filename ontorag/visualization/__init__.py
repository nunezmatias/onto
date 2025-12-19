"""
OntoRAG Visualization Module

This module provides visualization tools for knowledge graphs and ontologies.

Key Components:
    - GraphVisualizer: Generate interactive graph visualizations
    - OntologyVisualizer: Visualize ontology hierarchies

Output Formats:
    - HTML: Interactive visualization with D3.js or vis.js
    - PNG: Static image (requires matplotlib)
    - JSON: Data for custom visualizations

Example:
    >>> from ontorag.visualization import GraphVisualizer, OntologyVisualizer
    >>> from ontorag.graph import KnowledgeGraph
    >>> 
    >>> # Visualize graph
    >>> graph = KnowledgeGraph.load("my_graph.json")
    >>> viz = GraphVisualizer(graph)
    >>> viz.to_html("graph.html")
    >>> 
    >>> # Visualize ontology
    >>> onto = create_climate_ontology()
    >>> onto_viz = OntologyVisualizer(onto)
    >>> onto_viz.to_html("ontology.html")
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from ..graph import KnowledgeGraph
    from ..ontology import Ontology

logger = logging.getLogger(__name__)


# =============================================================================
# HTML TEMPLATES
# =============================================================================

GRAPH_HTML_TEMPLATE = """<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>{title}</title>
    <script src="https://d3js.org/d3.v7.min.js"></script>
    <style>
        body {{
            margin: 0;
            padding: 0;
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
            background: #1a1a2e;
            color: #eee;
        }}
        #container {{
            display: flex;
            height: 100vh;
        }}
        #graph {{
            flex: 1;
            background: #16213e;
        }}
        #sidebar {{
            width: 300px;
            padding: 20px;
            background: #0f3460;
            overflow-y: auto;
        }}
        h1 {{
            font-size: 1.5rem;
            margin-bottom: 1rem;
            color: #e94560;
        }}
        .stats {{
            background: rgba(255,255,255,0.1);
            padding: 10px;
            border-radius: 5px;
            margin-bottom: 1rem;
        }}
        .stats p {{
            margin: 5px 0;
        }}
        .node {{
            cursor: pointer;
        }}
        .node circle {{
            stroke: #fff;
            stroke-width: 2px;
        }}
        .node text {{
            font-size: 10px;
            fill: #eee;
        }}
        .link {{
            stroke: #666;
            stroke-opacity: 0.6;
        }}
        .link-label {{
            font-size: 8px;
            fill: #999;
        }}
        #tooltip {{
            position: absolute;
            background: rgba(0,0,0,0.8);
            padding: 10px;
            border-radius: 5px;
            font-size: 12px;
            display: none;
            max-width: 300px;
        }}
    </style>
</head>
<body>
    <div id="container">
        <div id="graph"></div>
        <div id="sidebar">
            <h1>{title}</h1>
            <div class="stats">
                <p><strong>Nodes:</strong> {num_nodes}</p>
                <p><strong>Edges:</strong> {num_edges}</p>
            </div>
            <div id="node-info">
                <p>Click a node to see details</p>
            </div>
        </div>
    </div>
    <div id="tooltip"></div>
    
    <script>
        const data = {graph_data};
        
        const width = document.getElementById('graph').clientWidth;
        const height = document.getElementById('graph').clientHeight;
        
        const colorScale = d3.scaleOrdinal(d3.schemeCategory10);
        
        const svg = d3.select('#graph')
            .append('svg')
            .attr('width', '100%')
            .attr('height', '100%')
            .attr('viewBox', [0, 0, width, height]);
        
        const g = svg.append('g');
        
        // Zoom behavior
        svg.call(d3.zoom()
            .extent([[0, 0], [width, height]])
            .scaleExtent([0.1, 10])
            .on('zoom', (event) => g.attr('transform', event.transform)));
        
        // Force simulation
        const simulation = d3.forceSimulation(data.nodes)
            .force('link', d3.forceLink(data.links).id(d => d.id).distance(100))
            .force('charge', d3.forceManyBody().strength(-300))
            .force('center', d3.forceCenter(width / 2, height / 2))
            .force('collision', d3.forceCollide().radius(30));
        
        // Links
        const link = g.append('g')
            .selectAll('line')
            .data(data.links)
            .join('line')
            .attr('class', 'link')
            .attr('stroke-width', d => Math.sqrt(d.weight || 1));
        
        // Link labels
        const linkLabel = g.append('g')
            .selectAll('text')
            .data(data.links)
            .join('text')
            .attr('class', 'link-label')
            .text(d => d.relation);
        
        // Nodes
        const node = g.append('g')
            .selectAll('.node')
            .data(data.nodes)
            .join('g')
            .attr('class', 'node')
            .call(d3.drag()
                .on('start', dragstarted)
                .on('drag', dragged)
                .on('end', dragended));
        
        node.append('circle')
            .attr('r', d => 8 + (d.degree || 1))
            .attr('fill', d => colorScale(d.type));
        
        node.append('text')
            .attr('dx', 12)
            .attr('dy', 4)
            .text(d => d.label);
        
        // Tooltip
        const tooltip = d3.select('#tooltip');
        
        node.on('mouseover', (event, d) => {{
            tooltip
                .style('display', 'block')
                .style('left', (event.pageX + 10) + 'px')
                .style('top', (event.pageY - 10) + 'px')
                .html(`<strong>${{d.label}}</strong><br>Type: ${{d.type}}`);
        }})
        .on('mouseout', () => tooltip.style('display', 'none'))
        .on('click', (event, d) => {{
            document.getElementById('node-info').innerHTML = `
                <h3>${{d.label}}</h3>
                <p><strong>ID:</strong> ${{d.id}}</p>
                <p><strong>Type:</strong> ${{d.type}}</p>
                <p><strong>Connections:</strong> ${{d.degree || 0}}</p>
            `;
        }});
        
        // Simulation tick
        simulation.on('tick', () => {{
            link
                .attr('x1', d => d.source.x)
                .attr('y1', d => d.source.y)
                .attr('x2', d => d.target.x)
                .attr('y2', d => d.target.y);
            
            linkLabel
                .attr('x', d => (d.source.x + d.target.x) / 2)
                .attr('y', d => (d.source.y + d.target.y) / 2);
            
            node.attr('transform', d => `translate(${{d.x}},${{d.y}})`);
        }});
        
        function dragstarted(event) {{
            if (!event.active) simulation.alphaTarget(0.3).restart();
            event.subject.fx = event.subject.x;
            event.subject.fy = event.subject.y;
        }}
        
        function dragged(event) {{
            event.subject.fx = event.x;
            event.subject.fy = event.y;
        }}
        
        function dragended(event) {{
            if (!event.active) simulation.alphaTarget(0);
            event.subject.fx = null;
            event.subject.fy = null;
        }}
    </script>
</body>
</html>"""


ONTOLOGY_HTML_TEMPLATE = """<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>{title}</title>
    <script src="https://d3js.org/d3.v7.min.js"></script>
    <style>
        body {{
            margin: 0;
            padding: 20px;
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
            background: #f5f5f5;
        }}
        h1 {{
            color: #333;
        }}
        .container {{
            display: flex;
            gap: 20px;
        }}
        .tree-container {{
            flex: 1;
            background: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .relations-container {{
            width: 400px;
            background: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .node circle {{
            fill: #69b3a2;
            stroke: #333;
            stroke-width: 2px;
        }}
        .node text {{
            font-size: 12px;
        }}
        .link {{
            fill: none;
            stroke: #ccc;
            stroke-width: 2px;
        }}
        .relation {{
            margin: 10px 0;
            padding: 10px;
            background: #f0f0f0;
            border-radius: 5px;
        }}
        .relation h4 {{
            margin: 0 0 5px 0;
            color: #e94560;
        }}
        .relation p {{
            margin: 5px 0;
            font-size: 14px;
        }}
    </style>
</head>
<body>
    <h1>{title}</h1>
    <div class="container">
        <div class="tree-container">
            <h2>Class Hierarchy</h2>
            <div id="tree"></div>
        </div>
        <div class="relations-container">
            <h2>Relations</h2>
            <div id="relations">{relations_html}</div>
        </div>
    </div>
    
    <script>
        const treeData = {tree_data};
        
        const margin = {{top: 20, right: 120, bottom: 20, left: 120}};
        const width = 800 - margin.left - margin.right;
        const height = 600 - margin.top - margin.bottom;
        
        const svg = d3.select('#tree')
            .append('svg')
            .attr('width', width + margin.left + margin.right)
            .attr('height', height + margin.top + margin.bottom)
            .append('g')
            .attr('transform', `translate(${{margin.left}},${{margin.top}})`);
        
        const tree = d3.tree().size([height, width]);
        const root = d3.hierarchy(treeData);
        tree(root);
        
        // Links
        svg.selectAll('.link')
            .data(root.links())
            .join('path')
            .attr('class', 'link')
            .attr('d', d3.linkHorizontal()
                .x(d => d.y)
                .y(d => d.x));
        
        // Nodes
        const node = svg.selectAll('.node')
            .data(root.descendants())
            .join('g')
            .attr('class', 'node')
            .attr('transform', d => `translate(${{d.y}},${{d.x}})`);
        
        node.append('circle').attr('r', 8);
        
        node.append('text')
            .attr('dy', 4)
            .attr('x', d => d.children ? -12 : 12)
            .attr('text-anchor', d => d.children ? 'end' : 'start')
            .text(d => d.data.name);
    </script>
</body>
</html>"""


# =============================================================================
# GRAPH VISUALIZER
# =============================================================================

class GraphVisualizer:
    """
    Visualize knowledge graphs.
    
    Generates interactive HTML visualizations using D3.js.
    
    Example:
        >>> viz = GraphVisualizer(graph)
        >>> viz.to_html("graph.html", title="My Knowledge Graph")
        >>> 
        >>> # Get visualization data
        >>> data = viz.to_dict()
    """
    
    def __init__(self, graph: "KnowledgeGraph"):
        """
        Initialize visualizer with a graph.
        
        Args:
            graph: Knowledge graph to visualize
        """
        self.graph = graph
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert graph to visualization-ready dictionary.
        
        Returns:
            Dictionary with nodes and links
        """
        # Calculate node degrees
        degrees = {}
        for edge in self.graph.edges:
            degrees[edge.source] = degrees.get(edge.source, 0) + 1
            degrees[edge.target] = degrees.get(edge.target, 0) + 1
        
        # Build nodes
        nodes = []
        for node in self.graph.nodes:
            nodes.append({
                "id": node.id,
                "label": node.label,
                "type": node.node_type,
                "degree": degrees.get(node.id, 0),
            })
        
        # Build links
        links = []
        for edge in self.graph.edges:
            links.append({
                "source": edge.source,
                "target": edge.target,
                "relation": edge.relation,
                "weight": edge.weight,
            })
        
        return {"nodes": nodes, "links": links}
    
    def to_html(
        self,
        path: str,
        title: str = "Knowledge Graph",
    ) -> str:
        """
        Generate interactive HTML visualization.
        
        Args:
            path: Output file path
            title: Page title
            
        Returns:
            Path to generated file
        """
        data = self.to_dict()
        
        html = GRAPH_HTML_TEMPLATE.format(
            title=title,
            num_nodes=len(data["nodes"]),
            num_edges=len(data["links"]),
            graph_data=json.dumps(data),
        )
        
        Path(path).write_text(html)
        logger.info(f"Graph visualization saved to {path}")
        
        return path
    
    def to_json(self, path: str) -> str:
        """
        Export visualization data as JSON.
        
        Args:
            path: Output file path
            
        Returns:
            Path to generated file
        """
        data = self.to_dict()
        Path(path).write_text(json.dumps(data, indent=2))
        return path


# =============================================================================
# ONTOLOGY VISUALIZER
# =============================================================================

class OntologyVisualizer:
    """
    Visualize ontology hierarchies.
    
    Generates tree visualizations showing class hierarchies
    and relation definitions.
    
    Example:
        >>> viz = OntologyVisualizer(ontology)
        >>> viz.to_html("ontology.html")
    """
    
    def __init__(self, ontology: "Ontology"):
        """
        Initialize visualizer with an ontology.
        
        Args:
            ontology: Ontology to visualize
        """
        self.ontology = ontology
    
    def _build_tree(self) -> Dict[str, Any]:
        """Build tree structure from class hierarchy."""
        # Find root classes (no parent)
        roots = [
            cls_id for cls_id, cls in self.ontology.classes.items()
            if not cls.parent
        ]
        
        def build_node(cls_id: str) -> Dict[str, Any]:
            cls = self.ontology.get_class(cls_id)
            children = self.ontology.get_subclasses(cls_id, recursive=False)
            
            node = {
                "name": cls.name if cls else cls_id,
                "id": cls_id,
            }
            
            if children:
                node["children"] = [build_node(c) for c in children]
            
            return node
        
        if len(roots) == 1:
            return build_node(roots[0])
        else:
            return {
                "name": "Ontology",
                "children": [build_node(r) for r in roots],
            }
    
    def _build_relations_html(self) -> str:
        """Build HTML for relations section."""
        html_parts = []
        
        for rel_id, rel in self.ontology.relations.items():
            html_parts.append(f"""
                <div class="relation">
                    <h4>{rel_id}</h4>
                    <p><strong>Name:</strong> {rel.name}</p>
                    <p><strong>Type:</strong> {rel.relation_type.value if hasattr(rel.relation_type, 'value') else rel.relation_type}</p>
                    <p><strong>Description:</strong> {rel.description or 'N/A'}</p>
                </div>
            """)
        
        return "".join(html_parts)
    
    def to_html(
        self,
        path: str,
        title: str = None,
    ) -> str:
        """
        Generate HTML visualization.
        
        Args:
            path: Output file path
            title: Page title
            
        Returns:
            Path to generated file
        """
        title = title or f"Ontology: {self.ontology.name}"
        tree_data = self._build_tree()
        relations_html = self._build_relations_html()
        
        html = ONTOLOGY_HTML_TEMPLATE.format(
            title=title,
            tree_data=json.dumps(tree_data),
            relations_html=relations_html,
        )
        
        Path(path).write_text(html)
        logger.info(f"Ontology visualization saved to {path}")
        
        return path
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Get ontology as visualization-ready dictionary.
        
        Returns:
            Dictionary with tree and relations
        """
        return {
            "tree": self._build_tree(),
            "relations": [r.to_dict() for r in self.ontology.relations.values()],
        }


__all__ = [
    "GraphVisualizer",
    "OntologyVisualizer",
]
