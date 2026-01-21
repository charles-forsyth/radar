from typing import List
from radar.db.models import Entity, Connection
import json


def generate_graph_html(
    entities: List[Entity],
    connections: List[Connection],
    output_file: str = "radar_graph.html",
):
    """Generates an interactive 3D force-directed graph HTML file."""

    # Prepare data for 3d-force-graph
    nodes = []
    for e in entities:
        color = "#1f77b4"  # Default Blue
        if e.type == "COMPANY":
            color = "#ff7f0e"  # Orange
        elif e.type == "TECH":
            color = "#2ca02c"  # Green
        elif e.type == "PERSON":
            color = "#d62728"  # Red
        elif e.type == "MARKET":
            color = "#9467bd"  # Purple

        nodes.append(
            {
                "id": str(e.id),
                "name": e.name,
                "val": 1,
                "color": color,
                "desc": e.details.get("description", "") if e.details else "",
            }
        )

    links = []
    for c in connections:
        links.append(
            {
                "source": str(c.source_uuid),
                "target": str(c.target_uuid),
                "name": c.type,
                "desc": c.meta_data.get("description", "") if c.meta_data else "",
            }
        )

    graph_data = {"nodes": nodes, "links": links}
    json_data = json.dumps(graph_data)

    html_content = f"""
    <head>
      <style> body {{ margin: 0; }} </style>
      <script src="//unpkg.com/3d-force-graph"></script>
    </head>
    <body>
      <div id="3d-graph"></div>
      <script>
        const gData = {json_data};
        const Graph = ForceGraph3D()
          (document.getElementById('3d-graph'))
            .graphData(gData)
            .nodeLabel('name')
            .nodeAutoColorBy('group')
            .linkDirectionalArrowLength(3.5)
            .linkDirectionalArrowRelPos(1)
            .linkLabel(link => link.name + ": " + link.desc)
            .onNodeClick(node => {{
                // Aim at node from outside it
                const distance = 40;
                const distRatio = 1 + distance/Math.hypot(node.x, node.y, node.z);

                Graph.cameraPosition(
                  {{ x: node.x * distRatio, y: node.y * distRatio, z: node.z * distRatio }}, // new position
                  node, // lookAt ({0}, {1}, {2})
                  3000  // ms transition duration
                );
            }});
      </script>
    </body>
    """

    with open(output_file, "w") as f:
        f.write(html_content)

    return output_file
