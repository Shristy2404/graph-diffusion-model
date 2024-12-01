import json

def save_graph(graph, filename):
    """
    Save a NetworkX graph to a file.

    :param graph: The NetworkX graph
    :param filename: Output filename (JSON format)
    """
    data = {
        "nodes": {
            node: {"features": data["features"]}
            for node, data in graph.nodes(data=True)
        },
        "edges": list(graph.edges)
    }
    with open(filename, "w") as f:
        json.dump(data, f)
