import networkx as nx
import numpy as np
import json
from utils import save_graph

def generate_graph(n_nodes=10):
    """
    Generate a random geometric graph.

    Node features include:
    - x, y coordinates (from graph layout)
    - Degree of the node
    - Clustering coefficient of the node

    :param n_nodes: Number of nodes in the graph
    :return: A NetworkX graph with meaningful features added to each node
    """
    # Generate a random geometric graph
    graph = nx.random_geometric_graph(n_nodes, radius=0.3)
    pos = nx.get_node_attributes(graph, 'pos')  # Get (x, y) coordinates for layout

    # Add meaningful features to each node
    for node in graph.nodes:
        x, y = pos[node]  # Extract (x, y) coordinates
        degree = graph.degree[node]  # Node degree
        clustering = nx.clustering(graph, node)  # Node clustering coefficient

        # Node features: [x, y, degree, clustering]
        graph.nodes[node]['features'] = [x, y, degree, clustering]

    return graph

if __name__ == "__main__":
    graph = generate_graph(n_nodes=10)
    print("Generated graph with nodes:")
    for node, data in graph.nodes(data=True):
        print(f"Node {node}: {data['features']}")
    save_graph(graph, "sample_graphs/clean_graph.json")
