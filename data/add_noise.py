""" To experiemnt with various noise additions to graph node features """

import json
import networkx as nx
import numpy as np
from utils import save_graph

def add_noise_to_graph(graph, noise_std=0.1):
    """
    Add Gaussian noise to the x, y coordinates of the nodes in the graph.

    :param graph: Input NetworkX graph
    :param noise_std: Standard deviation of the noise
    :return: A new graph with noisy x, y coordinates
    """
    noisy_graph = graph.copy()
    for node in noisy_graph.nodes:
        features = noisy_graph.nodes[node]['features']
        x, y = features[:2]  # Extract x, y coordinates
        noisy_x = x + np.random.normal(0, noise_std)
        noisy_y = y + np.random.normal(0, noise_std)
        # Update the noisy coordinates, keeping other features intact
        noisy_graph.nodes[node]['features'] = [noisy_x, noisy_y] + features[2:]
    return noisy_graph

def load_graph(filename):
    """
    Load a graph from a JSON file.

    :param filename: Input filename
    :return: A NetworkX graph
    """
    with open(filename, "r") as f:
        data = json.load(f)
    graph = nx.Graph()
    for node, attrs in data["nodes"].items():
        graph.add_node(int(node), features=attrs["features"])
    graph.add_edges_from(data["edges"])
    return graph

if __name__ == "__main__":
    # Load the clean graph and add noise
    clean_graph = load_graph("sample_graphs/clean_graph.json")
    noisy_graph = add_noise_to_graph(clean_graph, noise_std=0.1)

    # Print the noisy graph
    print("Noisy graph with nodes:")
    for node, data in noisy_graph.nodes(data=True):
        print(f"Node {node}: {data['features']}")

    # Save the noisy graph
    save_graph(noisy_graph, "sample_graphs/noisy_graph.json")
