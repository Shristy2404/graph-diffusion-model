import matplotlib.pyplot as plt
from data.generate_graph import generate_graph
from data.add_noise import add_noise_to_graph

def plot_graph(graph, title="Graph", highlight_changes=False, denoised_graph=None):
    """
    Visualize a graph with nodes positioned by their x, y coordinates.

    :param graph: Input NetworkX graph
    :param title: Title of the plot
    :param highlight_changes: If True, visualize differences between noisy and denoised graphs
    :param denoised_graph: Optional denoised graph for comparison
    """
    plt.figure(figsize=(8, 8))

    for node, data in graph.nodes(data=True):
        x, y = data['features'][:2]
        plt.scatter(x, y, label=f"Node {node}", c='blue', alpha=0.7)

        if highlight_changes and denoised_graph is not None:
            dx, dy = denoised_graph.nodes[node]['features'][:2]
            plt.arrow(x, y, dx - x, dy - y, color='red', alpha=0.5, width=0.002)

    plt.title(title)
    plt.xlabel("X-coordinate")
    plt.ylabel("Y-coordinate")
    plt.grid(True)
    plt.show()

def plot_comparison(clean_graph, noisy_graph, denoised_graph, filename=None):
    """
    Plot and compare clean, noisy, and denoised graphs side by side.

    :param clean_graph: Clean (ground truth) NetworkX graph
    :param noisy_graph: Noisy NetworkX graph
    :param denoised_graph: Denoised NetworkX graph
    :param filename: If provided, save the figure to this file
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    def _plot(ax, graph, title):
        for node, data in graph.nodes(data=True):
            x, y = data['features'][:2]
            ax.scatter(x, y, alpha=0.7, label=f"Node {node}")
        ax.set_title(title)
        ax.set_xlabel("X-coordinate")
        ax.set_ylabel("Y-coordinate")
        ax.grid(True)

    _plot(axes[0], clean_graph, "Clean Graph")
    _plot(axes[1], noisy_graph, "Noisy Graph")
    _plot(axes[2], denoised_graph, "Denoised Graph")

    plt.tight_layout()
    if filename:
        plt.savefig(filename)
    plt.show()

if __name__ == "__main__":

    # Generate clean and noisy graphs
    clean_graph = generate_graph(n_nodes=10)
    noisy_graph = add_noise_to_graph(clean_graph, noise_std=0.1)

    # Mock denoised graph (for demonstration purposes, use noisy graph directly)
    denoised_graph = noisy_graph

    # Visualize graphs
    plot_graph(clean_graph, title="Clean Graph")
    plot_graph(noisy_graph, title="Noisy Graph")
    plot_comparison(clean_graph, noisy_graph, denoised_graph)
