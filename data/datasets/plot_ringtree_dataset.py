import networkx as nx
import matplotlib.pyplot as plt

from data.datasets.ring_utils import generate_ring_transfer_graph_dataset
from torch_geometric.utils import convert


def visualise_ringtree_dataset():
    dataset = generate_ring_transfer_graph_dataset(nodes=10, samples=100, classes=5)
    data = dataset[0]

    graph = convert.to_networkx(data, to_undirected=True)
    plt.figure()
    nx.draw_networkx(graph)
    plt.show()


if __name__ == "__main__":
    visualise_ringtree_dataset()
