import networkx as nx
import torch
from torch_geometric.utils import to_undirected


def load_sr_dataset(path):
    """Load the Strongly Regular Graph Dataset from the supplied path."""
    nx_graphs = nx.read_graph6(path)
    graphs = list()
    for nx_graph in nx_graphs:
        n = nx_graph.number_of_nodes()
        edge_index = to_undirected(torch.tensor(list(nx_graph.edges()), dtype=torch.long).transpose(1,0))
        graphs.append((edge_index, n))
        
    return graphs
