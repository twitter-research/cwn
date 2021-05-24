import numpy as np
import torch
import random

from torch_geometric.data import Data


def generate_ringtree_graph(nodes):
    x = np.empty((nodes, 2))
    # Assign all the other nodes in the ring a unique key and value
    x[1:, 0] = np.arange(1, nodes)
    x[1:, 1] = np.random.permutation(nodes - 1)

    # Assign the source node one of these random keys and set the value to -1
    key = random.randint(1, nodes - 1)
    x[0, 0] = key
    x[0, 1] = -1
    x = torch.tensor(x, dtype=torch.float32)

    edge_index = []
    for i in range(nodes-1):
        # Add in one direction
        edge_index.append([i, i + 1])
        edge_index.append([i + 1, i])

    # Add the edges that close the ring
    edge_index.append([0, nodes - 1])
    edge_index.append([nodes - 1, 0])

    edge_index = np.array(edge_index, dtype=np.long).T
    edge_index = torch.tensor(edge_index, dtype=torch.long)

    # Create a mask for the target node of the graph
    mask = torch.zeros(nodes, dtype=torch.bool)
    mask[0] = 1

    # Add the label of the graph as a graph label
    y = torch.tensor([x[key, 1].item()], dtype=torch.long)
    return Data(x=x, edge_index=edge_index, mask=mask, y=y)


def generate_ringtree_graph_dataset(nodes, samples=1000):
    # Generate the dataset
    dataset = []
    for i in range(samples):
        graph = generate_ringtree_graph(nodes)
        dataset.append(graph)
    return dataset
