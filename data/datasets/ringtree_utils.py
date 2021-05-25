import numpy as np
import torch
import random

from torch_geometric.data import Data
from sklearn.preprocessing import LabelBinarizer


def generate_ringtree_dictionary_lookup_graph(nodes):
    """This generates a dictionary lookup ring. No longer being used for now."""
    x = np.empty((nodes, 2 * (nodes-1)))
    # Assign all the other nodes in the ring a unique key and value
    keys = np.random.permutation(nodes - 1)
    vals = np.random.permutation(nodes - 1)

    oh_keys = np.array(LabelBinarizer().fit_transform(keys))
    oh_vals = np.array(LabelBinarizer().fit_transform(vals))
    oh_all = np.concatenate((oh_keys, oh_vals), axis=-1)
    x[1:, :] = oh_all

    # Assign the source node one of these random keys and set the value to -1
    key_idx = random.randint(0, nodes - 2)
    val = vals[key_idx]
    x[0, :] = -1
    x[0, :(nodes - 1)] = oh_keys[key_idx]

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
    y = torch.tensor([val], dtype=torch.long)
    return Data(x=x, edge_index=edge_index, mask=mask, y=y)


def generate_ringtree_graph(nodes, target_label):
    opposite_node = len(nodes) // 2

    # Initialise the feature matrix. All nodes have random features except the target and the opp.
    # The target has [0, ..., 0] while the opposite has the one-hot encoded target label
    x = np.random.normal(0.0, 1.0, size=(nodes, len(target_label)))
    x[0:, ] = 0.0
    x[opposite_node, :] = target_label
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
    y = torch.tensor([np.argmax(target_label)], dtype=torch.long)
    return Data(x=x, edge_index=edge_index, mask=mask, y=y)


def generate_ringtree_graph_dataset(nodes, classes=5, samples=10000):
    # Generate the dataset
    dataset = []
    samples_per_class = samples // classes
    for i in range(samples):
        label = i // samples_per_class
        target_class = np.zeros(classes)
        target_class[label] = 1.0
        graph = generate_ringtree_graph(nodes, target_class)
        dataset.append(graph)
    return dataset
