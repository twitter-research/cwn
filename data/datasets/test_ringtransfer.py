import networkx as nx
import matplotlib.pyplot as plt
import os.path as osp
import pytest

from data.datasets.ring_utils import generate_ringtree_graph_dataset
from data.utils import convert_graph_dataset_with_rings
from torch_geometric.utils import convert
from data.datasets import RingTransferDataset
from definitions import ROOT_DIR


@pytest.mark.runslow
def test_visualise_ringtree_dataset():
    dataset = generate_ringtree_graph_dataset(nodes=10, samples=100, classes=5)
    data = dataset[0]

    graph = convert.to_networkx(data, to_undirected=True)
    plt.figure()
    nx.draw_networkx(graph)
    plt.show()


def test_ringtree_dataset_generation():
    dataset = generate_ringtree_graph_dataset(nodes=10, samples=100)
    labels = dict()
    for data in dataset:
        assert data.edge_index.min() == 0
        assert data.edge_index.max() == 9

        label = data.y.item()
        if label not in labels:
            labels[label] = 0
        labels[label] += 1

    assert list(range(5)) == list(sorted(labels.keys()))
    assert {20} == set(labels.values())


def test_ringtree_dataset_conversion():
    dataset = generate_ringtree_graph_dataset(nodes=10, samples=100)
    complexes, _, _ = convert_graph_dataset_with_rings(dataset, max_ring_size=10,
                                                       include_down_adj=False)

    for complex in complexes:
        assert 2 in complex.chains
        assert complex.chains[2].num_simplices == 1
        assert complex.chains[1].num_simplices == 10
        assert complex.chains[0].num_simplices == 10


def test_ringtree_dataset_loading():
    # Test everything runs without errors.
    root = osp.join(ROOT_DIR, 'RINGTREE')
    dataset = RingTransferDataset(root=root)
    dataset.get(0)
