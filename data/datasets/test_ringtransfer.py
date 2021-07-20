import os.path as osp
import pytest

from data.datasets.ring_utils import generate_ring_transfer_graph_dataset
from data.utils import convert_graph_dataset_with_rings
from data.datasets import RingTransferDataset
from definitions import ROOT_DIR


def test_ringtree_dataset_generation():
    dataset = generate_ring_transfer_graph_dataset(nodes=10, samples=100, classes=5)
    labels = dict()
    for data in dataset:
        assert data.edge_index[0].min() == 0
        assert data.edge_index[1].min() == 0
        assert data.edge_index[0].max() == 9
        assert data.edge_index[1].max() == 9
        assert data.x.size(0) == 10
        assert data.x.size(1) == 5

        label = data.y.item()
        if label not in labels:
            labels[label] = 0
        labels[label] += 1

    assert list(range(5)) == list(sorted(labels.keys()))
    assert {20} == set(labels.values())


def test_ringtree_dataset_conversion():
    dataset = generate_ring_transfer_graph_dataset(nodes=10, samples=10, classes=5)
    complexes, _, _ = convert_graph_dataset_with_rings(dataset, max_ring_size=10,
                                                       include_down_adj=False, init_rings=True)

    for complex in complexes:
        assert 2 in complex.chains
        assert complex.chains[2].num_simplices == 1
        assert complex.chains[1].num_simplices == 10
        assert complex.chains[0].num_simplices == 10
        assert complex.nodes.x.size(0) == 10
        assert complex.nodes.x.size(1) == 5
        assert complex.edges.x.size(0) == 10
        assert complex.edges.x.size(1) == 5
        assert complex.triangles.x.size(0) == 1
        assert complex.triangles.x.size(1) == 5


def test_ringtree_dataset_loading():
    # Test everything runs without errors.
    root = osp.join(ROOT_DIR, 'datasets', 'RING-TRANSFER')
    dataset = RingTransferDataset(root=root, train=100, test=20)
    dataset.get(0)
