import itertools

import torch
import os.path as osp
import pytest
import networkx as nx

from data.data_loading import load_dataset
from torch_geometric.datasets import ZINC
from torch_geometric.utils import convert
from torch_geometric.data import Data


def check_edge_index_are_the_same(upper_index, edge_index):
    # These two tensors should have the same content but in different order.
    assert upper_index.size() == edge_index.size()
    num_edges = edge_index.size(1)

    edge_set1 = set()
    edge_set2 = set()
    for i in range(num_edges):
        e1, e2 = edge_index[0, i].item(), edge_index[1, i].item()
        edge1 = tuple(sorted([e1, e2]))
        edge_set1.add(edge1)

        e1, e2 = upper_index[0, i].item(), upper_index[1, i].item()
        edge2 = tuple(sorted([e1, e2]))
        edge_set2.add(edge2)

    assert edge_set1 == edge_set2


def get_table(face_index):
    elements = face_index.size(1)
    id_to_cell = dict()
    for i in range(elements):
        cell_id = face_index[1, i].item()
        face = face_index[0, i].item()
        if cell_id not in id_to_cell:
            id_to_cell[cell_id] = []
        id_to_cell[cell_id].append(face)
    return id_to_cell


def check_edge_attr_are_the_same(face_index, ex, edge_index, edge_attr):
    # The maximum node that has an edge must be the same.
    assert face_index[0, :].max() == edge_index.max()
    # The number of edges present in both tensors should be the same.
    assert face_index.size(1) == edge_index.size(1)

    id_to_edge = get_table(face_index)

    edge_to_id = dict()
    for edge_idx, edge in id_to_edge.items():
        edge_to_id[tuple(sorted(edge))] = edge_idx

    edges = face_index.size(1)
    for i in range(edges):
        e1, e2 = edge_index[0, i].item(), edge_index[1, i].item()
        edge = tuple(sorted([e1, e2]))

        edge_attr1 = ex[edge_to_id[edge]].squeeze()
        edge_attr2 = edge_attr[i].squeeze()
        assert edge_attr1 == edge_attr2


def get_rings(n, edge_index, max_ring):
    x = torch.zeros((n, 1))
    data = Data(x, edge_index=edge_index)
    graph = convert.to_networkx(data)

    def is_cycle_edge(i1, i2, cycle):
        if i2 == i1 + 1:
            return True
        if i1 == 0 and i2 == len(cycle) - 1:
            return True
        return False

    def is_chordless(cycle):
        for (i1, v1), (i2, v2) in itertools.combinations(enumerate(cycle), 2):
            if not is_cycle_edge(i1, i2, cycle) and graph.has_edge(v1, v2):
                return False
        return True

    nx_rings = set()
    for cycle in nx.simple_cycles(graph):
        # Because we need to use a DiGraph for this method, it will also return each edge
        # as a cycle. So we skip these together with cycles above the maximum length.
        if len(cycle) <= 2 or len(cycle) > max_ring:
            continue

        # We remove the chordless cycles
        if not is_chordless(cycle):
            continue

        nx_rings.add(tuple(sorted(cycle)))

    return nx_rings


def get_complex_rings(r_face_index, e_face_index):
    # Construct the edge and ring tables
    id_to_ring = get_table(r_face_index)
    id_to_edge = get_table(e_face_index)

    rings = set()
    for ring, edges in id_to_ring.items():
        # Compose the two tables to extract the vertices in the ring.
        vertices = []
        for edge in edges:
            vertices += id_to_edge[edge]
        # Eliminate duplicates.
        vertices = set(vertices)
        # Store the vertices in sorted order.
        vertices = tuple(sorted(vertices))
        rings.add(vertices)
    return rings


@pytest.mark.slow
def test_zinc_splits_are_retained():
    dataset1 = load_dataset("ZINC", max_ring_size=7, use_edge_features=True)
    dataset1_train = dataset1.get_split('train')
    dataset1_valid = dataset1.get_split('valid')
    dataset1_test = dataset1.get_split('test')

    raw_dir = osp.join(dataset1.root, 'raw')
    dataset2_train = ZINC(raw_dir, subset=True, split='train')
    dataset2_valid = ZINC(raw_dir, subset=True, split='val')
    dataset2_test = ZINC(raw_dir, subset=True, split='test')

    datasets1 = [dataset1_train, dataset1_valid, dataset1_test]
    datasets2 = [dataset2_train, dataset2_valid, dataset2_test]
    datasets = zip(datasets1, datasets2)

    for datas1, datas2 in datasets:
        for i, _ in enumerate(datas1):
            data1, data2 = datas1[i], datas2[i]

            assert torch.equal(data1.y, data2.y)
            assert torch.equal(data1.chains[0].x, data2.x)
            assert data1.chains[1].x.size(0) == (data2.edge_index.size(1) // 2)
            check_edge_index_are_the_same(data1.chains[0].upper_index, data2.edge_index)
            check_edge_attr_are_the_same(data1.chains[1].face_index,
                                         data1.chains[1].x, data2.edge_index, data2.edge_attr)


@pytest.mark.slow
def test_we_find_all_the_induced_cycles_on_zinc():
    max_ring = 7
    dataset = load_dataset("ZINC", max_ring_size=max_ring, use_edge_features=True)
    # Check only on validation to save time. I've also run once on the whole dataset and passes.
    dataset = dataset.get_split('valid')

    for complex in dataset:
        nx_rings = get_rings(complex.nodes.num_simplices, complex.nodes.upper_index,
                             max_ring=max_ring)
        if 2 not in complex.chains:
            assert len(nx_rings) == 0
            continue

        complex_rings = get_complex_rings(complex.chains[2].face_index, complex.edges.face_index)
        assert len(complex_rings) > 0
        assert len(nx_rings) == complex.chains[2].num_simplices
        assert nx_rings == complex_rings


