from data.data_loading import load_dataset, load_graph_dataset
import torch
import os.path as osp
import pytest
from torch_geometric.datasets import ZINC


def check_edge_index_are_the_same(upper_index, edge_index):
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


def check_edge_attr_are_the_same(face_index, ex, edge_index, edge_attr):
    # The maximum node that has an edge must be the same.
    assert face_index[0, :].max() == edge_index.max()
    # The number of edges present in both tensors should be the same.
    assert face_index.size(1) == edge_index.size(1)

    edges = face_index.size(1)
    id_to_edge = dict()
    for i in range(edges):
        edge_id = face_index[1, i].item()
        vertex = face_index[0, i].item()
        if edge_id not in id_to_edge:
            id_to_edge[edge_id] = []
        id_to_edge[edge_id].append(vertex)

    edge_to_id = dict()
    for edge_idx, edge in id_to_edge.items():
        edge_to_id[tuple(sorted(edge))] = edge_idx

    for i in range(edges):
        e1, e2 = edge_index[0, i].item(), edge_index[1, i].item()
        edge = tuple(sorted([e1, e2]))

        edge_attr1 = ex[edge_to_id[edge]].squeeze()
        edge_attr2 = edge_attr[i].squeeze()
        assert edge_attr1 == edge_attr2


@pytest.mark.slow
def test_zinc_splits_are_retained():
    dataset1 = load_dataset("ZINC", max_ring_size=3, use_edge_features=True)
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
