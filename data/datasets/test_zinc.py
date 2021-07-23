import torch
import os.path as osp
import pytest

from data.data_loading import load_dataset
from data.helper_test import (check_edge_index_are_the_same, 
                              check_edge_attr_are_the_same, get_rings, 
                              get_complex_rings)
from torch_geometric.datasets import ZINC


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
            assert torch.equal(data1.cochains[0].x, data2.x)
            assert data1.cochains[1].x.size(0) == (data2.edge_index.size(1) // 2)
            check_edge_index_are_the_same(data1.cochains[0].upper_index, data2.edge_index)
            check_edge_attr_are_the_same(data1.cochains[1].boundary_index,
                                         data1.cochains[1].x, data2.edge_index, data2.edge_attr)


@pytest.mark.slow
def test_we_find_only_the_induced_cycles_on_zinc():
    max_ring = 7
    dataset = load_dataset("ZINC", max_ring_size=max_ring, use_edge_features=True)
    # Check only on validation to save time. I've also run once on the whole dataset and passes.
    dataset = dataset.get_split('valid')

    for complex in dataset:
        nx_rings = get_rings(complex.nodes.num_cells, complex.nodes.upper_index,
                             max_ring=max_ring)
        if 2 not in complex.cochains:
            assert len(nx_rings) == 0
            continue

        complex_rings = get_complex_rings(complex.cochains[2].boundary_index, complex.edges.boundary_index)
        assert len(complex_rings) > 0
        assert len(nx_rings) == complex.cochains[2].num_cells
        assert nx_rings == complex_rings


