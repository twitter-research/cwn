from data.data_loading import load_dataset, load_graph_dataset
import torch
import os.path as osp
import pytest
from torch_geometric.datasets import ZINC


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

    datasets = [
        (dataset1_train, dataset2_train),
        (dataset1_valid, dataset2_valid),
        (dataset1_test, dataset2_test),
    ]

    for datas1, datas2 in datasets:
        for i, _ in enumerate(datas1):
            data1, data2 = datas1[i], datas2[i]
            assert torch.equal(data1.chains[0].x, data2.x)
            assert torch.equal(data1.y, data2.y)
            assert data1.chains[1].x.size(0) == (data2.edge_index.size(1) // 2)
