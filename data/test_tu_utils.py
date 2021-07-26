import pytest
import os
import numpy as np
import torch
import random
from data.tu_utils import get_fold_indices, load_data, S2V_to_PyG
from torch_geometric.utils import degree
from definitions import ROOT_DIR


@pytest.fixture
def imdbbinary_graphs():
    data, num_classes = load_data(os.path.join(ROOT_DIR, 'datasets', 'IMDBBINARY', 'raw'), 'IMDBBINARY', True)
    graph_list = [S2V_to_PyG(datum) for datum in data]
    return graph_list

@pytest.fixture
def imdbbinary_nonattributed_graphs():
    data, num_classes = load_data(os.path.join(ROOT_DIR, 'datasets', 'IMDBBINARY', 'raw'), 'IMDBBINARY', False)
    graph_list = [S2V_to_PyG(datum) for datum in data]
    return graph_list

@pytest.fixture
def proteins_graphs():
    data, num_classes = load_data(os.path.join(ROOT_DIR, 'datasets', 'PROTEINS', 'raw'), 'PROTEINS', True)
    graph_list = [S2V_to_PyG(datum) for datum in data]
    return graph_list


def validate_degree_as_tag(graphs):
    
    degree_set = set()
    degrees = dict()
    for g, graph in enumerate(graphs):
        d = degree(graph.edge_index[0])
        d = d.numpy().astype(int).tolist()
        degree_set |= set(d)
        degrees[g] = d
    encoder = {deg: d for d, deg in enumerate(sorted(degree_set))}
    for g, graph in enumerate(graphs):
        feats = graph.x
        edge_index = graph.edge_index
        assert feats.shape[1] == len(encoder)
        row_sum = torch.sum(feats, 1)
        assert torch.equal(row_sum, torch.ones(feats.shape[0]))
        tags = torch.argmax(feats, 1)
        d = degrees[g]
        encoded = torch.LongTensor([encoder[deg] for deg in d])
        assert torch.equal(tags, encoded), '{}\n{}'.format(tags, encoded)
        
        
def validate_get_fold_indices(graphs):

    seeds = [0, 42, 43, 666]
    folds = list(range(10))
    
    prev_train  = None
    prev_test  = None
    for fold in folds:
        for seed in seeds:
            torch.manual_seed(43)
            np.random.seed(43)
            random.seed(43)
            train_idx_0, test_idx_0 = get_fold_indices(graphs, seed, fold)
            torch.manual_seed(0)
            np.random.seed(0)
            random.seed(0)
            train_idx_1, test_idx_1 = get_fold_indices(graphs, seed, fold)
            # check the splitting procedure is deterministic and robust w.r.t. global seeds
            assert np.all(np.equal(train_idx_0, train_idx_1))
            assert np.all(np.equal(test_idx_0, test_idx_1))
            # check test and train form a partition
            assert len(set(train_idx_0) & set(test_idx_0)) == 0
            assert len(set(train_idx_0) | set(test_idx_0)) == len(graphs)
            # check idxs are different across seeds
            if prev_train is not None:
                assert np.any(~np.equal(train_idx_0, prev_train))
                assert np.any(~np.equal(test_idx_0, prev_test))
            prev_train = train_idx_0
            prev_test = test_idx_0


def validate_constant_scalar_features(graphs):
    
    for graph in graphs:
        feats = graph.x
        assert feats.shape[1]
        expected = torch.ones(feats.shape[0], 1)
        assert torch.equal(feats, expected)
        

@pytest.mark.data
def test_get_fold_indices_on_imdbbinary(imdbbinary_graphs):
    validate_get_fold_indices(imdbbinary_graphs)


@pytest.mark.data
def test_degree_as_tag_on_imdbbinary(imdbbinary_graphs):
    validate_degree_as_tag(imdbbinary_graphs)


@pytest.mark.data
def test_constant_scalar_features_on_imdbbinary_without_tags(imdbbinary_nonattributed_graphs):
    validate_constant_scalar_features(imdbbinary_nonattributed_graphs)


@pytest.mark.data
def test_degree_as_tag_on_proteins(proteins_graphs):
    validate_degree_as_tag(proteins_graphs)
