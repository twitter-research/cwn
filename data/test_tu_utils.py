import pytest
import os
import numpy as np
import torch
import random
from data.tu_utils import get_fold_indices, load_data, S2V_to_PyG
from definitions import ROOT_DIR

@pytest.fixture
def redditbinary_graphs():
    data, num_classes = load_data(os.path.join(ROOT_DIR, 'datasets', 'REDDITBINARY', 'raw'), 'REDDITBINARY', True)
    graph_list = [S2V_to_PyG(datum) for datum in data]
    return graph_list

def test_get_fold_indices(redditbinary_graphs):

    seeds = [0, 42, 43, 666]
    folds = list(range(10))
    
    prev_train  = None
    prev_test  = None
    for fold in folds:
        for seed in seeds:
            torch.manual_seed(43)
            np.random.seed(43)
            random.seed(43)
            train_idx_0, test_idx_0 = get_fold_indices(redditbinary_graphs, seed, fold)
            torch.manual_seed(0)
            np.random.seed(0)
            random.seed(0)
            train_idx_1, test_idx_1 = get_fold_indices(redditbinary_graphs, seed, fold)
            # check the splitting procedure is deterministic and robust w.r.t. global seeds
            assert np.all(np.equal(train_idx_0, train_idx_1))
            assert np.all(np.equal(test_idx_0, test_idx_1))
            # check test and train form a partition
            assert len(set(train_idx_0) & set(test_idx_0)) == 0
            assert len(set(train_idx_0) | set(test_idx_0)) == len(redditbinary_graphs)
            # check idxs are different across seeds
            if prev_train is not None:
                assert np.any(~np.equal(train_idx_0, prev_train))
                assert np.any(~np.equal(test_idx_0, prev_test))
            prev_train = train_idx_0
            prev_test = test_idx_0