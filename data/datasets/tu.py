import os
import torch
import pickle
import numpy as np
from definitions import ROOT_DIR

from data.tu_utils import load_data, S2V_to_PyG, get_fold_indices
from data.utils import convert_graph_dataset_with_gudhi, convert_graph_dataset_with_rings
from data.datasets import InMemoryComplexDataset


def load_tu_graph_dataset(name, root=os.path.join(ROOT_DIR, 'datasets'), degree_as_tag=False, fold=0, seed=0):
    raw_dir = os.path.join(root, name, 'raw')
    load_from = os.path.join(raw_dir, '{}_graph_list_degree_as_tag_{}.pkl'.format(name, degree_as_tag))
    if os.path.isfile(load_from):
        with open(load_from, 'rb') as handle:
            graph_list = pickle.load(handle)
    else:
        data, num_classes = load_data(raw_dir, name, degree_as_tag)
        print('Converting graph data into PyG format...')
        graph_list = [S2V_to_PyG(datum) for datum in data]
        with open(load_from, 'wb') as handle:
            pickle.dump(graph_list, handle)
    train_filename = os.path.join(raw_dir, '10fold_idx', 'train_idx-{}.txt'.format(fold + 1))  
    test_filename = os.path.join(raw_dir, '10fold_idx', 'test_idx-{}.txt'.format(fold + 1))
    if os.path.isfile(train_filename) and os.path.isfile(test_filename):
        # NB: we consider the loaded test indices as val_ids ones and set test_ids to None
        #     to make it more convenient to work with the training pipeline
        train_ids = np.loadtxt(train_filename, dtype=int).tolist()
        val_ids = np.loadtxt(test_filename, dtype=int).tolist()
    else:
        train_ids, val_ids = get_fold_indices(graph_list, seed, fold)
    test_ids = None
    return graph_list, train_ids, val_ids, test_ids


class TUDataset(InMemoryComplexDataset):
    """A dataset of complexes obtained by lifting graphs from TUDatasets."""

    def __init__(self, root, name, max_dim=2, num_classes=2, degree_as_tag=False, fold=0,
                 init_method='sum', seed=0, include_down_adj=False, max_ring_size=None):
        self.name = name
        self.degree_as_tag = degree_as_tag
        assert max_ring_size is None or max_ring_size > 3
        self._max_ring_size = max_ring_size
        cellular = (max_ring_size is not None)
        if cellular:
            assert max_dim == 2

        super(TUDataset, self).__init__(root, max_dim=max_dim, num_classes=num_classes,
            init_method=init_method, include_down_adj=include_down_adj, cellular=cellular)

        self.data, self.slices = torch.load(self.processed_paths[0])
            
        self.fold = fold
        self.seed = seed
        train_filename = os.path.join(self.raw_dir, '10fold_idx', 'train_idx-{}.txt'.format(fold + 1))  
        test_filename = os.path.join(self.raw_dir, '10fold_idx', 'test_idx-{}.txt'.format(fold + 1))
        if os.path.isfile(train_filename) and os.path.isfile(test_filename):
            # NB: we consider the loaded test indices as val_ids ones and set test_ids to None
            #     to make it more convenient to work with the training pipeline
            self.train_ids = np.loadtxt(train_filename, dtype=int).tolist()
            self.val_ids = np.loadtxt(test_filename, dtype=int).tolist()
        else:
            train_ids, val_ids = get_fold_indices(self, self.seed, self.fold)
            self.train_ids = train_ids
            self.val_ids = val_ids
        self.test_ids = None
        # TODO: Add this later to our zip
        # tune_train_filename = os.path.join(self.raw_dir, 'tests_train_split.txt'.format(fold + 1))
        # self.tune_train_ids = np.loadtxt(tune_train_filename, dtype=int).tolist()
        # tune_test_filename = os.path.join(self.raw_dir, 'tests_val_split.txt'.format(fold + 1))
        # self.tune_val_ids = np.loadtxt(tune_test_filename, dtype=int).tolist()
        # self.tune_test_ids = None

    @property
    def processed_dir(self):
        """This is overwritten, so the cellular complex data is placed in another folder"""
        directory = super(TUDataset, self).processed_dir
        suffix = f"_{self._max_ring_size}rings" if self._cellular else ""
        suffix += f"_down_adj" if self.include_down_adj else ""
        return directory + suffix
            
    @property
    def processed_file_names(self):
        return ['{}_complex_list.pt'.format(self.name)]
    
    @property
    def raw_file_names(self):
        # The processed graph files are our raw files.
        # They are obtained when running the initial data conversion S2V_to_PyG. 
        return ['{}_graph_list_degree_as_tag_{}.pkl'.format(self.name, self.degree_as_tag)]
    
    def download(self):
        # This will process the raw data into a list of PyG Data objs.
        data, num_classes = load_data(self.raw_dir, self.name, self.degree_as_tag)
        self._num_classes = num_classes
        print('Converting graph data into PyG format...')
        graph_list = [S2V_to_PyG(datum) for datum in data]
        with open(self.raw_paths[0], 'wb') as handle:
            pickle.dump(graph_list, handle)
        
    def process(self):
        with open(self.raw_paths[0], 'rb') as handle:
            graph_list = pickle.load(handle)        
        
        if self._cellular:
            print("Converting the dataset accounting for rings...")
            complexes, _, _ = convert_graph_dataset_with_rings(graph_list, max_ring_size=self._max_ring_size,
                                                               include_down_adj=self.include_down_adj,
                                                               init_method=self._init_method,
                                                               init_edges=True, init_rings=True)
        else:
            print("Converting the dataset with gudhi...")
            # TODO: eventually remove the following comment
            # What about the init_method here? Adding now, although I remember we had handled this
            complexes, _, _ = convert_graph_dataset_with_gudhi(graph_list, expansion_dim=self.max_dim,
                                                               include_down_adj=self.include_down_adj,
                                                               init_method=self._init_method)

        torch.save(self.collate(complexes, self.max_dim), self.processed_paths[0])

    def get_tune_idx_split(self):
        raise NotImplementedError('Not implemented yet')
        # idx_split = {
        #     'train': self.tune_train_ids,
        #     'valid': self.tune_val_ids,
        #     'test': self.tune_test_ids}
        # return idx_split
