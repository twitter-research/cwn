import os
import pickle
import numpy as np
from definitions import ROOT_DIR

from data.tu_utils import load_data, S2V_to_PyG, get_fold_indices
from data.utils import convert_graph_dataset_with_gudhi
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

    def __init__(self, root, name, max_dim=2, num_classes=2, degree_as_tag=False, fold=0,
                 init_method='sum', max_dim_limit=None, randomize_ids=False, max_density=1.0,
                 seed=0):
        self.name = name
        self.degree_as_tag = degree_as_tag
        self.max_dim_limit = max_dim_limit
        self.randomize_ids = randomize_ids
        self.max_density = max_density
        super(TUDataset, self).__init__(root, max_dim=max_dim, num_classes=num_classes,
            init_method=init_method)

        with open(self.processed_paths[0], 'rb') as handle:
            self._data_list = pickle.load(handle)
            
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
            train_ids, val_ids = get_fold_indices(self._data_list, self.seed, self.fold)
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
        processed_dir = super(TUDataset, self).processed_dir
        if self.max_dim_limit is not None:
            processed_dir += f'_max_dim_limit{self.max_dim_limit}'
        if self.randomize_ids:
            processed_dir += f'_randomized_ids'
        if self.max_density < 1.0:
            processed_dir += f'_max_density{self.max_density}'
        return processed_dir
            
    @property
    def processed_file_names(self):
        return ['{}_complex_list.pkl'.format(self.name)]
    
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
        
        print("Converting the dataset with gudhi...")
        complexes, _, _ = convert_graph_dataset_with_gudhi(graph_list, expansion_dim=self.max_dim,
                                                           include_down_adj=self.include_down_adj,
                                                           max_density=self.max_density,
                                                           max_dim_limit=self.max_dim_limit,
                                                           randomize_ids=self.randomize_ids)
        
        with open(self.processed_paths[0], 'wb') as handle:
            pickle.dump(complexes, handle)

    def get_tune_idx_split(self):
        raise NotImplementedError('Not implemented yet')
        # idx_split = {
        #     'train': self.tune_train_ids,
        #     'valid': self.tune_val_ids,
        #     'test': self.tune_test_ids}
        # return idx_split

