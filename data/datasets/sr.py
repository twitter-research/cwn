import os
import torch
import pickle

from data.sr_utils import load_sr_dataset
from data.utils import compute_clique_complex_with_gudhi
from data.datasets import InMemoryComplexDataset

import os.path as osp
import errno

def makedirs(path):
    try:
        os.makedirs(osp.expanduser(osp.normpath(path)))
    except OSError as e:
        if e.errno != errno.EEXIST and osp.isdir(path):
            raise e

class SRDataset(InMemoryComplexDataset):

    def __init__(self, root, name, max_dim=2, num_classes=2,
                 train_ids=None, val_ids=None, test_ids=None):
        self.name = name
        self._num_classes = num_classes
        super(SRDataset, self).__init__(root, max_dim=max_dim, num_classes=num_classes)
        
        with open(self.processed_paths[0], 'rb') as handle:
            self._data_list = pickle.load(handle)
            
        self.train_ids = list(range(self.len())) if train_ids is None else train_ids
        self.val_ids = list(range(self.len())) if val_ids is None else val_ids
        self.test_ids = list(range(self.len())) if test_ids is None else test_ids

    @property
    def processed_file_names(self):
        return ['{}_complex_list.pkl'.format(self.name)]       

    def process(self):
        data = load_sr_dataset(os.path.join(self.raw_dir, self.name + '.g6'))
        exp_dim = self.max_dim

        num_features = [None for _ in range(exp_dim+1)]
        complexes = list()
        max_dim = -1
        for datum in data:
            edge_index, num_nodes = datum
            x = torch.ones(num_nodes, 1, dtype=torch.float32)
            complex = compute_clique_complex_with_gudhi(x, edge_index, num_nodes,
                                                        expansion_dim=exp_dim)
            if complex.dimension > max_dim:
                max_dim = complex.dimension
            for dim in range(complex.dimension + 1):
                if num_features[dim] is None:
                    num_features[dim] = complex.chains[dim].num_features
                else:
                    assert num_features[dim] == complex.chains[dim].num_features
            complexes.append(complex)
        if max_dim != self.max_dim:
            self.max_dim = max_dim
            makedirs(self.processed_dir)
            
        # Here we add dummy labels to each of the complex.
        # They are not used in isomorphism testing; rather each complex
        # is embedded in a self.num_classes-dimensional space and pairwise
        # distances are computed and inspected.
        y = torch.ones(self.num_classes, dtype=torch.long)
        for complex in complexes:
            complex.y = y
        path = self.processed_paths[0]
        with open(path, 'wb') as handle:
            pickle.dump(complexes, handle)
