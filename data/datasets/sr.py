import os
import torch
import pickle

from data.sr_utils import load_sr_dataset
from data.utils import compute_clique_complex_with_gudhi
from data.datasets import InMemoryComplexDataset


class SRDataset(InMemoryComplexDataset):

    def __init__(self, root, name, max_dim=2, num_classes=2,
                 train_ids=None, val_ids=None, test_ids=None):
        self.name = name
        super(SRDataset, self).__init__(root, max_dim=max_dim, num_classes=num_classes)

        self.train_ids = train_ids
        self.val_ids = val_ids
        self.test_ids = test_ids

        with open(self.processed_paths[0], 'rb') as handle:
            self._data_list = pickle.load(handle)

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
        self.max_dim = max_dim
        y = torch.ones(len(complexes), dtype=torch.long)
        for complex in complexes:
            complex.y = y
        path = self.processed_paths[0]
        with open(path, 'wb') as handle:
            pickle.dump(complexes, handle)
