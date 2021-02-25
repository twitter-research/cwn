import torch
import pickle

from data.datasets import InMemoryComplexDataset
from data.datasets.flow_utils import load_flow_dataset


class FlowDataset(InMemoryComplexDataset):

    def __init__(self, root, name, num_points, train_samples, val_samples):
        self.name = name
        self._num_classes = 3
        self._num_points = num_points
        self._train_samples = train_samples
        self._val_samples = val_samples
        self._num_points = 600
        super(FlowDataset, self).__init__(root, max_dim=1, num_classes=3, include_down_adj=True)

        with open(self.processed_paths[0], 'rb') as handle:
            self._data_list = pickle.load(handle)

        self.train_ids = list(range(train_samples))
        self.val_ids = list(range(train_samples, train_samples + val_samples))
        self.test_ids = None

    @property
    def processed_file_names(self):
        return ['{}_complex_list.pkl'.format(self.name)]

    def process(self):
        train, val = load_flow_dataset(num_points=self._num_points,
            num_train=self._train_samples, num_test=self._val_samples)

        chains = train + val
        path = self.processed_paths[0]
        with open(path, 'wb') as handle:
            pickle.dump(chains, handle)
