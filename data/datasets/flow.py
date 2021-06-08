import pickle
import os.path as osp

from data.datasets import InMemoryComplexDataset
from data.datasets.flow_utils import load_flow_dataset


# TODO: Set up a chain dataset structure or make complex dataset better support chain-only data.
# TODO: Make this dataset use the new storage system.
class FlowDataset(InMemoryComplexDataset):

    def __init__(self, root, name, num_points, train_samples, val_samples,
                 load_graph=False, train_orient='default', test_orient='default'):
        self.name = name
        self._num_classes = 2
        self._num_points = num_points
        self._train_samples = train_samples
        self._val_samples = val_samples
        self._train_orient = train_orient
        self._test_orient = test_orient

        super(FlowDataset, self).__init__(root, max_dim=1,
            num_classes=self._num_classes, include_down_adj=True)

        with open(self.processed_paths[0], 'rb') as handle:
            self.__data_list__ = pickle.load(handle)

        self.G = None
        if load_graph:
            with open(self.processed_paths[1], 'rb') as handle:
                self.G = pickle.load(handle)

        self.train_ids = list(range(train_samples))
        self.val_ids = list(range(train_samples, train_samples + val_samples))
        self.test_ids = None

    @property
    def processed_dir(self):
        """This is overwritten, so the simplicial complex data is placed in another folder"""
        return osp.join(self.root,
            f'flow{self._num_points}_orient_{self._train_orient}_{self._test_orient}')

    @property
    def processed_file_names(self):
        return ['{}_complex_list.pkl'.format(self.name), '{}_graph.pkl'.format(self.name)]

    def process(self):
        train, val, G = load_flow_dataset(num_points=self._num_points,
            num_train=self._train_samples, num_test=self._val_samples)

        chains = train + val
        path = self.processed_paths[0]
        with open(path, 'wb') as handle:
            pickle.dump(chains, handle)

        graph_path = self.processed_paths[1]
        with open(graph_path, 'wb') as handle:
            pickle.dump(G, handle)

    @property
    def raw_file_names(self):
        return ""

    def download(self):
        pass
