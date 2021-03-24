import pickle
import os.path as osp

from data.datasets import InMemoryComplexDataset
from data.datasets.madag_utils import load_madagascar_dataset


# TODO: Set up a chain dataset structure or make complex dataset better support chain-only data.
class MadagascarDataset(InMemoryComplexDataset):

    def __init__(self, root, name, load_graph=False):
        self.name = name
        self._num_classes = 2

        super(MadagascarDataset, self).__init__(root, max_dim=1,
            num_classes=self._num_classes, include_down_adj=True)

        with open(self.processed_paths[0], 'rb') as handle:
            train = pickle.load(handle)

        with open(self.processed_paths[1], 'rb') as handle:
            val = pickle.load(handle)

        self._data_list = train + val

        self.G = None
        if load_graph:
            with open(self.processed_paths[2], 'rb') as handle:
                self.G = pickle.load(handle)

        self.train_ids = list(range(len(train)))
        self.val_ids = list(range(len(train), len(train) + len(val)))
        self.test_ids = None

    @property
    def processed_dir(self):
        """This is overwritten, so the simplicial complex data is placed in another folder"""
        return osp.join(self.root, 'complex')

    @property
    def processed_file_names(self):
        return ['train_{}_complex_list.pkl'.format(self.name),
                'val_{}_complex_list.pkl'.format(self.name),
                '{}_graph.pkl'.format(self.name)]

    def process(self):
        train, val, G = load_madagascar_dataset()

        train_path = self.processed_paths[0]
        with open(train_path, 'wb') as handle:
            pickle.dump(train, handle)

        val_path = self.processed_paths[1]
        with open(val_path, 'wb') as handle:
            pickle.dump(val, handle)

        graph_path = self.processed_paths[2]
        with open(graph_path, 'wb') as handle:
            pickle.dump(G, handle)
