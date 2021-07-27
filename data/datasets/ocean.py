import pickle
import os.path as osp

from data.datasets import InMemoryComplexDataset
from data.datasets.ocean_utils import load_ocean_dataset


# TODO: Set up a cochain dataset structure or make complex dataset better support cochain-only data.
# TODO: Refactor the dataset to use the latest storage formatting.
class OceanDataset(InMemoryComplexDataset):
    """A real-world dataset for edge-flow classification.

    The dataset is adapted from https://arxiv.org/abs/1807.05044
    """

    def __init__(self, root, name, load_graph=False, train_orient='default',
                 test_orient='default'):
        self.name = name
        self._num_classes = 2
        self._train_orient = train_orient
        self._test_orient = test_orient

        super(OceanDataset, self).__init__(root, max_dim=1,
            num_classes=self._num_classes, include_down_adj=True)

        with open(self.processed_paths[0], 'rb') as handle:
            train = pickle.load(handle)

        with open(self.processed_paths[1], 'rb') as handle:
            val = pickle.load(handle)

        self.__data_list__ = train + val

        self.G = None
        if load_graph:
            with open(self.processed_paths[2], 'rb') as handle:
                self.G = pickle.load(handle)

        self.train_ids = list(range(len(train)))
        self.val_ids = list(range(len(train), len(train) + len(val)))
        self.test_ids = None

    @property
    def processed_dir(self):
        """This is overwritten, so the cellular complex data is placed in another folder"""
        return osp.join(self.root, f'complex_{self._train_orient}_{self._test_orient}')

    @property
    def processed_file_names(self):
        return ['train_{}_complex_list.pkl'.format(self.name),
                'val_{}_complex_list.pkl'.format(self.name),
                '{}_graph.pkl'.format(self.name)]

    def process(self):
        train, val, G = load_ocean_dataset(self._train_orient, self._test_orient)

        train_path = self.processed_paths[0]
        print(f"Saving train dataset to {train_path}")
        with open(train_path, 'wb') as handle:
            pickle.dump(train, handle)

        val_path = self.processed_paths[1]
        print(f"Saving val dataset to {val_path}")
        with open(val_path, 'wb') as handle:
            pickle.dump(val, handle)

        graph_path = self.processed_paths[2]
        with open(graph_path, 'wb') as handle:
            pickle.dump(G, handle)

    def len(self):
        """Override method to make the class work with deprecated stoarage"""
        return len(self.__data_list__)
