import os.path as osp
import numpy as np
import torch

from data.datasets import InMemoryComplexDataset
from data.utils import convert_graph_dataset_with_rings
from torch_geometric.datasets import GNNBenchmarkDataset
from torch_geometric.utils import remove_self_loops


class CSLDataset(InMemoryComplexDataset):
    """This is the CSL (Circular Skip Link) dataset from the Benchmarking GNNs paper.

    The dataset contains 10 isomorphism classes of regular graphs that must be classified.
    """

    def __init__(self, root, transform=None,
                 pre_transform=None, pre_filter=None, max_ring_size=6, fold=0, init_method='sum',
                 n_jobs=2):
        self.name = 'CSL'
        self._max_ring_size = max_ring_size
        self._n_jobs = n_jobs
        super(CSLDataset, self).__init__(root, transform, pre_transform, pre_filter,
                                         max_dim=2, cellular=True, init_method=init_method,
                                         num_classes=10)

        assert 0 <= fold <= 4
        self.fold = fold

        self.data, self.slices = self.load_dataset()

        self.num_node_type = 1
        self.num_edge_type = 1

        # These cross-validation splits have been taken from
        # https://github.com/graphdeeplearning/benchmarking-gnns/tree/master/data/CSL
        train_filename = osp.join(self.root, 'splits', 'CSL_train.txt')
        valid_filename = osp.join(self.root, 'splits', 'CSL_val.txt')
        test_filename = osp.join(self.root, 'splits', 'CSL_test.txt')

        self.train_ids = np.loadtxt(train_filename, dtype=int, delimiter=',')[fold].tolist()
        self.val_ids = np.loadtxt(valid_filename, dtype=int, delimiter=',')[fold].tolist()
        self.test_ids = np.loadtxt(test_filename, dtype=int, delimiter=',')[fold].tolist()

        # Make sure the split ratios are as expected (3:1:1)
        assert len(self.train_ids) == 3 * len(self.test_ids)
        assert len(self.val_ids) == len(self.test_ids)
        # Check all splits contain numbers that are smaller than the total number of graphs
        assert max(self.train_ids) < 150
        assert max(self.val_ids) < 150
        assert max(self.test_ids) < 150

    @property
    def raw_file_names(self):
        return ['data.pt']

    @property
    def processed_file_names(self):
        return ['complexes.pt']

    def download(self):
        # Instantiating this will download and process the graph dataset.
        GNNBenchmarkDataset(self.raw_dir, 'CSL')

    def load_dataset(self):
        """Load the dataset from here and process it if it doesn't exist"""
        print("Loading dataset from disk...")
        data, slices = torch.load(self.processed_paths[0])
        return data, slices

    def process(self):
        # At this stage, the graph dataset is already downloaded and processed
        print(f"Processing cell complex dataset for {self.name}")
        # This dataset has no train / val / test splits and we must use cross-validation
        data = GNNBenchmarkDataset(self.raw_dir, 'CSL')
        assert len(data) == 150

        # Check that indeed there are no features
        assert data[0].x is None
        assert data[0].edge_attr is None

        print("Populating graph with features")
        # Initialise everything with zero as in the Benchmarking GNNs code
        # https://github.com/graphdeeplearning/benchmarking-gnns/blob/ef8bd8c7d2c87948bc1bdd44099a52036e715cd0/data/CSL.py#L144
        new_data = []
        for i, datum in enumerate(data):
            edge_index = datum.edge_index
            num_nodes = datum.num_nodes
            # Make sure we have no self-loops in this dataset
            edge_index, _ = remove_self_loops(edge_index)
            num_edges = edge_index.size(1)

            vx = torch.zeros((num_nodes, 1), dtype=torch.long)
            edge_attr = torch.zeros(num_edges, dtype=torch.long)
            setattr(datum, 'edge_index', edge_index)
            setattr(datum, 'x', vx)
            setattr(datum, 'edge_attr', edge_attr)
            new_data.append(datum)

        assert new_data[0].x is not None
        assert new_data[0].edge_attr is not None

        print("Converting the train dataset to a cell complex...")
        complexes, _, _ = convert_graph_dataset_with_rings(
            new_data,
            max_ring_size=self._max_ring_size,
            include_down_adj=False,
            init_edges=True,
            init_rings=False,
            n_jobs=self._n_jobs)

        path = self.processed_paths[0]
        print(f'Saving processed dataset in {path}....')
        torch.save(self.collate(complexes, 2), path)

    @property
    def processed_dir(self):
        """Overwrite to change name based on edges"""
        directory = super(CSLDataset, self).processed_dir
        suffix1 = f"_{self._max_ring_size}rings" if self._cellular else ""
        return directory + suffix1
