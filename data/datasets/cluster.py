import pickle

from data.datasets import InMemoryComplexDataset
from data.utils import convert_graph_dataset_with_gudhi
from torch_geometric.datasets import GNNBenchmarkDataset


class ClusterDataset(InMemoryComplexDataset):
    """This is the Cluster dataset from the Benchmarking GNNs paper.

    The dataset contains multiple graphs and we have to do node classification on all these graphs.
    """

    def __init__(self, root, transform=None,
                 pre_transform=None, pre_filter=None, max_dim=2):
        self.name = 'CLUSTER'
        super(ClusterDataset, self).__init__(root, transform, pre_transform, pre_filter,
                                             max_dim=max_dim)

        self.max_dim = max_dim

        self._data_list, idx = self.load_dataset()
        self.train_ids = idx[0]
        self.val_ids = idx[1]
        self.test_ids = idx[2]

    @property
    def raw_file_names(self):
        name = self.name
        # The processed graph files are our raw files.
        # I've obtained this from inside the GNNBenchmarkDataset class
        return [f'{name}_train.pt', f'{name}_val.pt', f'{name}_test.pt']

    @property
    def processed_file_names(self):
        return ['complex_train.pkl', 'complex_val.pkl', 'complex_test.pkl']

    def download(self):
        # Instantiating this will download and process the graph dataset.
        GNNBenchmarkDataset('./datasets/', 'CLUSTER')

    def load_dataset(self):
        """Load the dataset from here and process it if it doesn't exist"""
        data_list, idx = [], []
        start = 0
        for path in self.processed_paths:
            with open(path, 'rb') as handle:
                data_list.extend(pickle.load(handle))
                idx.append(list(range(start, len(data_list))))
                start = len(data_list)
        return data_list, idx

    def process(self):
        # At this stage, the graph dataset is already downloaded and processed
        print(f"Processing cellular complex dataset for {self.name}")
        train_data = GNNBenchmarkDataset('./datasets/', 'CLUSTER', split='train')
        val_data = GNNBenchmarkDataset('./datasets/', 'CLUSTER', split='val')
        test_data = GNNBenchmarkDataset('./datasets/', 'CLUSTER', split='test')

        # For testing
        # train_data = list(train_data)[:3]
        # val_data = list(val_data)[:3]
        # test_data = list(test_data)[:3]

        print("Converting the train dataset with gudhi...")
        train_complexes, _, _ = convert_graph_dataset_with_gudhi(train_data,
            expansion_dim=self.max_dim, include_down_adj=self.include_down_adj)
        print("Converting the validation dataset with gudhi...")
        val_complexes, _, _ = convert_graph_dataset_with_gudhi(val_data, expansion_dim=self.max_dim, include_down_adj=self.include_down_adj)
        print("Converting the test dataset with gudhi...")
        test_complexes, _, _ = convert_graph_dataset_with_gudhi(test_data,
                                                                expansion_dim=self.max_dim)
        complexes = [train_complexes, val_complexes, test_complexes]

        for i, path in enumerate(self.processed_paths):
            with open(path, 'wb') as handle:
                pickle.dump(complexes[i], handle)
