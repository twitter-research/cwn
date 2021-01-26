import pickle

from data.datasets import InMemoryComplexDataset
from data.utils import convert_graph_dataset_with_gudhi
from torch_geometric.datasets import GNNBenchmarkDataset


class ClusterDataset(InMemoryComplexDataset):
    """This is the Cluster dataset from the Benchmarking GNNs paper.

    The dataset contains multiple graphs and we have to do node classification on all these graphs.
    """

    def __init__(self, root, split='train', transform=None,
                 pre_transform=None, pre_filter=None, max_dim=2):
        self.name = 'CLUSTER'
        self.max_dim = max_dim

        super(ClusterDataset, self).__init__(root, transform, pre_transform, pre_filter)

        if split == 'train':
            path = self.processed_paths[0]
        elif split == 'val':
            path = self.processed_paths[1]
        elif split == 'test':
            path = self.processed_paths[2]
        else:
            raise ValueError((f'Split {split} found, but expected either '
                              'train, val, trainval or test'))

        self.__data_list__ = self.load_from_path(path)

    @property
    def raw_file_names(self):
        name = self.name
        # The processed graph filers are our raw files.
        # I've obtained this from inside the GNNBenchmarkDataset class
        return [f'{name}_train.pt', f'{name}_val.pt', f'{name}_test.pt']

    @property
    def processed_file_names(self):
        return ['complex_train.pkl', 'complex_val.pkl', 'complex_test.pkl']

    def download(self):
        # Instantiating this will download and process the graph dataset.
        GNNBenchmarkDataset('./datasets/', 'CLUSTER')

    def load_from_path(self, path):
        """Load the dataset from here and process it if it doesn't exist"""
        with open(path, 'rb') as handle:
            return pickle.load(handle)

    def process(self):
        # At this stage, the graph dataset is already downloaded and processed
        print(f"Processing simplicial complex dataset for {self.name}")
        train_data = GNNBenchmarkDataset('./datasets/', 'CLUSTER', split='train')
        val_data = GNNBenchmarkDataset('./datasets/', 'CLUSTER', split='val')
        test_data = GNNBenchmarkDataset('./datasets/', 'CLUSTER', split='test')

        print("Converting the train dataset with gudhi...")
        train_complexes, _, _ = convert_graph_dataset_with_gudhi(train_data,
                                                                 expansion_dim=self.max_dim)
        print("Converting the validation dataset with gudhi...")
        val_complexes, _, _ = convert_graph_dataset_with_gudhi(val_data, expansion_dim=self.max_dim)
        print("Converting the test dataset with gudhi...")
        test_complexes, _, _ = convert_graph_dataset_with_gudhi(test_data,
                                                                expansion_dim=self.max_dim)
        complexes = [train_complexes, val_complexes, test_complexes]

        for i, path in enumerate(self.processed_paths):
            with open(path, 'wb') as handle:
                pickle.dump(complexes[i], handle)
