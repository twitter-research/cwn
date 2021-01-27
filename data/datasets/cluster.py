import pickle

from data.datasets import InMemoryComplexDataset
from data.utils import convert_graph_dataset_with_gudhi
from torch_geometric.datasets import GNNBenchmarkDataset


class ClusterDataset(InMemoryComplexDataset):
    """This is the Cluster dataset from the Benchmarking GNNs paper.

    The dataset contains multiple graphs and we have to do node classification on all these graphs.
    """

    def __init__(self, root, name, eval_metric, task_type, max_dim=2, num_classes=2,
                 process_args=None, split='train', **kwargs):
        super(ClusterDataset, self).__init__(root, name, eval_metric, task_type, max_dim=max_dim,
                                             num_classes=num_classes, process_args=process_args,
                                             **kwargs)

        if split == 'train':
            path = self.processed_paths[0]
        elif split == 'val':
            path = self.processed_paths[1]
        elif split == 'test':
            path = self.processed_paths[2]
        else:
            raise ValueError((f'Split {split} found, but expected either '
                              'train, val, trainval or test'))

        self._load(path)

    @property
    def raw_file_names(self):
        name = self.name
        return [f'{name}_train.pt', f'{name}_val.pt', f'{name}_test.pt']

    @property
    def processed_file_names(self):
        return ['complex_train.pkl', 'complex_val.pkl', 'complex_test.pkl']

    def download(self):
        # Instantiating this will download the dataset
        GNNBenchmarkDataset('./datasets/', 'CLUSTER')

    def _load(self, path):
        """Load the dataset from here and process it if it doesn't exist"""
        with open(path, 'rb') as handle:
            self._data += pickle.load(handle)

    def process(self):
        # At this stage, the graph dataset is already downloaded and processed
        train_data = GNNBenchmarkDataset('./datasets/', 'CLUSTER', split='train')
        val_data = GNNBenchmarkDataset('./datasets/', 'CLUSTER', split='val')
        test_data = GNNBenchmarkDataset('./datasets/', 'CLUSTER', split='test')

        train_complexes, _, _ = convert_graph_dataset_with_gudhi(train_data,
                                                                 expansion_dim=self.max_dim)
        val_complexes, _, _ = convert_graph_dataset_with_gudhi(val_data, expansion_dim=self.max_dim)
        test_complexes, _, _ = convert_graph_dataset_with_gudhi(test_data,
                                                                expansion_dim=self.max_dim)
        complexes = [train_complexes, val_complexes, test_complexes]

        for i, path in enumerate(self.processed_paths):
            with open(path, 'wb') as handle:
                pickle.dump(complexes[i], handle)

    def _set_task(self, task_type, num_classes):
        super(ClusterDataset, self)._set_task(task_type, num_classes)
        self.num_classes = len(self)
