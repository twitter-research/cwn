import pickle
import os.path as osp

from data.utils import convert_graph_dataset_with_rings
from data.datasets import InMemoryComplexDataset
from ogb.graphproppred import PygGraphPropPredDataset


class OGBDataset(InMemoryComplexDataset):
    """This is OGB graph-property prediction. This are graph-wise classification tasks."""

    def __init__(self, root, name, max_ring_size, num_classes=2, use_edge_features=False, transform=None,
                 pre_transform=None, pre_filter=None, init_method='sum'):
        self.name = name
        self._max_ring_size = max_ring_size
        self._use_edge_features = use_edge_features
        super(OGBDataset, self).__init__(root, transform, pre_transform, pre_filter,
                                         max_dim=2, num_classes=num_classes,
                                         init_method=init_method, cellular=True)

        self._data_list, idx = self.load_dataset()
        self.train_ids = idx[0]
        self.val_ids = idx[1]
        self.test_ids = idx[2]
        
        # TODO: handle simple vs. full feats
        # if args.feature == 'full':
        #     pass 
        # elif args.feature == 'simple':
        #     print('using simple feature')
        #     # only retain the top two node/edge features
        #     dataset.data.x = dataset.data.x[:,:2]
        #     dataset.data.edge_attr = dataset.data.edge_attr[:,:2]

    @property
    def raw_dir(self):
        directory = super(OGBDataset, self).raw_dir
        return osp.join(directory, self.name, 'processed')
        
    @property
    def raw_file_names(self):
        name = self.name
        # The processed graph files are our raw files.
        return ['geometric_data_processed.pt', 'pre_filter.pt', 'pre_transform.pt']

    @property
    def processed_file_names(self):
        return ['complex_train.pkl', 'complex_val.pkl', 'complex_test.pkl']

    def download(self):
        # Instantiating this will download and process the graph dataset.
        dataset = PygGraphPropPredDataset(self.name, self.root)
        self.num_tasks = dataset.num_tasks

    def load_dataset(self):
        """Load the dataset from here and process it if it doesn't exist"""
        data_list, idx = [], []
        start = 0
        print("Loading dataset from disk...")
        for path in self.processed_paths:
            with open(path, 'rb') as handle:
                data_list.extend(pickle.load(handle))
                idx.append(list(range(start, len(data_list))))
                start = len(data_list)
        return data_list, idx

    def process(self):
        # At this stage, the graph dataset is already downloaded and processed
        
        dataset = PygGraphPropPredDataset(self.name, self.root)
        split_idx = dataset.get_idx_split()
        
        print(f"Processing simplicial complex dataset for {self.name}")
        train_data = dataset[split_idx['train']]
        val_data = dataset[split_idx['valid']]
        test_data = dataset[split_idx['test']]

        print("Converting the train dataset to a cell complex...")
        train_complexes, _, _ = convert_graph_dataset_with_rings(
            train_data,
            max_ring_size=self._max_ring_size,
            include_down_adj=self.include_down_adj,
            init_edges=self._use_edge_features,
            init_rings=False)
        print("Converting the validation dataset to a cell complex...")
        val_complexes, _, _ = convert_graph_dataset_with_rings(
            val_data,
            max_ring_size=self._max_ring_size,
            include_down_adj=self.include_down_adj,
            init_edges=self._use_edge_features,
            init_rings=False)
        print("Converting the test dataset to a cell complex...")
        test_complexes, _, _ = convert_graph_dataset_with_rings(
            test_data,
            max_ring_size=self._max_ring_size,
            include_down_adj=self.include_down_adj,
            init_edges=self._use_edge_features,
            init_rings=False)
        complexes = [train_complexes, val_complexes, test_complexes]

        for i, path in enumerate(self.processed_paths):
            print(f'Saving processed dataset in {path}....')
            with open(path, 'wb') as handle:
                pickle.dump(complexes[i], handle)

    @property
    def processed_dir(self):
        """Overwrite to change name based on edges"""
        directory = super(OGBDataset, self).processed_dir
        suffix1 = f"_{self._max_ring_size}rings" if self._cellular else ""
        suffix2 = "-E" if self._use_edge_features else ""
        return directory + suffix1 + suffix2
