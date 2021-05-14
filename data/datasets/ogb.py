import torch
import os.path as osp

from data.utils import convert_graph_dataset_with_rings
from data.datasets import InMemoryComplexDataset
from ogb.graphproppred import PygGraphPropPredDataset


class OGBDataset(InMemoryComplexDataset):
    """This is OGB graph-property prediction. This are graph-wise classification tasks."""

    def __init__(self, root, name, max_ring_size, num_classes=2, use_edge_features=False, transform=None,
                 pre_transform=None, pre_filter=None, init_method='sum', full=True):
        self.name = name
        self._max_ring_size = max_ring_size
        self._use_edge_features = use_edge_features
        self._full = full
        super(OGBDataset, self).__init__(root, transform, pre_transform, pre_filter,
                                         max_dim=2, num_classes=num_classes,
                                         init_method=init_method, cellular=True)
        self.data, self.slices, idx, self.num_tasks = self.load_dataset()
        self.train_ids = idx['train']
        self.val_ids = idx['valid']
        self.test_ids = idx['test']

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
        return [f'{self.name}_complex.pt', f'{self.name}_idx.pt', f'{self.name}_tasks.pt']
    
    @property
    def processed_dir(self):
        """Overwrite to change name based on edge and full feats"""
        directory = super(OGBDataset, self).processed_dir
        suffix1 = f"_{self._max_ring_size}rings" if self._cellular else ""
        suffix2 = "-E" if self._use_edge_features else ""
        suffix3 = "" if self._full else "-S"
        return directory + suffix1 + suffix2 + suffix3

    def download(self):
        # Instantiating this will download and process the graph dataset.
        dataset = PygGraphPropPredDataset(self.name, self.root)

    def load_dataset(self):
        """Load the dataset from here and process it if it doesn't exist"""
        print("Loading dataset from disk...")
        data, slices = torch.load(self.processed_paths[0])
        idx = torch.load(self.processed_paths[1])
        tasks = torch.load(self.processed_paths[2])
        return data, slices, idx, tasks

    def process(self):
        
        # At this stage, the graph dataset is already downloaded and processed
        dataset = PygGraphPropPredDataset(self.name, self.root)
        split_idx = dataset.get_idx_split()
        if not self._full:  # Only retain the top two node/edge features
            print('Using simple features')
            dataset.data.x = dataset.data.x[:,:2]
            dataset.data.edge_attr = dataset.data.edge_attr[:,:2]
        
        print(f"Converting the {self.name} dataset to a cell complex...")
        complexes, _, _ = convert_graph_dataset_with_rings(
            dataset,
            max_ring_size=self._max_ring_size,
            include_down_adj=self.include_down_adj,
            init_edges=self._use_edge_features,
            init_rings=False)
        
        print(f'Saving processed dataset in {self.processed_paths[0]}...')
        torch.save(self.collate(complexes, self.max_dim), self.processed_paths[0])
        
        print(f'Saving idx in {self.processed_paths[1]}...')
        torch.save(split_idx, self.processed_paths[1])
        
        print(f'Saving num_tasks in {self.processed_paths[2]}...')
        torch.save(dataset.num_tasks, self.processed_paths[2])
