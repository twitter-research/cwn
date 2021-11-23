import os
import torch
import pickle

from data.sr_utils import load_sr_dataset
from data.utils import compute_clique_complex_with_gudhi, compute_ring_2complex
from data.utils import convert_graph_dataset_with_rings, convert_graph_dataset_with_gudhi
from data.datasets import InMemoryComplexDataset
from definitions import ROOT_DIR
from torch_geometric.data import Data

import os.path as osp
import errno


def makedirs(path):
    try:
        os.makedirs(osp.expanduser(osp.normpath(path)))
    except OSError as e:
        if e.errno != errno.EEXIST and osp.isdir(path):
            raise e


def load_sr_graph_dataset(name, root=os.path.join(ROOT_DIR, 'datasets'), prefer_pkl=False):
    raw_dir = os.path.join(root, 'SR_graphs', 'raw')
    load_from = os.path.join(raw_dir, '{}.g6'.format(name))
    load_from_pkl = os.path.join(raw_dir, '{}.pkl'.format(name))
    if prefer_pkl and osp.exists(load_from_pkl):
        print(f"Loading SR graph {name} from pickle dump...")
        with open(load_from_pkl, 'rb') as handle:
            data = pickle.load(handle)
    else:
        data = load_sr_dataset(load_from)
    graphs = list()
    for datum in data:
        edge_index, num_nodes = datum
        x = torch.ones(num_nodes, 1, dtype=torch.float32)
        graph = Data(x=x, edge_index=edge_index, y=None, edge_attr=None, num_nodes=num_nodes)
        graphs.append(graph) 
    train_ids = list(range(len(graphs)))
    val_ids = list(range(len(graphs)))
    test_ids = list(range(len(graphs)))
    return graphs, train_ids, val_ids, test_ids


class SRDataset(InMemoryComplexDataset):
    """A dataset of complexes obtained by lifting Strongly Regular graphs."""

    def __init__(self, root, name, max_dim=2, num_classes=16, train_ids=None, val_ids=None, test_ids=None, 
                 include_down_adj=False, max_ring_size=None, n_jobs=2, init_method='sum'):
        self.name = name
        self._num_classes = num_classes
        self._n_jobs = n_jobs
        assert max_ring_size is None or max_ring_size > 3
        self._max_ring_size = max_ring_size
        cellular = (max_ring_size is not None)
        if cellular:
            assert max_dim == 2
        super(SRDataset, self).__init__(root, max_dim=max_dim, num_classes=num_classes,
            include_down_adj=include_down_adj, cellular=cellular, init_method=init_method)
        
        self.data, self.slices = torch.load(self.processed_paths[0])
            
        self.train_ids = list(range(self.len())) if train_ids is None else train_ids
        self.val_ids = list(range(self.len())) if val_ids is None else val_ids
        self.test_ids = list(range(self.len())) if test_ids is None else test_ids
        
    @property
    def processed_dir(self):
        """This is overwritten, so the cellular complex data is placed in another folder"""
        directory = super(SRDataset, self).processed_dir
        suffix = f"_{self._max_ring_size}rings" if self._cellular else ""
        suffix += f"_down_adj" if self.include_down_adj else ""
        return directory + suffix

    @property
    def processed_file_names(self):
        return ['{}_complex_list.pt'.format(self.name)]       

    def process(self):
        
        graphs, _, _, _ = load_sr_graph_dataset(self.name, prefer_pkl=True)
        exp_dim = self.max_dim
        if self._cellular:
            print(f"Converting the {self.name} dataset to a cell complex...")
            complexes, max_dim, num_features = convert_graph_dataset_with_rings(
                graphs,
                max_ring_size=self._max_ring_size,
                include_down_adj=self.include_down_adj,
                init_method=self._init_method,
                init_edges=True,
                init_rings=True,
                n_jobs=self._n_jobs)
        else:
            print(f"Converting the {self.name} dataset with gudhi...")
            complexes, max_dim, num_features = convert_graph_dataset_with_gudhi(
                graphs,
                expansion_dim=exp_dim,                                               
                include_down_adj=self.include_down_adj,                    
                init_method=self._init_method)
        
        if self._max_ring_size is not None:
            assert max_dim <= 2
        if max_dim != self.max_dim:
            self.max_dim = max_dim
            makedirs(self.processed_dir)
        
        # Now we save in opt format.
        path = self.processed_paths[0]
        torch.save(self.collate(complexes, self.max_dim), path)
