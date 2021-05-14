import os
import torch
import pickle

from data.sr_utils import load_sr_dataset
from data.utils import compute_clique_complex_with_gudhi, compute_ring_2complex
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

def load_sr_graph_dataset(name, root=os.path.join(ROOT_DIR, 'datasets'), emb_dim=16):
    raw_dir = os.path.join(root, 'SR_graphs', 'raw')
    load_from = os.path.join(raw_dir, '{}.g6'.format(name))
    data = load_sr_dataset(load_from)
    y = torch.ones(emb_dim, dtype=torch.long)
    graphs = list()
    for datum in data:
        edge_index, num_nodes = datum
        x = torch.ones(num_nodes, 1, dtype=torch.float32)
        graph = Data(x=x, edge_index=edge_index, y=y, num_nodes=num_nodes)
        graphs.append(graph) 
    train_ids = list(range(len(graphs)))
    val_ids = list(range(len(graphs)))
    test_ids = list(range(len(graphs)))
    return graphs, train_ids, val_ids, test_ids
            
class SRDataset(InMemoryComplexDataset):

    def __init__(self, root, name, max_dim=2, num_classes=2,
                 train_ids=None, val_ids=None, test_ids=None, include_down_adj=False, max_ring_size=None):
        self.name = name
        self._num_classes = num_classes
        assert max_ring_size is None or max_ring_size > 3
        self._max_ring_size = max_ring_size
        cellular = (max_ring_size is not None)
        if cellular:
            assert max_dim == 2
        super(SRDataset, self).__init__(root, max_dim=max_dim, num_classes=num_classes,
            include_down_adj=include_down_adj, cellular=cellular)
        
        self.data, self.slices = torch.load(self.processed_paths[0])
            
        self.train_ids = list(range(self.len())) if train_ids is None else train_ids
        self.val_ids = list(range(self.len())) if val_ids is None else val_ids
        self.test_ids = list(range(self.len())) if test_ids is None else test_ids
        
    @property
    def processed_dir(self):
        """This is overwritten, so the simplicial complex data is placed in another folder"""
        directory = super(SRDataset, self).processed_dir
        suffix = f"_{self._max_ring_size}rings" if self._cellular else ""
        suffix += f"_down_adj" if self.include_down_adj else ""
        return directory + suffix

    @property
    def processed_file_names(self):
        return ['{}_complex_list.pt'.format(self.name)]       

    def process(self):
        data = load_sr_dataset(os.path.join(self.raw_dir, self.name + '.g6'))
        exp_dim = self.max_dim

        num_features = [None for _ in range(exp_dim+1)]
        complexes = list()
        max_dim = -1
        for datum in data:
            edge_index, num_nodes = datum
            x = torch.ones(num_nodes, 1, dtype=torch.float32)
            
            if self._cellular:
                complex = compute_ring_2complex(x, edge_index, None,
                          num_nodes, max_k=self._max_ring_size,
                          include_down_adj=self.include_down_adj,
                          init_edges=True, init_rings=True)
            else:
                complex = compute_clique_complex_with_gudhi(x, edge_index, num_nodes,
                    expansion_dim=exp_dim, include_down_adj=self.include_down_adj)
                
            if complex.dimension > max_dim:
                max_dim = complex.dimension
            if self._max_ring_size is not None:
                assert max_dim <= 2
            for dim in range(complex.dimension + 1):
                if num_features[dim] is None:
                    num_features[dim] = complex.chains[dim].num_features
                else:
                    assert num_features[dim] == complex.chains[dim].num_features
            complexes.append(complex)
        if max_dim != self.max_dim:
            self.max_dim = max_dim
            makedirs(self.processed_dir)
            
        # Here we add dummy labels to each of the complex.
        # They are not used in isomorphism testing; rather each complex
        # is embedded in a self.num_classes-dimensional space and pairwise
        # distances are computed and inspected.
        # y = torch.ones(self.num_classes, dtype=torch.long)
        # for complex in complexes:
        #     complex.y = y
            
        # Now we save in opt format.
        path = self.processed_paths[0]
        torch.save(self.collate(complexes, self.max_dim), path)
