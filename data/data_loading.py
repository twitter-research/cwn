import os
import torch
from torch.utils.data.dataloader import default_collate

from torch_geometric.data import Data, Batch
from torch._six import container_abcs, string_classes, int_classes

from definitions import ROOT_DIR
from data.complex import Chain, ChainBatch, Complex, ComplexBatch
from data.datasets import SRDataset, ClusterDataset, TUDataset
from data.sr_utils import load_sr_dataset

class Collater(object):
    def __init__(self, follow_batch, max_dim=2):
        self.follow_batch = follow_batch
        self.max_dim = max_dim

    def collate(self, batch):
        elem = batch[0]
        if isinstance(elem, Chain):
            return ChainBatch.from_chain_list(batch, self.follow_batch)
        elif isinstance(elem, Complex):
            return ComplexBatch.from_complex_list(batch, self.follow_batch, max_dim=self.max_dim)
        elif isinstance(elem, Data):
            return Batch.from_data_list(batch, self.follow_batch)
        elif isinstance(elem, torch.Tensor):
            return default_collate(batch)
        elif isinstance(elem, float):
            return torch.tensor(batch, dtype=torch.float)
        elif isinstance(elem, int_classes):
            return torch.tensor(batch)
        elif isinstance(elem, string_classes):
            return batch
        elif isinstance(elem, container_abcs.Mapping):
            return {key: self.collate([d[key] for d in batch]) for key in elem}
        elif isinstance(elem, tuple) and hasattr(elem, '_fields'):
            return type(elem)(*(self.collate(s) for s in zip(*batch)))
        elif isinstance(elem, container_abcs.Sequence):
            return [self.collate(s) for s in zip(*batch)]

        raise TypeError('DataLoader found invalid type: {}'.format(type(elem)))

    def __call__(self, batch):
        return self.collate(batch)


class DataLoader(torch.utils.data.DataLoader):
    r"""Data loader which merges data objects to a mini-batch.

    Args:
        dataset (Dataset): The dataset from which to load the data.
        batch_size (int, optional): How many samples per batch to load.
            (default: :obj:`1`)
        shuffle (bool, optional): If set to :obj:`True`, the data will be
            reshuffled at every epoch. (default: :obj:`False`)
        follow_batch (list or tuple, optional): Creates assignment batch
            vectors for each key in the list. (default: :obj:`[]`)
    """
    def __init__(self, dataset, batch_size=1, shuffle=False, follow_batch=[],
                 max_dim=2, **kwargs):

        if "collate_fn" in kwargs:
            del kwargs["collate_fn"]

        # Save for Pytorch Lightning...
        self.follow_batch = follow_batch

        super(DataLoader,
              self).__init__(dataset, batch_size, shuffle,
                             collate_fn=Collater(follow_batch, max_dim), **kwargs)


def load_dataset(name, root=os.path.join(ROOT_DIR, 'datasets'), max_dim=2, fold=0, **kwargs):
    if name.startswith('sr'):
        dataset = SRDataset(os.path.join(root, 'SR_graphs'), name, max_dim=max_dim, num_classes=kwargs['emb_dim'])
    elif name == 'CLUSTER':
        dataset = ClusterDataset(os.path.join(root, 'CLUSTER'), max_dim)
    elif name == 'IMDBBINARY':
        dataset = TUDataset(os.path.join(root, name), name, max_dim=max_dim, num_classes=2, fold=fold, degree_as_tag=True)
    elif name == 'PROTEINS':
        dataset = TUDataset(os.path.join(root, name), name, max_dim=max_dim, num_classes=2, fold=fold, degree_as_tag=False)
    else:
        raise NotImplementedError
    return dataset

def load_sr_graph_dataset(name, emb_dim, root=os.path.join(ROOT_DIR, 'datasets')):
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