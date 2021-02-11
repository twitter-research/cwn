import os
import torch
from torch.utils.data.dataloader import default_collate

from torch_geometric.data import Data, Batch
from torch._six import container_abcs, string_classes, int_classes

from definitions import ROOT_DIR
from data.complex import Chain, ChainBatch, Complex, ComplexBatch
from data.datasets import SRDataset, ClusterDataset, TUDataset, ComplexDataset, load_sr_graph_dataset

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


def load_dataset(name, root=os.path.join(ROOT_DIR, 'datasets'), max_dim=2, fold=0,
                 init_method='sum', **kwargs) -> ComplexDataset:
    if name.startswith('sr'):
        dataset = SRDataset(os.path.join(root, 'SR_graphs'), name, max_dim=max_dim, num_classes=kwargs['emb_dim'])
    elif name == 'CLUSTER':
        dataset = ClusterDataset(os.path.join(root, 'CLUSTER'), max_dim)
    elif name == 'IMDBBINARY':
        dataset = TUDataset(os.path.join(root, name), name, max_dim=max_dim, num_classes=2, fold=fold, degree_as_tag=True, init_method=init_method)
    elif name == 'IMDBMULTI':
        dataset = TUDataset(os.path.join(root, name), name, max_dim=max_dim, num_classes=3, fold=fold, degree_as_tag=True, init_method=init_method)
    elif name == 'PROTEINS':
        dataset = TUDataset(os.path.join(root, name), name, max_dim=max_dim, num_classes=2,
            fold=fold, degree_as_tag=False, init_method=init_method)
    elif name == 'NCI1':
        dataset = TUDataset(os.path.join(root, name), name, max_dim=max_dim, num_classes=2,
            fold=fold, degree_as_tag=False, init_method=init_method)
    elif name == 'PTC':
        dataset = TUDataset(os.path.join(root, name), name, max_dim=max_dim, num_classes=2,
            fold=fold, degree_as_tag=False, init_method=init_method)
    else:
        raise NotImplementedError
    return dataset


def load_graph_dataset(name, root=os.path.join(ROOT_DIR, 'datasets'), fold=0, **kwargs):
    if name.startswith('sr'):
        data = load_sr_graph_dataset(name, root=root, emb_dim=kwargs['emb_dim'])
    else:
        raise NotImplementedError
    return data
