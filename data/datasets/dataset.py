import copy
import re
from abc import ABC

import torch
import os.path as osp

from torch_geometric.data import Dataset


def __repr__(obj):
    if obj is None:
        return 'None'
    return re.sub('(<.*?)\\s.*(>)', r'\1\2', obj.__repr__())


class ComplexDataset(Dataset, ABC):
    """
        Base class for simplicial complex datasets.
    """

    def __init__(self, root=None, transform=None, pre_transform=None, pre_filter=None,
                 max_dim: int = None, num_classes: int = None, init_method: str = 'sum',
                 cellular: bool = False):
        # These have to be initialised before calling the super class.
        self._max_dim = max_dim
        self._num_features = [None for _ in range(max_dim+1)]
        self._init_method = init_method
        self._cellular = cellular

        super(ComplexDataset, self).__init__(root, transform, pre_transform, pre_filter)
        self._num_classes = num_classes
        self.train_ids = None
        self.val_ids = None
        self.test_ids = None

    @property
    def max_dim(self):
        return self._max_dim

    @max_dim.setter
    def max_dim(self, value):
        self._max_dim = value

    @property
    def num_classes(self):
        return self._num_classes

    @property
    def processed_dir(self):
        """This is overwritten, so the simplicial complex data is placed in another folder"""
        prefix = "cell_" if self._cellular else ""
        return osp.join(self.root, f'{prefix}complex_dim{self.max_dim}_{self._init_method}')

    def num_features_in_dim(self, dim):
        if dim > self.max_dim:
            raise ValueError('`dim` {} larger than max allowed dimension {}.'.format(dim, self.max_dim))
        if self._num_features[dim] is None:
            self._look_up_num_features()
        return self._num_features[dim]

    def _look_up_num_features(self):
        for complex in self:
            for dim in range(complex.dimension + 1):
                if self._num_features[dim] is None:
                    self._num_features[dim] = complex.chains[dim].num_features
                else:
                    assert self._num_features[dim] == complex.chains[dim].num_features

    def get_idx_split(self):
        idx_split = {
            'train': self.train_ids,
            'valid': self.val_ids,
            'test': self.test_ids}
        return idx_split


class InMemoryComplexDataset(ComplexDataset):

    @property
    def raw_file_names(self):
        r"""The name of the files to find in the :obj:`self.raw_dir` folder in
        order to skip the download."""
        raise NotImplementedError

    @property
    def processed_file_names(self):
        r"""The name of the files to find in the :obj:`self.processed_dir`
        folder in order to skip the processing."""
        raise NotImplementedError

    def download(self):
        r"""Downloads the dataset to the :obj:`self.raw_dir` folder."""
        raise NotImplementedError

    def process(self):
        r"""Processes the dataset to the :obj:`self.processed_dir` folder."""
        raise NotImplementedError
    
    def __init__(self, root=None, transform=None, pre_transform=None,
                 pre_filter=None, max_dim: int = None, num_classes: int = None,
                 include_down_adj=False, init_method=None, cellular: bool = False):
        self.include_down_adj = include_down_adj
        super(InMemoryComplexDataset, self).__init__(root, transform, pre_transform, pre_filter,
                                                     max_dim, num_classes, init_method=init_method,
                                                     cellular=cellular)
        self._data_list = None
                
    def len(self):
        return len(self._data_list)
    
    def get(self, idx):
        return copy.copy(self._data_list[idx])

    def copy(self, idx=None):
        if idx is None:
            data_list = [self.get(i) for i in range(len(self))]
        else:
            data_list = [self.get(i) for i in idx]
        dataset = copy.copy(self)
        dataset.__indices__ = None
        dataset._data_list = data_list
        return dataset
