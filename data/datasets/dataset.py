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
                 max_dim: int = None, num_classes: int = None):
        # These have to be initialised before calling the super class.
        self._max_dim = max_dim
        self._num_features = [None in range(max_dim+1)]

        super(ComplexDataset, self).__init__(root, transform, pre_transform, pre_filter)
        self._num_classes = num_classes

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
        return osp.join(self.root, f'complex_dim{self.max_dim}')

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
        
    def index_select(self, idx):
        """We override this because we store data in a list for now."""
        indices = list(range(self.len()))
        if isinstance(idx, slice):
            indices = indices[idx]
        elif torch.is_tensor(idx):
            if idx.dtype == torch.long:
                if len(idx.shape) == 0:
                    idx = idx.unsqueeze(0)
                return self.index_select(idx.tolist())
            elif idx.dtype == torch.bool or idx.dtype == torch.uint8:
                return self.index_select(
                    idx.nonzero(as_tuple=False).flatten().tolist())
        elif isinstance(idx, list) or isinstance(idx, tuple):
            indices = [indices[i] for i in idx]
        else:
            raise IndexError(
                'Only integers, slices (`:`), list, tuples, and long or bool '
                'tensors are valid indices (got {}).'.format(
                    type(idx).__name__))

        # TODO: PyG abstracts this by having an additional __indices__ property and by returning another
        # dataset rather than a list, as a copy of self but with only the selected __indices__
        data = [self.get(i) for i in indices]
        return data

    def get_idx_split(self):
        """
        In this dataset, if not explicit split is provided,
        we don't distinguish between train, val, test sets.
        """
        train_ids = list(range(self.len())) if self._train_ids is None else self.train_ids
        val_ids = list(range(self.len())) if self._val_ids is None else self.val_ids
        test_ids = list(range(self.len())) if self._test_ids is None else self.test_ids
        idx_split = {
            'train': train_ids,
            'valid': val_ids,
            'test': test_ids}
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
                 pre_filter=None, max_dim: int = None, num_classes: int = None):
        super(InMemoryComplexDataset, self).__init__(root, transform, pre_transform, pre_filter,
                                                     max_dim, num_classes)
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
        dataset._data_list = data_list
        return dataset
