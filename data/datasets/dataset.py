import copy
import os
import re
from abc import ABC

import torch
import pickle
import errno
import logging
import os.path as osp

from torch_geometric.data import Dataset


def __repr__(obj):
    if obj is None:
        return 'None'
    return re.sub('(<.*?)\\s.*(>)', r'\1\2', obj.__repr__())


class ComplexDataset(Dataset, ABC):
    """
        Base class for simplicial complex datasets.
        Args:
            root (str): root folder where raw data files are found.     
            name (str): dataset name.
            eval_metric (str): the predefined evaluation metric for the dataset.
            task_type (str): the task solved, either 'regression', 'classification' or 'isomorphism'.
            max_dim (:obj: int, optional): max allowed dimension for chains in the dataset.
            num_classes (:obj:`int`, optional): number of output classes/tasks.
            process_args (:obj:`dict`, optional): args for data processing.
    
        Attributes:
            root (str): root folder where raw data files are found.     
            name (str): dataset name.
            eval_metric (str): the predefined evaluation metric for the dataset.
            task_type (str): the task solved, either 'regression', 'classification' or 'isomorphism'.
            max_dim (:obj: int): max allowed dimension for chains in the dataset.
            num_classes (:obj:`int`): number of output classes/tasks.
            maximize (:obj: `bool`): whether the `eval_metric` is to be maximized, automatically inferred.
            process_args (:obj:`dict`): args for data processing.
    """

    def __init__(self, root=None, transform=None, pre_transform=None, pre_filter=None):
        super(ComplexDataset, self).__init__(root, transform, pre_transform, pre_filter)

    @property
    def processed_dir(self):
        """This is overwritten, so the simplicial complex data is placed in another folder"""
        return osp.join(self.root, 'complex')

    def num_features_in_dim(self, dim):
        if dim > self.max_dim:
            raise ValueError('`dim` {} larger than max allowed dimension {}.'.format(dim, self.max_dim))
        return self[0].chains[dim].num_features
        
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
        data = copy.copy([self.get(i) for i in indices])
        return data


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
                 pre_filter=None):
        super(InMemoryComplexDataset, self).__init__(root, transform, pre_transform, pre_filter)
        self.__data_list__ = None
                
    def len(self):
        return len(self.__data_list__)
    
    def get(self, idx):
        return copy.copy(self.__data_list__[idx])

    def copy(self, idx=None):
        if idx is None:
            data_list = [self.get(i) for i in range(len(self))]
        else:
            data_list = [self.get(i) for i in idx]
        dataset = copy.copy(self)
        dataset.__data_list__ = data_list
        return dataset
