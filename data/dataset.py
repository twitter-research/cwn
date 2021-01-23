import copy
import os
import torch

from data.sr_utils import load_sr_dataset
from data.utils import compute_clique_complex_with_gudhi
    
__max_metrics__ = ['accuracy', 'roc_auc', 'average_precision']
__min_metrics__ = ['mae', 'mse']
__other_metrics__ = ['isomorphism']
__task_types__ = ['regression', 'classification', 'isomorphism']
    
class ComplexDataset(object):
    """
        Base class for simplicial complex datasets that can be held in main memory.
        Args:
            root (str): root folder where raw data files are found.     
            name (str): dataset name.
            eval_metric (str): the predefined evaluation metric for the dataset.
            task_type (str): the task solved, either 'regression', 'classification' or 'isomorphism'.
            num_classes (:obj:`int`, optional): number of output classes/tasks.
        Attributes:
            root (str): root folder where raw data files are found.     
            name (str): dataset name.
            eval_metric (str): the predefined evaluation metric for the dataset.
            task_type (str): the task solved, either 'regression', 'classification' or 'isomorphism'.
            num_classes (:obj:`int`, optional): number of output classes/tasks.
            maximize (:obj: `bool`): whether the `eval_metric` is to be maximized, automatically inferred.
    """
    def __init__(self, root, name, eval_metric, task_type, num_classes=2, **kwargs):
        self.root = root
        self.name = name
        self._set_metric(eval_metric)
        self._data = []
        self._load_data()
        self._set_task(task_type, num_classes)
        self._train_ids = None
        self._val_ids = None
        self._test_ids = None
        for k in kwargs.keys():
            if k == 'train_ids':
                self._train_ids = kwargs[k]
            elif k == 'val_ids':
                self._val_ids = kwargs[k]
            elif k == 'test_ids':
                self._test_ids = kwargs[k]
            else:
                self.__setattr__(k, kwargs[k])

    def _set_metric(self, eval_metric):
        """
            Internal. Sets the `eval_metric` and `maximize` attributes.
        """
        if eval_metric in __max_metrics__:
            self.maximize = True
        elif eval_metric in __min_metrics__:
            self.maximize = False
        elif eval_metric in __other_metrics__:
            self.maximize = None
        else:
            raise NotImplementedError('Metric {} not yet supported.'.format(eval_metric))
        self.eval_metric = eval_metric
        
    def _set_task(self, task_type, num_classes):
        """
            Internal. Sets the `task_type` and `num_classes` attributes.
        """
        if task_type not in __task_types__:
            raise ValueError('Task type can only be {}, {} or {}, found {}.'.format(*__task_types__+[task_type]))
        self.task_type = task_type
        if self.task_type == 'classification':
            if num_classes <= 1:
                raise ValueError('The number of classes must be greater or equal than 2, found {}.'.format(num_classes))
        self.num_classes = num_classes
    
    def _load_data(self):
        """
            Internal. Loads raw data from disk and prepares the complex samples, if needed.
            The implementation of this method is required in concrete subclasses.
        """
        raise NotImplementedError()
        
    def get_idx_split(self):
        """
            Returns the split dictionary.
            The implementation of this method is required in concrete subclasses.
        """
        raise NotImplementedError()
        
    def __len__(self):
        """
            Returns the number of examples in the dataset.
        """
        return self.len()
        
    def __getitem__(self, idx):
        """
            Gets the data object at index :obj:`idx`.
            In case :obj:`idx` is a slicing object, *e.g.*, :obj:`[2:5]`, a list, a
            tuple, a  LongTensor or a BoolTensor, will return a subset of the
            dataset at the specified indices.
        """
        if isinstance(idx, int):
            data = self.get(idx)
            return data
        else:
            return self.index_select(idx)

    def len(self):
        return len(self._data)
    
    def get(self, idx):
        # TODO: do we need copy, deep copy, or simple reassignment?
        return copy.copy(self._data[idx])

    def index_select(self, idx):
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

        data = [self.get(i) for i in indices]
        return data

    def __repr__(self):  # pragma: no cover
        return f'{self.__class__.__name__}({len(self)})'
    
class SRDataset(ComplexDataset):
    
    def __init__(self, root, name, eval_metric, task_type, num_classes=2, exp_dim=4, **kwargs):
        self.exp_dim = exp_dim
        super(SRDataset, self).__init__(root, name, eval_metric, task_type, num_classes=num_classes, **kwargs)
        
    def get_idx_split(self):
        # In this dataset, if not explicit split is provided, we don't distinguish between train, val, test sets.
        train_ids = list(range(self.len())) if self._train_ids is None else self._train_ids
        val_ids = list(range(self.len())) if self._val_ids is None else self._val_ids
        test_ids = list(range(self.len())) if self._test_ids is None else self._test_ids
        idx_split = {
            'train': train_ids,
            'valid': val_ids,
            'test': test_ids}
        return idx_split
                        
    def _load_data(self):
        data = load_sr_dataset(os.path.join(self.root, self.name+'.g6'))
        complexes = list()
        for datum in data:
            edge_index, num_nodes = datum
            x = torch.ones(num_nodes, 1, dtype=torch.float32)
            y = torch.ones(1, dtype=torch.long)
            complex = compute_clique_complex_with_gudhi(x, edge_index, num_nodes, expansion_dim=self.exp_dim, y=y)
            complexes.append(complex)
        self._data = complexes

    def _set_task(self, task_type, num_classes):
        super(SRDataset, self)._set_task(task_type, num_classes)
        self.num_classes = len(self)