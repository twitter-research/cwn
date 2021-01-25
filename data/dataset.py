import copy
import os
import torch
import pickle
import errno
import logging

from data.sr_utils import load_sr_dataset
from data.utils import compute_clique_complex_with_gudhi
    
__max_metrics__ = ['accuracy', 'roc_auc', 'average_precision']
__min_metrics__ = ['mae', 'mse']
__other_metrics__ = ['isomorphism']
__task_types__ = ['regression', 'classification', 'isomorphism']

def files_exist(files):
    return len(files) != 0 and all(os.path.exists(f) for f in files)

def to_list(x):
    if not isinstance(x, (tuple, list)):
        x = [x]
    return x

def makedirs(path):
    try:
        os.makedirs(os.path.expanduser(os.path.normpath(path)))
    except OSError as e:
        if e.errno != errno.EEXIST and osp.isdir(path):
            raise e
    
class ComplexDataset(torch.utils.data.Dataset):
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
    def __init__(self, root, name, eval_metric, task_type, max_dim=2, num_classes=2, process_args={}, **kwargs):
        self.root = root
        self.name = name
        self.max_dim = max_dim
        self._set_metric(eval_metric)
        self._num_features = {dim: None for dim in range(max_dim+1)}
        self.process_args = process_args
        if 'process' in self.__class__.__dict__.keys():
            self._process()
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
                
    @property
    def processed_file_names(self):
        """
            The name of the files to find in the :obj:`self.processed_dir`
            folder in order to skip the processing.
        """
        raise NotImplementedError()
        
    @property
    def raw_dir(self):
        return os.path.join(self.root, 'raw')

    @property
    def processed_dir(self):
        return os.path.join(self.root, 'processed')

    @property
    def processed_paths(self):
        """
            The filepaths to find in the :obj:`self.processed_dir`
            folder in order to skip the processing.
        """
        files = to_list(self.processed_file_names)
        return [os.path.join(self.processed_dir, f) for f in files]
    
    def _same_process_args(self, other):
        current = self.process_args
        current_keys = set(current.keys())
        other_keys = set(other.keys())
        if current_keys != other_keys:
            return False
        for key in current_keys:
            if current[key] != other[key]:
                return False
        return True
    
    def _process(self):
        """
            Internal. Loads raw data from disk and prepares the complex samples, if needed.
        """
        f = os.path.join(self.processed_dir, '{}_process_args.pkl'.format(self.name))
        if os.path.exists(f):
            with open(f, 'rb') as handle:
                other = pickle.load(handle)
                if not self._same_process_args(other):
                    logging.warning(
                        'The `process` arguments differs from the one used in '
                        'a previously processed version of this dataset. If you really '
                        'want to make use of another processing technique, make '
                        'sure to delete the `{}` files first.'.format(
                            os.path.join(self.processed_dir, '{}*'.format(self.name))))

        if files_exist(self.processed_paths):  # pragma: no cover
            print('Processed dataset found at {}'.format(self.processed_paths))
            return

        print('Processing...')

        makedirs(self.processed_dir)
        self.process()

        path = os.path.join(self.processed_dir, '{}_process_args.pkl'.format(self.name))
        with open(path, 'wb') as handle:
            pickle.dump(self.process_args, handle)

        print('Done!')
        
    def num_features(self, dim):
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
        
    def process(self):
        raise NotImplementedError()
    
    def len(self):
        """
            Returns the number of samples in the dataset.
        """
        raise NotImplementedError()
    
    def get(self, idx):
        """
            Gets the data object at index :obj:`idx`.
        """
        raise NotImplementedError()
        
    def get_idx_split(self):
        """
            Returns the split dictionary.
        """
        raise NotImplementedError()
        
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
        # TODO: PyG abstracts this by having an additional __indices__ property and by returning another
        # dataset rather than a list, as a copy of self but with only the selected __indices__
        data = [self.get(i) for i in indices]
        return data

    def __repr__(self):  # pragma: no cover
        return f'{self.__class__.__name__}({len(self)})'
    
class InMemoryComplexDataset(ComplexDataset):
    
    def __init__(self, root, name, eval_metric, task_type, max_dim=2, num_classes=2, process_args={}, **kwargs):
        super(InMemoryComplexDataset, self).__init__(root, name, eval_metric, task_type, max_dim=max_dim, num_classes=num_classes, process_args=process_args, **kwargs)
    
    def _process(self):
        super(InMemoryComplexDataset, self)._process()
        self._load_data()
        if self.num_features(0) is None:  # we have loaded form disk
            self._look_up_num_features()
            # TODO: ^^^ find a better way to do this, it could also be dumping them
        
    def _load_data(self):
        self._data = list()
        for path in self.processed_paths:
            with open(path, 'rb') as handle:
                self._data += pickle.load(handle)
                
    def len(self):
        return len(self._data)
    
    def get(self, idx):
        # TODO: do we need copy, deep copy, or simple reassignment?
        return copy.copy(self._data[idx])
        
class SRDataset(InMemoryComplexDataset):
    
    def __init__(self, root, name, eval_metric, task_type, max_dim=2, num_classes=2, process_args={}, **kwargs):
        process_args['exp_dim'] = max_dim
        super(SRDataset, self).__init__(root, name, eval_metric, task_type, max_dim=max_dim, num_classes=num_classes, process_args=process_args, **kwargs)
        
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

    @property
    def processed_file_names(self):
        return ['{}_complex_list.pkl'.format(self.name)]
        
    def process(self):
        data = load_sr_dataset(os.path.join(self.raw_dir, self.name+'.g6'))
        exp_dim = self.process_args['exp_dim']
        complexes = list()
        max_dim = -1
        for datum in data:
            edge_index, num_nodes = datum
            x = torch.ones(num_nodes, 1, dtype=torch.float32)
            y = torch.ones(1, dtype=torch.long)
            complex = compute_clique_complex_with_gudhi(x, edge_index, num_nodes, expansion_dim=exp_dim, y=y)
            if complex.dimension > max_dim:
                max_dim = complex.dimension
            for dim in range(complex.dimension + 1):
                if self._num_features[dim] is None:
                    self._num_features[dim] = complex.chains[dim].num_features
                else:
                    assert self._num_features[dim] == complex.chains[dim].num_features
            complexes.append(complex)
        self.max_dim = max_dim
        path = self.processed_paths[0]
        with open(path, 'wb') as handle:
            pickle.dump(complexes, handle)

    def _set_task(self, task_type, num_classes):
        super(SRDataset, self)._set_task(task_type, num_classes)
        self.num_classes = len(self)