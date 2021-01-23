import torch
import logging
import copy

from collections import OrderedDict
from torch import Tensor
from torch_sparse import SparseTensor
from mp.smp import ChainMessagePassingParams
from torch_geometric.typing import Adj
from typing import List

__num_warn_msg__ = (
    'The number of {0} in your chain object can only be inferred by its {1}, '
    'and hence may result in unexpected batch-wise behavior, e.g., '
    'in case there exists isolated simplices. Please consider explicitly setting '
    'the number of simplices for this data object by assigning it to '
    'chain.num_{0}.')

__complex_max_dim_lower_bound__ = 2


class Chain(object):
    """
        Class representing a chain of k-dim simplices.
    """
    def __init__(self, dim: int, x: Tensor = None, upper_index: Adj = None, lower_index: Adj = None,
                 shared_faces: Tensor = None, shared_cofaces: Tensor = None, mapping: Tensor = None,
                 y=None, **kwargs):
        """
            Constructs a `dim`-chain.
            - `dim`: dim of the simplices in the chain
            - `x`: feature matrix, shape [num_simplices, num_features]; may not be available
            - `upper_index`: upper adjacency, matrix, shape [2, num_upper_connections];
            may not be available, e.g. when `dim` is the top level dim of a complex
            - `lower_index`: lower adjacency, matrix, shape [2, num_lower_connections];
            may not be available, e.g. when `dim` is the top level dim of a complex
            - `y`: labels over simplices in the chain, shape [num_simplices,]
        """
        self.dim = dim
        # TODO: check default for x
        self.__x = x
        self.upper_index = upper_index
        self.lower_index = lower_index
        self.y = y
        self.shared_faces = shared_faces
        self.shared_cofaces = shared_cofaces
        self.__oriented = False
        self.__hodge_laplacian = None
        # TODO: Figure out what to do with mapping.
        self.__mapping = mapping
        for key, item in kwargs.items():
            if key == 'num_simplices':
                self.__num_simplices__ = item
            else:
                self[key] = item

    @property
    def x(self):
        return self.__x

    @x.setter
    def x(self, new_x):
        assert len(self.x) == len(new_x)
        self.__x = new_x

    @property
    def keys(self):
        """
            Returns all names of chain attributes.
        """
        keys = [key for key in self.__dict__.keys() if self[key] is not None]
        keys = [key for key in keys if key[:2] != '__' and key[-2:] != '__']
        return keys

    def __getitem__(self, key):
        """
            Gets the data of the attribute :obj:`key`.
        """
        return getattr(self, key, None)

    def __setitem__(self, key, value):
        """
            Sets the attribute :obj:`key` to :obj:`value`.
        """
        setattr(self, key, value)

    def __contains__(self, key):
        """
            Returns :obj:`True`, if the attribute :obj:`key` is present in the
            data.
        """
        return key in self.keys

    def __cat_dim__(self, key, value):
        """
            Returns the dimension for which :obj:`value` of attribute
            :obj:`key` will get concatenated when creating batches.
        """
        if key in ['upper_index', 'lower_index', 'shared_faces', 'shared_cofaces']:
            return -1
        # by default, concatenate sparse matrices diagonally.
        elif isinstance(value, SparseTensor):
            return (0, 1)
        return 0

    def __inc__(self, key, value):
        """
            Returns the incremental count to cumulatively increase the value
            of the next attribute of :obj:`key` when creating batches.
        """
        if key in ['upper_index', 'lower_index']:
            inc = self.num_simplices
        elif key == 'shared_faces':
            inc = self.num_faces
        elif key == 'shared_cofaces':
            inc = self.num_cofaces
        else:
            inc = 0
        if inc is None:
            inc = 0
        return inc
    
    def __call__(self, *keys):
        """
            Iterates over all attributes :obj:`*keys` in the chain, yielding
            their attribute names and content.
            If :obj:`*keys` is not given this method will iterative over all
            present attributes.
        """
        for key in sorted(self.keys) if not keys else keys:
            if key in self:
                yield key, self[key]

    @property
    def num_simplices(self):
        """
            Returns or sets the number of simplices in the chain.

        .. note::
            The number of simplices in your chain object is typically automatically
            inferred, *e.g.*, when chain features :obj:`x` are present.
            In some cases however, a chain may only be given by its upper/lower
            indices :obj:`edge_index`.
            The code then *guesses* the number of simplices
            according to :obj:`{upper,lower}_index.max().item() + 1`, but in case there
            exists isolated simplices, this number has not to be correct and can
            therefore result in unexpected batch-wise behavior.
            Thus, we recommend to set the number of simplices in the chain object
            explicitly via :obj:`data.num_simplices = ...`.
            You will be given a warning that requests you to do so.
        """
        if hasattr(self, '__num_simplices__'):
            return self.__num_simplices__
        if self.x is not None:
            return self.x.size(self.__cat_dim__('x', self.x))
        if self.upper_index is not None:
            logging.warning(__num_warn_msg__.format('simplices', 'upper_index'))
            return int(self.upper_index.max()) + 1
        if self.lower_index is not None:
            logging.warning(__num_warn_msg__.format('simplices', 'lower_index'))
            return int(self.lower_index.max()) + 1
        return None

    @num_simplices.setter
    def num_simplices(self, num_simplices):
        self.__num_simplices__ = num_simplices

    @property
    def num_cofaces(self):
        """
            Returns or sets the number of overall cofaces in the chain.
        """
        if hasattr(self, '__num_cofaces__'):
            return self.__num_cofaces__
        if self.upper_index is None:
            return 0
        if self.shared_cofaces is not None:
            logging.warning(__num_warn_msg__.format('cofaces', 'shared_cofaces'))
            return int(self.shared_cofaces.max()) + 1
        return None

    @num_cofaces.setter
    def num_cofaces(self, num_cofaces):
        self.__num_cofaces__ = num_cofaces

    @property
    def num_faces(self):
        """
            Returns or sets the number of overall faces in the chain.
        """
        if hasattr(self, '__num_faces__'):
            return self.__num_faces__
        if self.lower_index is None:
            return 0
        if self.shared_faces is not None:
            logging.warning(__num_warn_msg__.format('faces', 'shared_faces'))
            return int(self.shared_faces.max()) + 1
        # TODO: better to swap these two?
        if self.num_simplices is not None:
            logging.warning(__num_warn_msg__.format('faces', 'num_simplices'))
            return (self.dim + 1) * self.num_simplices
        return None

    @num_faces.setter
    def num_faces(self, num_faces):
        self.__num_faces__ = num_faces

    def __apply__(self, item, func):
        if torch.is_tensor(item):
            return func(item)
        elif isinstance(item, SparseTensor):
            # Not all apply methods are supported for `SparseTensor`, e.g.,
            # `contiguous()`. We can get around it by capturing the exception.
            try:
                return func(item)
            except AttributeError:
                return item
        elif isinstance(item, (tuple, list)):
            return [self.__apply__(v, func) for v in item]
        elif isinstance(item, dict):
            return {k: self.__apply__(v, func) for k, v in item.items()}
        else:
            return item

    def apply(self, func, *keys):
        """
            Applies the function :obj:`func` to all tensor attributes
            :obj:`*keys`. If :obj:`*keys` is not given, :obj:`func` is applied to
            all present attributes.
        """
        for key, item in self(*keys):
            self[key] = self.__apply__(item, func)
        return self

    def contiguous(self, *keys):
        """
            Ensures a contiguous memory layout for all attributes :obj:`*keys`.
            If :obj:`*keys` is not given, all present attributes are ensured to
            have a contiguous memory layout.
        """
        return self.apply(lambda x: x.contiguous(), *keys)

    def to(self, device, *keys, **kwargs):
        """
            Performs tensor dtype and/or device conversion to all attributes
            :obj:`*keys`.
            If :obj:`*keys` is not given, the conversion is applied to all present
            attributes.
        """
        return self.apply(lambda x: x.to(device, **kwargs), *keys)

    def clone(self):
        return self.__class__.from_dict({
            k: v.clone() if torch.is_tensor(v) else copy.deepcopy(v)
            for k, v in self.__dict__.items()
        })
    
    @property
    def mapping(self):
        return self.__mapping

    def orient(self, arbitrary=None):
        """
            Enforces orientation to the chain.
            If `arbitrary` orientation is provided, it enforces that. Otherwise the canonical one
            is enforced.
        """
        raise NotImplementedError
        # TODO: what is the impact of this on lower/upper signals?
        # ...
        # ...
        # self.lower_orientation = ...  # <- shape [1, num_lower_connections], content: +/- 1.0
        # self.upper_orientation = ...  # <- shape [1, num_upper_connections], content: +/- 1.0
        # self.__oriented = True
        # return

    def get_hodge_laplacian(self):
        """
            Returns the Hodge Laplacian.
            Orientation is required; if not present, the chain will first be oriented according
            to the canonical ordering.
        """
        raise NotImplementedError
        # if self.__hodge_laplacian is None:  # caching
        #     if not self.__oriented:
        #         self.orient()
        #     self.__hodge_laplacian = ...
        #     # ^^^ here we need to perform two sparse matrix multiplications
        #     # -- we can leverage on torch_sparse
        #     # indices of the sparse matrices are self.lower_index and self.upper_index,
        #     # their values are those in
        #     # self.lower_orientation and self.upper_orientation
        # return self.__hodge_laplacian

    def initialize_features(self, strategy='constant'):
        """
            Routine to initialize simplex-wise features based on the provided `strategy`.
        """
        raise NotImplementedError
        # self.x = ...
        # return


class ChainBatch(Chain):

    def __init__(self, dim, batch=None, ptr=None, **kwargs):
        super(ChainBatch, self).__init__(dim, **kwargs)

        for key, item in kwargs.items():
            if key == 'num_simplices':
                self.__num_simplices__ = item
            else:
                self[key] = item

        self.batch = batch
        self.ptr = ptr
        self.__data_class__ = Chain
        self.__slices__ = None
        self.__cumsum__ = None
        self.__cat_dims__ = None
        self.__num_simplices_list__ = None
        self.__num_faces_list__ = None
        self.__num_cofaces_list__ = None
        self.__num_chains__ = None

    @classmethod
    def from_chain_list(cls, data_list, follow_batch=[]):
        """
            Constructs a batch object from a python list holding
            :class:`Chain` objects.
            The assignment vector :obj:`batch` is created on the fly.
            Additionally, creates assignment batch vectors for each key in
            :obj:`follow_batch`.
        """

        keys = [set(data.keys) for data in data_list]
        keys = list(set.union(*keys))
        assert 'batch' not in keys and 'ptr' not in keys

        batch = cls(data_list[0].dim)
        for key in data_list[0].__dict__.keys():
            if key[:2] != '__' and key[-2:] != '__':
                batch[key] = None

        batch.__num_chains__ = len(data_list)
        batch.__data_class__ = data_list[0].__class__
        for key in keys + ['batch']:
            batch[key] = []
        batch['ptr'] = [0]

        device = None
        slices = {key: [0] for key in keys}
        cumsum = {key: [0] for key in keys}
        cat_dims = {}
        num_simplices_list = []
        num_cofaces_list = []
        num_faces_list = []
        for i, data in enumerate(data_list):
            for key in keys:
                item = data[key]
                
                if item is not None:
                    
                    # Increase values by `cumsum` value.
                    cum = cumsum[key][-1]
                    if isinstance(item, Tensor) and item.dtype != torch.bool:
                        if not isinstance(cum, int) or cum != 0:
                            item = item + cum
                    elif isinstance(item, SparseTensor):
                        value = item.storage.value()
                        if value is not None and value.dtype != torch.bool:
                            if not isinstance(cum, int) or cum != 0:
                                value = value + cum
                            item = item.set_value(value, layout='coo')
                    elif isinstance(item, (int, float)):
                        item = item + cum

                    # Treat 0-dimensional tensors as 1-dimensional.
                    if isinstance(item, Tensor) and item.dim() == 0:
                        item = item.unsqueeze(0)

                    batch[key].append(item)

                    # Gather the size of the `cat` dimension.
                    size = 1
                    cat_dim = data.__cat_dim__(key, data[key])
                    cat_dims[key] = cat_dim
                    if isinstance(item, Tensor):
                        size = item.size(cat_dim)
                        device = item.device
                    elif isinstance(item, SparseTensor):
                        size = torch.tensor(item.sizes())[torch.tensor(cat_dim)]
                        device = item.device()
                    
                    # TODO: do we really need slices, and, are we managing them correctly?
                    slices[key].append(size + slices[key][-1])
                    
                    if key in follow_batch:
                        if isinstance(size, Tensor):
                            for j, size in enumerate(size.tolist()):
                                tmp = f'{key}_{j}_batch'
                                batch[tmp] = [] if i == 0 else batch[tmp]
                                batch[tmp].append(
                                    torch.full((size, ), i, dtype=torch.long,
                                               device=device))
                        else:
                            tmp = f'{key}_batch'
                            batch[tmp] = [] if i == 0 else batch[tmp]
                            batch[tmp].append(
                                torch.full((size, ), i, dtype=torch.long,
                                           device=device))
                    
                inc = data.__inc__(key, item)
                if isinstance(inc, (tuple, list)):
                    inc = torch.tensor(inc)
                cumsum[key].append(inc + cumsum[key][-1])

            if hasattr(data, '__num_simplices__'):
                num_simplices_list.append(data.__num_simplices__)
            else:
                num_simplices_list.append(None)

            if hasattr(data, '__num_cofaces__'):
                num_cofaces_list.append(data.__num_cofaces__)
            else:
                num_cofaces_list.append(None)

            if hasattr(data, '__num_faces__'):
                num_faces_list.append(data.__num_faces__)
            else:
                num_faces_list.append(None)

            num_simplices = data.num_simplices
            if num_simplices is not None:
                item = torch.full((num_simplices, ), i, dtype=torch.long,
                                  device=device)
                batch.batch.append(item)
                batch.ptr.append(batch.ptr[-1] + num_simplices)

        # Fix initial slice values:
        for key in keys:
            slices[key][0] = slices[key][1] - slices[key][1]

        batch.batch = None if len(batch.batch) == 0 else batch.batch
        batch.ptr = None if len(batch.ptr) == 1 else batch.ptr
        batch.__slices__ = slices
        batch.__cumsum__ = cumsum
        batch.__cat_dims__ = cat_dims
        batch.__num_simplices_list__ = num_simplices_list
        batch.__num_cofaces_list__ = num_cofaces_list
        batch.__num_faces_list__ = num_faces_list

        ref_data = data_list[0]
        for key in batch.keys:
            items = batch[key]
            item = items[0]
            if isinstance(item, Tensor):
                batch[key] = torch.cat(items, ref_data.__cat_dim__(key, item))
            elif isinstance(item, SparseTensor):
                batch[key] = torch.cat(items, ref_data.__cat_dim__(key, item))
            elif isinstance(item, (int, float)):
                batch[key] = torch.tensor(items)

        return batch.contiguous()

    # TODO: is the 'get_example' method needed for now?

    # TODO: is the 'index_select' method needed for now?

    def __getitem__(self, idx):
        if isinstance(idx, str):
            return super(ChainBatch, self).__getitem__(idx)
        elif isinstance(idx, int):
            raise NotImplementedError
            #return self.get_example(idx)
        else:
            raise NotImplementedError
            # return self.index_select(idx)

    # TODO: is the 'to_chain_list' method needed for now?
    def to_chain_list(self) -> List[Chain]:
        r"""Reconstructs the list of :class:`torch_geometric.data.Data` objects
        from the batch object.
        The batch object must have been created via :meth:`from_data_list` in
        order to be able to reconstruct the initial objects."""
        raise NotImplementedError
        #return [self.get_example(i) for i in range(self.num_chains)]

    @property
    def num_chains(self) -> int:
        """Returns the number of chains in the batch."""
        if self.__num_chains__ is not None:
            return self.__num_chains__
        return self.ptr.numel() + 1


class Complex(object):
    """
        Class representing an attributed simplicial complex.
    """

    def __init__(self, *chains: Chain, y: torch.Tensor = None, dimension: int = None):

        if dimension is None:
            dimension = len(chains) - 1
        if len(chains) < dimension + 1:
            raise ValueError('Not enough chains passed (expected {}, received {})'.format(dimension + 1, len(chains)))
        
        # TODO: This needs some data checking to check that these chains are consistent together
        self.dimension = dimension
        self.chains = {i: chains[i] for i in range(dimension + 1)}
        self.nodes = chains[0]
        self.edges = chains[1] if dimension >= 1 else None
        self.triangles = chains[2] if dimension >= 2 else None

        self.y = y  # complex-wise label for complex-level tasks
        return

    def get_chain_params(self, dim) -> ChainMessagePassingParams:
        """
            Conveniently returns all necessary input parameters to perform higher-dim
            neural message passing at the specified `dim`.
        """
        if dim in self.chains:
            simplices = self.chains[dim]
            x = simplices.x
            if simplices.upper_index is not None:
                upper_index = simplices.upper_index
                upper_features = self.chains[dim + 1].x
                if upper_features is not None:
                    upper_features = torch.index_select(upper_features, 0,
                                                        self.chains[dim].shared_cofaces)
            else:
                upper_index = None
                upper_features = None
            if simplices.lower_index is not None:
                lower_index = simplices.lower_index
                lower_features = self.chains[dim - 1].x
                if lower_features is not None:
                    lower_features = torch.index_select(lower_features, 0,
                                                        self.chains[dim].shared_faces)
            else:
                lower_index = None
                lower_features = None
            inputs = ChainMessagePassingParams(x, upper_index, lower_index,
                                               up_attr=upper_features, down_attr=lower_features)
        else:
            raise NotImplementedError(
                'Dim {} is not present in the complex or not yet supported.'.format(dim))
        return inputs

    def get_all_chain_params(self):
        all_params = []
        for dim in range(self.dimension+1):
            all_params.append(self.get_chain_params(dim))
        return all_params

    def get_labels(self, dim=None):
        """
            Returns target labels.
            If `dim`==k (integer in [0, self.dimension]) then the labels over
            k-simplices are returned.
            In the case `dim` is None the complex-wise label is returned.
        """
        if dim is None:
            y = self.y
        else:
            if dim in self.chains:
                y = self.chains[dim].y
            else:
                raise NotImplementedError(
                    'Dim {} is not present in the complex or not yet supported.'.format(dim))
        return y

    def set_xs(self, xs: List[Tensor]):
        """Sets the features of the chains to the values in the list"""
        assert (self.dimension + 1) == len(xs)
        for i, x in enumerate(xs):
            self.chains[i].x = x


class ComplexBatch(Complex):
    """
        Class representing a batch of complexes.
    """

    def __init__(self, *chains: ChainBatch, dimension: int, y: torch.Tensor = None, num_complexes: int = None):
        super(ComplexBatch, self).__init__(*chains, y=y)
        self.num_complexes = num_complexes
        self.dimension = dimension

    @classmethod
    def from_complex_list(cls, data_list: List[Complex], follow_batch=[]):
        dimension = max([complex.dimension for complex in data_list])
        chains = [list() for _ in range(dimension + 1)]
        label_list = list()
        per_complex_labels = True
        for comp in data_list:
            for dim in range(dimension+1):
                if dim not in comp.chains:
                    chains[dim].append(Chain(dim=dim))
                else:
                    chains[dim].append(comp.chains[dim])
            per_complex_labels &= comp.y is not None
            if per_complex_labels:
                label_list.append(comp.y)

        batched_chains = [ChainBatch.from_chain_list(chain_list, follow_batch=follow_batch) for chain_list in chains]
        y = None if not per_complex_labels else torch.cat(label_list, 0)
        batch = cls(*batched_chains, y=y, num_complexes=len(data_list), dimension=dimension)

        return batch

