"""
Based on https://github.com/rusty1s/pytorch_geometric/blob/master/torch_geometric/nn/conv/message_passing.py

MIT License

Copyright (c) 2020 Matthias Fey <matthias.fey@tu-dortmund.de>
Copyright (c) 2021 The CWN Project Authors

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
"""

from inspect import Parameter
from typing import List, Optional, Set
from torch_geometric.typing import Adj, Size

import torch
from torch import Tensor
from torch_sparse import SparseTensor
from torch_scatter import gather_csr, scatter, segment_csr

from torch_geometric.nn.conv.utils.helpers import expand_left
from mp.cell_mp_inspector import CellularInspector


class CochainMessagePassing(torch.nn.Module):
    """The base class for building message passing models on cochain complexes.

    # TODO: Add support for co-boundary adjacencies
    The class considers three types of adjacencies:
    boundary, upper and lower adjacencies.

    Args:
        up_msg_size (int): The dimensionality of the messages coming from the upper adjacent cells.
        down_msg_size (int): The dimensionality of the messages coming from the
            lower adjacent cells.
        aggr_up (string, optional): The aggregation scheme to use for upper-adjacencies
            (:obj:`"add"`, :obj:`"mean"`, :obj:`"max"` or :obj:`None`).
            (default: :obj:`"add"`)
        aggr_down (string, optional): The aggregation scheme to use for lower-adjacencies
            (:obj:`"add"`, :obj:`"mean"`, :obj:`"max"` or :obj:`None`).
            (default: :obj:`"add"`)
        aggr_boundary (string, optional): The aggregation scheme to use for boundary adjacencies.
        flow (string, optional): The flow adjacency of message passing
            (:obj:`"source_to_target"` or :obj:`"target_to_source"`).
            (default: :obj:`"source_to_target"`)
        node_dim (int, optional): The axis along which to propagate.
            (default: :obj:`-2`)
        boundary_msg_size (int, optional): The dimensionality of the messages coming from the
            boundary cells.
        use_down_msg (bool): Whether to propagate messages via the lower adjacencies.
        use_boundary_msg (bool): Whether to propagate messages via the boundary adjacencies.
    """

    special_args: Set[str] = {
        'up_index', 'up_adj_t', 'up_index_i', 'up_index_j', 'up_size',
        'up_size_i', 'up_size_j', 'up_ptr', 'agg_up_index', 'up_dim_size',

        'down_index', 'down_adj_t', 'down_index_i', 'down_index_j', 'down_size',
        'down_size_i', 'down_size_j', 'down_ptr', 'agg_down_index', 'down_dim_size',

        'boundary_index', 'boundary_adj_t', 'boundary_index_i', 'boundary_index_j', 'boundary_size',
        'boundary_size_i', 'boundary_size_j', 'boundary_ptr', 'agg_boundary_index', 'boundary_dim_size',
    }

    def __init__(self,
                 up_msg_size,
                 down_msg_size,
                 aggr_up: Optional[str] = "add",
                 aggr_down: Optional[str] = "add",
                 aggr_boundary: Optional[str] = "add",
                 flow: str = "source_to_target",
                 node_dim: int = -2,
                 boundary_msg_size=None,
                 use_down_msg=True,
                 use_boundary_msg=True):

        super(CochainMessagePassing, self).__init__()

        self.up_msg_size = up_msg_size
        self.down_msg_size = down_msg_size
        self.use_boundary_msg = use_boundary_msg
        self.use_down_msg = use_down_msg
        # Use the same out dimension for boundaries as for down adjacency by default
        self.boundary_msg_size = down_msg_size if boundary_msg_size is None else boundary_msg_size
        self.aggr_up = aggr_up
        self.aggr_down = aggr_down
        self.aggr_boundary = aggr_boundary
        assert self.aggr_up in ['add', 'mean', 'max', None]
        assert self.aggr_down in ['add', 'mean', 'max', None]

        self.flow = flow
        assert self.flow in ['source_to_target', 'target_to_source']

        # This is the dimension in which nodes live in the feature matrix x.
        # i.e. if x has shape [N, in_channels], then node_dim = 0 or -2
        self.node_dim = node_dim

        self.inspector = CellularInspector(self)
        # This stores the parameters of these functions. If pop first is true
        # the first parameter is not stored (I presume this is for self.)
        # I presume this doesn't pop first to avoid including the self parameter multiple times.
        self.inspector.inspect(self.message_up)
        self.inspector.inspect(self.message_down)
        self.inspector.inspect(self.message_boundary)
        self.inspector.inspect(self.aggregate_up, pop_first_n=1)
        self.inspector.inspect(self.aggregate_down, pop_first_n=1)
        self.inspector.inspect(self.aggregate_boundary, pop_first_n=1)
        self.inspector.inspect(self.message_and_aggregate_up, pop_first_n=1)
        self.inspector.inspect(self.message_and_aggregate_down, pop_first_n=1)
        self.inspector.inspect(self.message_and_aggregate_boundary, pop_first_n=1)
        self.inspector.inspect(self.update, pop_first_n=3)

        # Return the parameter name for these functions minus those specified in special_args
        # TODO(Cris): Split user args by type of adjacency to make sure no bugs are introduced.
        self.__user_args__ = self.inspector.keys(
            ['message_up', 'message_down', 'message_boundary', 'aggregate_up',
             'aggregate_down', 'aggregate_boundary']).difference(self.special_args)
        self.__fused_user_args__ = self.inspector.keys(
            ['message_and_aggregate_up',
             'message_and_aggregate_down',
             'message_and_aggregate_boundary']).difference(self.special_args)
        self.__update_user_args__ = self.inspector.keys(
            ['update']).difference(self.special_args)

        # Support for "fused" message passing.
        self.fuse_up = self.inspector.implements('message_and_aggregate_up')
        self.fuse_down = self.inspector.implements('message_and_aggregate_down')
        self.fuse_boundary = self.inspector.implements('message_and_aggregate_boundary')

    def __check_input_together__(self, index_up, index_down, size_up, size_down):
        # If we have both up and down adjacency, then check the sizes agree.
        if (index_up is not None and index_down is not None
                and size_up is not None and size_down is not None):
            assert size_up[0] == size_down[0]
            assert size_up[1] == size_down[1]

    def __check_input_separately__(self, index, size):
        """This gets an up or down index and the size of the assignment matrix"""
        the_size: List[Optional[int]] = [None, None]

        if isinstance(index, Tensor):
            assert index.dtype == torch.long
            assert index.dim() == 2
            assert index.size(0) == 2
            if size is not None:
                the_size[0] = size[0]
                the_size[1] = size[1]
            return the_size

        elif isinstance(index, SparseTensor):
            if self.flow == 'target_to_source':
                raise ValueError(
                    ('Flow adjacency "target_to_source" is invalid for '
                     'message propagation via `torch_sparse.SparseTensor`. If '
                     'you really want to make use of a reverse message '
                     'passing flow, pass in the transposed sparse tensor to '
                     'the message passing module, e.g., `adj_t.t()`.'))
            the_size[0] = index.sparse_size(1)
            the_size[1] = index.sparse_size(0)
            return the_size

        elif index is None:
            return the_size

        raise ValueError(
            ('`MessagePassing.propagate` only supports `torch.LongTensor` of '
             'shape `[2, num_messages]` or `torch_sparse.SparseTensor` for '
             'argument `edge_index`.'))

    def __set_size__(self, size: List[Optional[int]], dim: int, src: Tensor):
        the_size = size[dim]
        if the_size is None:
            size[dim] = src.size(self.node_dim)
        elif the_size != src.size(self.node_dim):
            raise ValueError(
                (f'Encountered tensor with size {src.size(self.node_dim)} in '
                 f'dimension {self.node_dim}, but expected size {the_size}.'))

    def __lift__(self, src, index, dim):
        if isinstance(index, Tensor):
            index = index[dim]
            return src.index_select(self.node_dim, index)
        elif isinstance(index, SparseTensor):
            if dim == 1:
                rowptr = index.storage.rowptr()
                rowptr = expand_left(rowptr, dim=self.node_dim, dims=src.dim())
                return gather_csr(src, rowptr)
            elif dim == 0:
                col = index.storage.col()
                return src.index_select(self.node_dim, col)
        raise ValueError

    def __collect__(self, args, index, size, adjacency, kwargs):
        i, j = (1, 0) if self.flow == 'source_to_target' else (0, 1)
        assert adjacency in ['up', 'down', 'boundary']

        out = {}
        for arg in args:
            # Here the x_i and x_j parameters are automatically extracted
            # from an argument having the prefix x.
            if arg[-2:] not in ['_i', '_j']:
                out[arg] = kwargs.get(arg, Parameter.empty)
            elif index is not None:
                dim = 0 if arg[-2:] == '_j' else 1
                # Extract any part up to _j or _i. So for x_j extract x
                if adjacency == 'up' and arg.startswith('up_'):
                    data = kwargs.get(arg[3:-2], Parameter.empty)
                    size_data = data
                elif adjacency == 'down' and arg.startswith('down_'):
                    data = kwargs.get(arg[5:-2], Parameter.empty)
                    size_data = data
                elif adjacency == 'boundary' and arg.startswith('boundary_'):
                    if dim == 0:
                        # We need to use the boundary attribute matrix (i.e. boundary_attr) for the features
                        # And we need to use the x matrix to extract the number of parent cells
                        data = kwargs.get('boundary_attr', Parameter.empty)
                        size_data = kwargs.get(arg[9:-2], Parameter.empty)
                    else:
                        data = kwargs.get(arg[9:-2], Parameter.empty)
                        size_data = data
                else:
                    continue

                # This was used before for the case when data is supplied directly
                # as (x_i, x_j) as opposed to a matrix X [N, in_channels]
                # (the 2nd case is handled by the next if)
                if isinstance(data, (tuple, list)):
                    raise ValueError('This format is not supported for cellular message passing')

                # This is the usual case when we get a feature matrix of shape [N, in_channels]
                if isinstance(data, Tensor):
                    # Same size checks as above.
                    self.__set_size__(size, dim, size_data)
                    # Select the features of the nodes indexed by i or j from the data matrix
                    data = self.__lift__(data, index, j if arg[-2:] == '_j' else i)

                out[arg] = data

        # Automatically builds some default parameters that can be used in the message passing
        # functions as needed. This was modified to be discriminative of upper and lower adjacency.
        if isinstance(index, Tensor):
            out[f'{adjacency}_adj_t'] = None
            out[f'{adjacency}_ptr'] = None
            out[f'{adjacency}_index'] = index
            out[f'{adjacency}_index_i'] = index[i]
            out[f'{adjacency}_index_j'] = index[j]
        elif isinstance(index, SparseTensor):
            out['edge_index'] = None
            out[f'{adjacency}_adj_t'] = index
            out[f'{adjacency}_index_i'] = index.storage.row()
            out[f'{adjacency}_index_j'] = index.storage.col()
            out[f'{adjacency}_ptr'] = index.storage.rowptr()
            out[f'{adjacency}_weight'] = index.storage.value()
            out[f'{adjacency}_attr'] = index.storage.value()
            out[f'{adjacency}_type'] = index.storage.value()

        # We need this if in contrast to pyg because index can be None for some adjacencies.
        if isinstance(index, Tensor) or isinstance(index, SparseTensor):
            # This is the old `index` argument used for aggregation of the messages.
            out[f'agg_{adjacency}_index'] = out[f'{adjacency}_index_i']

        out[f'{adjacency}_size'] = size
        out[f'{adjacency}_size_i'] = size[1] or size[0]
        out[f'{adjacency}_size_j'] = size[0] or size[1]
        out[f'{adjacency}_dim_size'] = out[f'{adjacency}_size_i']
        return out

    def get_msg_and_agg_func(self, adjacency):
        if adjacency == 'up':
            return self.message_and_aggregate_up
        if adjacency == 'down':
            return self.message_and_aggregate_down
        elif adjacency == 'boundary':
            return self.message_and_aggregate_boundary
        else:
            return None

    def get_msg_func(self, adjacency):
        if adjacency == 'up':
            return self.message_up
        elif adjacency == 'down':
            return self.message_down
        elif adjacency == 'boundary':
            return self.message_boundary
        else:
            return None

    def get_agg_func(self, adjacency):
        if adjacency == 'up':
            return self.aggregate_up
        elif adjacency == 'down':
            return self.aggregate_down
        elif adjacency == 'boundary':
            return self.aggregate_boundary
        else:
            return None

    def get_fuse_boolean(self, adjacency):
        if adjacency == 'up':
            return self.fuse_up
        elif adjacency == 'down':
            return self.fuse_down
        elif adjacency == 'boundary':
            return self.fuse_boundary
        else:
            return None

    def __message_and_aggregate__(self, index: Adj,
                                  adjacency: str,
                                  size: List[Optional[int]] = None,
                                  **kwargs):
        assert adjacency in ['up', 'down', 'boundary']

        # Fused message and aggregation
        fuse = self.get_fuse_boolean(adjacency)
        if isinstance(index, SparseTensor) and fuse:
            # Collect the objects to pass to the function params in __user_arg.
            coll_dict = self.__collect__(self.__fused_user_args__, index, size, adjacency, kwargs)

            # message and aggregation are fused in a single function
            msg_aggr_kwargs = self.inspector.distribute(
                f'message_and_aggregate_{adjacency}', coll_dict)
            message_and_aggregate = self.get_msg_and_agg_func(adjacency)
            return message_and_aggregate(index, **msg_aggr_kwargs)

        # Otherwise, run message and aggregation in separation.
        elif isinstance(index, Tensor) or not fuse:
            # Collect the objects to pass to the function params in __user_arg.
            coll_dict = self.__collect__(self.__user_args__, index, size, adjacency, kwargs)

            # Up message and aggregation
            msg_kwargs = self.inspector.distribute(f'message_{adjacency}', coll_dict)
            message = self.get_msg_func(adjacency)
            out = message(**msg_kwargs)
            
            # import pdb; pdb.set_trace()
            aggr_kwargs = self.inspector.distribute(f'aggregate_{adjacency}', coll_dict)
            aggregate = self.get_agg_func(adjacency)
            return aggregate(out, **aggr_kwargs)

    def propagate(self, up_index: Optional[Adj],
                  down_index: Optional[Adj],
                  boundary_index: Optional[Adj],  # The None default does not work here!
                  up_size: Size = None,
                  down_size: Size = None,
                  boundary_size: Size = None,
                  **kwargs):
        """The initial call to start propagating messages."""
        up_size = self.__check_input_separately__(up_index, up_size)
        down_size = self.__check_input_separately__(down_index, down_size)
        boundary_size = self.__check_input_separately__(boundary_index, boundary_size)
        self.__check_input_together__(up_index, down_index, up_size, down_size)

        up_out, down_out = None, None
        # Up messaging and aggregation
        if up_index is not None:
            up_out = self.__message_and_aggregate__(up_index, 'up', up_size, **kwargs)

        # Down messaging and aggregation
        if self.use_down_msg and down_index is not None:
            down_out = self.__message_and_aggregate__(down_index, 'down', down_size, **kwargs)

        # boundary messaging and aggregation
        boundary_out = None
        if self.use_boundary_msg and 'boundary_attr' in kwargs and kwargs['boundary_attr'] is not None:
            boundary_out = self.__message_and_aggregate__(boundary_index, 'boundary', boundary_size, **kwargs)

        coll_dict = {}
        up_coll_dict = self.__collect__(self.__update_user_args__, up_index, up_size, 'up',
                                        kwargs)
        down_coll_dict = self.__collect__(self.__update_user_args__,
                                          down_index, down_size, 'down', kwargs)
        coll_dict.update(up_coll_dict)
        coll_dict.update(down_coll_dict)
        update_kwargs = self.inspector.distribute('update', coll_dict)
        return self.update(up_out, down_out, boundary_out, **update_kwargs)

    def message_up(self, up_x_j: Tensor, up_attr: Tensor) -> Tensor:
        r"""Constructs upper messages from cell :math:`j` to cell :math:`i` for each edge in
        :obj:`up_index`. This function can take any argument as input which was initially
        passed to :meth:`propagate`. Furthermore, tensors passed to :meth:`propagate` can be mapped
        to the respective cells :math:`i` and :math:`j` by appending :obj:`_i` or
        :obj:`_j` to the variable name, *.e.g.* :obj:`x_i` and :obj:`x_j`. The parameter
        :obj:`up_attr` includes the features of the shared coboundary cell.
        """
        return up_x_j

    def message_down(self, down_x_j: Tensor, down_attr: Tensor) -> Tensor:
        r"""Constructs lower messages from cell :math:`j` to cell :math:`i` for each edge in
        :obj:`down_index`. This function can take any argument as input which was initially
        passed to :meth:`propagate`. Furthermore, tensors passed to :meth:`propagate` can be mapped
        to the respective cells :math:`i` and :math:`j` by appending :obj:`_i` or
        :obj:`_j` to the variable name, *.e.g.* :obj:`x_i` and :obj:`x_j`. The parameter
        :obj:`down_attr` includes the features of the shared boundary cell.
        """
        return down_x_j

    def message_boundary(self, boundary_x_j: Tensor):
        r"""Constructs boundary messages from cell :math:`j` to cell :math:`i` for each edge in
        :obj:`boundary_index`. This function can take any argument as input which was initially
        passed to :meth:`propagate`. Furthermore, tensors passed to :meth:`propagate` can be mapped
        to the respective cells :math:`i` and :math:`j` by appending :obj:`_i` or
        :obj:`_j` to the variable name, *.e.g.* :obj:`x_i` and :obj:`x_j`.
        """
        return boundary_x_j

    def aggregate_up(self, inputs: Tensor, agg_up_index: Tensor,
                     up_ptr: Optional[Tensor] = None,
                     up_dim_size: Optional[int] = None) -> Tensor:
        r"""Aggregates messages from upper adjacent cells.

        Takes in the output of message computation as first argument and any
        argument which was initially passed to :meth:`propagate`.

        By default, this function will delegate its call to scatter functions
        that support "add", "mean" and "max" operations as specified in
        :meth:`__init__` by the :obj:`aggr` argument.
        """
        if up_ptr is not None:
            up_ptr = expand_left(up_ptr, dim=self.node_dim, dims=inputs.dim())
            return segment_csr(inputs, up_ptr, reduce=self.aggr_up)
        else:
            return scatter(inputs, agg_up_index, dim=self.node_dim, dim_size=up_dim_size,
                           reduce=self.aggr_up)

    def aggregate_down(self, inputs: Tensor, agg_down_index: Tensor,
                       down_ptr: Optional[Tensor] = None,
                       down_dim_size: Optional[int] = None) -> Tensor:
        r"""Aggregates messages from lower adjacent cells.

        Takes in the output of message computation as first argument and any
        argument which was initially passed to :meth:`propagate`.

        By default, this function will delegate its call to scatter functions
        that support "add", "mean" and "max" operations as specified in
        :meth:`__init__` by the :obj:`aggr` argument.
        """
        if down_ptr is not None:
            down_ptr = expand_left(down_ptr, dim=self.node_dim, dims=inputs.dim())
            return segment_csr(inputs, down_ptr, reduce=self.aggr_down)
        else:
            return scatter(inputs, agg_down_index, dim=self.node_dim, dim_size=down_dim_size,
                           reduce=self.aggr_down)

    def aggregate_boundary(self, inputs: Tensor, agg_boundary_index: Tensor,
                       boundary_ptr: Optional[Tensor] = None,
                       boundary_dim_size: Optional[int] = None) -> Tensor:
        r"""Aggregates messages from the boundary cells.

        Takes in the output of message computation as first argument and any
        argument which was initially passed to :meth:`propagate`.

        By default, this function will delegate its call to scatter functions
        that support "add", "mean" and "max" operations as specified in
        :meth:`__init__` by the :obj:`aggr` argument.
        """
        # import pdb; pdb.set_trace()
        if boundary_ptr is not None:
            down_ptr = expand_left(boundary_ptr, dim=self.node_dim, dims=inputs.dim())
            return segment_csr(inputs, down_ptr, reduce=self.aggr_boundary)
        else:
            return scatter(inputs, agg_boundary_index, dim=self.node_dim, dim_size=boundary_dim_size,
                           reduce=self.aggr_boundary)

    def message_and_aggregate_up(self, up_adj_t: SparseTensor) -> Tensor:
        r"""Fuses computations of :func:`message_up` and :func:`aggregate_up` into a
        single function.
        If applicable, this saves both time and memory since messages do not
        explicitly need to be materialized.
        This function will only gets called in case it is implemented and
        propagation takes place based on a :obj:`torch_sparse.SparseTensor`.
        """
        raise NotImplementedError

    def message_and_aggregate_down(self, down_adj_t: SparseTensor) -> Tensor:
        r"""Fuses computations of :func:`message_down` and :func:`aggregate_down` into a
        single function.
        If applicable, this saves both time and memory since messages do not
        explicitly need to be materialized.
        This function will only gets called in case it is implemented and
        propagation takes place based on a :obj:`torch_sparse.SparseTensor`.
        """
        raise NotImplementedError

    def message_and_aggregate_boundary(self, boundary_adj_t: SparseTensor) -> Tensor:
        r"""Fuses computations of :func:`message_boundary` and :func:`aggregate_boundary` into a
        single function.
        If applicable, this saves both time and memory since messages do not
        explicitly need to be materialized.
        This function will only gets called in case it is implemented and
        propagation takes place based on a :obj:`torch_sparse.SparseTensor`.
        """
        raise NotImplementedError

    def update(self, up_inputs: Optional[Tensor], down_inputs: Optional[Tensor],
               boundary_inputs: Optional[Tensor], x: Tensor) -> (Tensor, Tensor, Tensor):
        r"""Updates cell embeddings. Takes in the output of the aggregations from different
        adjacencies as the first three arguments and any argument which was initially passed to
        :meth:`propagate`.
        """
        if up_inputs is None:
            up_inputs = torch.zeros(x.size(0), self.up_msg_size).to(device=x.device)
        if down_inputs is None:
            down_inputs = torch.zeros(x.size(0), self.down_msg_size).to(device=x.device)
        if boundary_inputs is None:
            boundary_inputs = torch.zeros(x.size(0), self.boundary_msg_size).to(device=x.device)

        return up_inputs, down_inputs, boundary_inputs


class CochainMessagePassingParams:
    """A helper class storing the parameters to be supplied to the propagate function.

    This object stores the equivalent of the `x` and `edge_index` objects from PyTorch Geometric.
    TODO: The boundary_index and boundary_attr as well as other essential parameters are
          currently passed as keyword arguments. Special parameters should be created.
    Args:
        x: The features of the cochain where message passing will be performed.
        up_index: The index for the upper adjacencies of the cochain.
        down_index: The index for the lower adjacencies of the cochain.
    """
    def __init__(self, x: Tensor, up_index: Adj = None, down_index: Adj = None, **kwargs):
        self.x = x
        self.up_index = up_index
        self.down_index = down_index
        self.kwargs = kwargs
        if 'boundary_index' in self.kwargs:
            self.boundary_index = self.kwargs['boundary_index']
        else:
            self.boundary_index = None
        if 'boundary_attr' in self.kwargs:
            self.boundary_attr = self.kwargs['boundary_attr']
        else:
            self.boundary_attr = None
