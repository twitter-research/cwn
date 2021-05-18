import torch
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool, global_add_pool

def get_nonlinearity(nonlinearity, return_module=True):
    if nonlinearity == 'relu':
        module = torch.nn.ReLU
        function = F.relu
    elif nonlinearity == 'elu':
        module = torch.nn.ELU
        function = F.elu
    elif nonlinearity == 'id':
        module = torch.nn.Identity
        function = lambda x: x
    elif nonlinearity == 'sigmoid':
        module = torch.nn.Sigmoid
        function = F.sigmoid
    elif nonlinearity == 'tanh':
        module = torch.nn.Tanh
        function = F.tanh
    else:
        raise NotImplementError('Nonlinearity {} is not currently supported.'.format(nonlinearity))
    if return_module:
        return module
    return function

def get_pooling_fn(readout):
    if readout == 'sum':
        return global_add_pool
    elif readout == 'mean':
        return global_mean_pool
    else:
        raise NotImplementError('Readout {} is not currently supported.'.format(readout))

def pool_complex(xs, data, max_dim, pooling_fn):
        # All complexes have nodes so we can extract the batch size from chains[0]
        batch_size = data.chains[0].batch.max() + 1
        # The MP output is of shape [message_passing_dim, batch_size, feature_dim]
        pooled_xs = torch.zeros(max_dim+1, batch_size, xs[0].size(-1),
                                device=batch_size.device)
        for i in range(len(xs)):
            # It's very important that size is supplied.
            pooled_xs[i, :, :] = pooling_fn(xs[i], data.chains[i].batch, size=batch_size)
        return pooled_xs