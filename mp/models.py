import torch
import torch.nn.functional as F

from torch.nn import Linear, Sequential, ReLU, BatchNorm1d as BN
from torch_geometric.nn import global_mean_pool, global_add_pool
from mp.layers import SINConv, DummySimplicialMessagePassing
from data.complex import Complex, ComplexBatch


class SIN0(torch.nn.Module):
    """
    A simplicial version of GIN.

    This model is based on
    https://github.com/rusty1s/pytorch_geometric/blob/master/benchmark/kernel/gin.py
    """
    def __init__(self, num_input_features, num_classes, num_layers, hidden, dropout_rate: float = 0.5,
                 max_dim: int = 2, linear_output: bool = False):
        super(SIN0, self).__init__()

        self.max_dim = max_dim
        self.dropout_rate = dropout_rate
        self.linear_output = linear_output
        self.convs = torch.nn.ModuleList()
        for i in range(num_layers):
            layer_dim = num_input_features if i == 0 else hidden
            conv_update = Sequential(
                Linear(layer_dim, hidden),
                ReLU(),
                Linear(hidden, hidden),
                ReLU(),
                BN(hidden))
            conv_up = Sequential(
                Linear(layer_dim * 2, layer_dim),
                ReLU(),
                BN(layer_dim))
            conv_down = Sequential(
                Linear(layer_dim * 2, layer_dim),
                ReLU(),
                BN(layer_dim))
            self.convs.append(
                SINConv(conv_up, conv_down, conv_update, train_eps=False, max_dim=self.max_dim))
        self.lin1 = Linear(hidden, hidden)
        self.lin2 = Linear(hidden, num_classes)

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()

    def forward(self, data: ComplexBatch):
        xs = None
        for i, conv in enumerate(self.convs):
            params = data.get_all_chain_params()
            xs = conv(*params)
            data.set_xs(xs)

        # All complexes have nodes so we can extract the batch size from chains[0]
        batch_size = data.chains[0].batch.max() + 1
        # The MP output is of shape [message_passing_dim, batch_size, feature_dim]
        # We assume that all layers have the same feature size.
        # Note that levels where we do MP at but where there was no data are set to 0.
        # TODO: shall we retain the device as an attribute of self? then `device=batch_size.device`
        # would become `device=self.device`
        pooled_xs = torch.zeros(self.max_dim+1, batch_size, xs[0].size(-1), device=batch_size.device)
        for i in range(len(xs)):
            # It's very important that size is supplied.
            # Otherwise, if we have complexes with no simplices at certain levels, the wrong
            # shape could be inferred automatically from data.chains[i].batch.
            # This makes sure the output tensor will have the right dimensions.
            pooled_xs[i, :, :] = global_mean_pool(xs[i], data.chains[i].batch, size=batch_size)
        # Reduce across the levels of the complexes
        x = pooled_xs.sum(dim=0)

        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=self.dropout_rate, training=self.training)
        x = self.lin2(x)
        if self.linear_output:
            return x
        return F.log_softmax(x, dim=-1)

    def __repr__(self):
        return self.__class__.__name__
    
    
class Dummy(torch.nn.Module):
    """
    A dummy simplicial network model.
    No parameters in the convolutional layers.
    Readout at each layer is by summation.
    Outputs are computed by one single linear layer.
    """
    def __init__(self, num_input_features, num_classes, num_layers, max_dim: int = 2, linear_output: bool = False):
        super(Dummy, self).__init__()

        self.max_dim = max_dim
        self.linear_output = linear_output
        self.convs = torch.nn.ModuleList()
        for i in range(num_layers):
            self.convs.append(DummySimplicialMessagePassing(max_dim=self.max_dim))
        self.lin = Linear(num_input_features, num_classes)

    def reset_parameters(self):
        self.lin.reset_parameters()

    def forward(self, data: ComplexBatch):
        xs = None
        for i, conv in enumerate(self.convs):
            params = data.get_all_chain_params()
            xs = conv(*params)
            data.set_xs(xs)

        # All complexes have nodes so we can extract the batch size from chains[0]
        batch_size = data.chains[0].batch.max() + 1
        # The MP output is of shape [message_passing_dim, batch_size, feature_dim]
        # We assume that all layers have the same feature size.
        # Note that levels where we do MP at but where there was no data are set to 0.
        # TODO: shall we retain the device as an attribute of self? then `device=batch_size.device`
        # would become `device=self.device`
        pooled_xs = torch.zeros(self.max_dim+1, batch_size, xs[0].size(-1), device=batch_size.device)
        for i in range(len(xs)):
            # It's very important that size is supplied.
            # Otherwise, if we have complexes with no simplices at certain levels, the wrong
            # shape could be inferred automatically from data.chains[i].batch.
            # This makes sure the output tensor will have the right dimensions.
            pooled_xs[i, :, :] = global_add_pool(xs[i], data.chains[i].batch, size=batch_size)
        # Reduce across the levels of the complexes
        x = pooled_xs.sum(dim=0)

        x = self.lin(x)
        if self.linear_output:
            return x
        return F.log_softmax(x, dim=-1)

    def __repr__(self):
        return self.__class__.__name__