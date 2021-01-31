import torch
import torch.nn.functional as F

from torch.nn import Linear, Sequential, ReLU, BatchNorm1d as BN
from torch_geometric.nn import global_mean_pool, global_add_pool, JumpingKnowledge
from mp.layers import SINConv, EdgeSINConv, DummySimplicialMessagePassing
from data.complex import Complex, ComplexBatch


class SIN0(torch.nn.Module):
    """
    A simplicial version of GIN.

    This model is based on
    https://github.com/rusty1s/pytorch_geometric/blob/master/benchmark/kernel/gin.py
    """
    def __init__(self, num_input_features, num_classes, num_layers, hidden, dropout_rate: float = 0.5,
                 max_dim: int = 2, linear_output: bool = False, jump_mode=None):
        super(SIN0, self).__init__()

        self.max_dim = max_dim
        self.dropout_rate = dropout_rate
        self.linear_output = linear_output
        self.jump_mode = jump_mode
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
                SINConv(layer_dim, layer_dim,
                        conv_up, conv_down, conv_update, train_eps=False, max_dim=self.max_dim))
        self.jump = JumpingKnowledge(jump_mode) if jump_mode is not None else None
        if jump_mode == 'cat':
            self.lin1 = Linear(num_layers * hidden, hidden)
        else:
            self.lin1 = Linear(hidden, hidden)
        self.lin2 = Linear(hidden, num_classes)

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        if self.jump_mode is not None:
            self.jump.reset_parameters()
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()

    def pool_complex(self, xs, data):
        # All complexes have nodes so we can extract the batch size from chains[0]
        batch_size = data.chains[0].batch.max() + 1
        # The MP output is of shape [message_passing_dim, batch_size, feature_dim]
        pooled_xs = torch.zeros(self.max_dim+1, batch_size, xs[0].size(-1),
                                device=batch_size.device)
        for i in range(len(xs)):
            # It's very important that size is supplied.
            pooled_xs[i, :, :] = global_mean_pool(xs[i], data.chains[i].batch, size=batch_size)
        return pooled_xs

    def jump_complex(self, jump_xs):
        # Perform JumpingKnowledge at each level of the complex
        xs = []
        for jumpx in jump_xs:
            xs += [self.jump(jumpx)]
        return xs

    def forward(self, data: ComplexBatch):
        xs, jump_xs = None, None
        for i, conv in enumerate(self.convs):
            params = data.get_all_chain_params(max_dim=self.max_dim)
            xs = conv(*params)
            data.set_xs(xs)

            if self.jump_mode is not None:
                if jump_xs is None:
                    jump_xs = [[] for _ in xs]
                for i, x in enumerate(xs):
                    jump_xs[i] += [x]

        if self.jump_mode is not None:
            xs = self.jump_complex(jump_xs)
        pooled_xs = self.pool_complex(xs, data)
        x = pooled_xs.sum(dim=0)

        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=self.dropout_rate, training=self.training)
        x = self.lin2(x)
        if self.linear_output:
            return x
        return F.log_softmax(x, dim=-1)

    def __repr__(self):
        return self.__class__.__name__


class EdgeSIN0(torch.nn.Module):
    """
    A variant of SIN0 operating up to edge level. It may optionally ignore triangle features.

    This model is based on
    https://github.com/rusty1s/pytorch_geometric/blob/master/benchmark/kernel/gin.py
    """
    def __init__(self, num_input_features, num_classes, num_layers, hidden, dropout_rate: float = 0.5,
                 linear_output: bool = False, jump_mode=None, include_top_features=True,
                 update_top_features=True):
        super(EdgeSIN0, self).__init__()

        self.max_dim = 1
        self.include_top_features = include_top_features
        # If the top features are included, then they can be updated by a network.
        self.update_top_features = include_top_features and update_top_features
        self.dropout_rate = dropout_rate
        self.linear_output = linear_output
        self.jump_mode = jump_mode
        self.convs = torch.nn.ModuleList()
        self.update_top_nns = torch.nn.ModuleList()

        for i in range(num_layers):
            layer_dim = num_input_features if i == 0 else hidden
            v_conv_update = Sequential(
                Linear(layer_dim, hidden),
                ReLU(),
                Linear(hidden, hidden),
                ReLU(),
                BN(hidden))
            e_conv_update = Sequential(
                Linear(layer_dim, hidden),
                ReLU(),
                Linear(hidden, hidden),
                ReLU(),
                BN(hidden))
            v_conv_up = Sequential(
                Linear(layer_dim * 2, layer_dim),
                ReLU(),
                BN(layer_dim))
            e_conv_down = Sequential(
                Linear(layer_dim * 2, layer_dim),
                ReLU(),
                BN(layer_dim))
            e_conv_inp_dim = layer_dim*2 if include_top_features else layer_dim
            e_conv_up = Sequential(
                Linear(e_conv_inp_dim, layer_dim),
                ReLU(),
                BN(layer_dim))
            self.convs.append(
                EdgeSINConv(layer_dim, layer_dim, v_conv_up, e_conv_down, e_conv_up,
                            v_conv_update, e_conv_update, train_eps=False))
            if self.update_top_features and i < num_layers - 1:
                self.update_top_nns.append(Sequential(
                    Linear(layer_dim, hidden),
                    ReLU(),
                    Linear(hidden, hidden),
                    ReLU(),
                    BN(hidden))
                )

        self.jump = JumpingKnowledge(jump_mode) if jump_mode is not None else None
        if jump_mode == 'cat':
            self.lin1 = Linear(num_layers * hidden, hidden)
        else:
            self.lin1 = Linear(hidden, hidden)
        self.lin2 = Linear(hidden, num_classes)

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        if self.jump_mode is not None:
            self.jump.reset_parameters()
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()

    def pool_complex(self, xs, data):
        # All complexes have nodes so we can extract the batch size from chains[0]
        batch_size = data.chains[0].batch.max() + 1
        # The MP output is of shape [message_passing_dim, batch_size, feature_dim]
        pooled_xs = torch.zeros(self.max_dim+1, batch_size, xs[0].size(-1),
                                device=batch_size.device)
        for i in range(len(xs)):
            # It's very important that size is supplied.
            pooled_xs[i, :, :] = global_mean_pool(xs[i], data.chains[i].batch, size=batch_size)
        return pooled_xs

    def jump_complex(self, jump_xs):
        # Perform JumpingKnowledge at each level of the complex
        xs = []
        for jumpx in jump_xs:
            xs += [self.jump(jumpx)]
        return xs

    def forward(self, data: ComplexBatch):
        xs, jump_xs = None, None
        for i, conv in enumerate(self.convs):
            params = data.get_all_chain_params(max_dim=self.max_dim,
                                               include_top_features=self.include_top_features)
            xs = conv(*params)
            if self.update_top_features and i < len(self.convs) - 1 and 2 in data.chains:
                top_x = self.update_top_nns[i](data.chains[2].x)
                data.set_xs(xs + [top_x])
            else:
                data.set_xs(xs)

            if self.jump_mode is not None:
                if jump_xs is None:
                    jump_xs = [[] for _ in xs]
                for i, x in enumerate(xs):
                    jump_xs[i] += [x]

        if self.jump_mode is not None:
            xs = self.jump_complex(jump_xs)

        pooled_xs = self.pool_complex(xs, data)
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
