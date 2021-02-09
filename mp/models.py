import torch
import torch.nn.functional as F

from torch.nn import Linear, Sequential, ReLU, BatchNorm1d as BN, LayerNorm as LN
from torch_geometric.nn import global_mean_pool, global_add_pool, JumpingKnowledge
from mp.layers import SINConv, EdgeSINConv, SparseSINConv, DummySimplicialMessagePassing
from data.complex import Complex, ComplexBatch


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


class SIN0(torch.nn.Module):
    """
    A simplicial version of GIN.

    This model is based on
    https://github.com/rusty1s/pytorch_geometric/blob/master/benchmark/kernel/gin.py
    """

    def __init__(self, num_input_features, num_classes, num_layers, hidden,
                 dropout_rate: float = 0.5,
                 max_dim: int = 2, jump_mode=None, nonlinearity='relu', readout='sum'):
        super(SIN0, self).__init__()

        self.max_dim = max_dim
        self.dropout_rate = dropout_rate
        self.jump_mode = jump_mode
        self.convs = torch.nn.ModuleList()
        self.nonlinearity = nonlinearity
        self.pooling_fn = get_pooling_fn(readout)
        conv_nonlinearity = get_nonlinearity(nonlinearity, return_module=True)
        for i in range(num_layers):
            layer_dim = num_input_features if i == 0 else hidden
            conv_update = Sequential(
                Linear(layer_dim, hidden),
                conv_nonlinearity(),
                Linear(hidden, hidden),
                conv_nonlinearity(),
                BN(hidden))
            conv_up = Sequential(
                Linear(layer_dim * 2, layer_dim),
                conv_nonlinearity(),
                BN(layer_dim))
            conv_down = Sequential(
                Linear(layer_dim * 2, layer_dim),
                conv_nonlinearity(),
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
        pooled_xs = torch.zeros(self.max_dim + 1, batch_size, xs[0].size(-1),
            device=batch_size.device)
        for i in range(len(xs)):
            # It's very important that size is supplied.
            pooled_xs[i, :, :] = self.pooling_fn(xs[i], data.chains[i].batch, size=batch_size)
        return pooled_xs

    def jump_complex(self, jump_xs):
        # Perform JumpingKnowledge at each level of the complex
        xs = []
        for jumpx in jump_xs:
            xs += [self.jump(jumpx)]
        return xs

    def forward(self, data: ComplexBatch):
        model_nonlinearity = get_nonlinearity(self.nonlinearity, return_module=False)
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

        x = model_nonlinearity(self.lin1(x))
        x = F.dropout(x, p=self.dropout_rate, training=self.training)
        x = self.lin2(x)
        return x

    def __repr__(self):
        return self.__class__.__name__


class SparseSIN(torch.nn.Module):
    """
    A simplicial version of GIN.

    This model is based on
    https://github.com/rusty1s/pytorch_geometric/blob/master/benchmark/kernel/gin.py
    """

    def __init__(self, num_input_features, num_classes, num_layers, hidden,
                 dropout_rate: float = 0.5,
                 max_dim: int = 2, jump_mode=None, nonlinearity='relu', readout='sum',
                 train_eps=False, final_hidden_multiplier: int = 2,
                 readout_dims=(0, 2, 3)):
        super(SparseSIN, self).__init__()

        self.max_dim = max_dim
        if readout_dims is not None:
            self.readout_dims = tuple([dim for dim in readout_dims if dim <= max_dim])
        else:
            self.readout_dims = list(range(max_dim+1))
        self.dropout_rate = dropout_rate
        self.jump_mode = jump_mode
        self.convs = torch.nn.ModuleList()
        self.nonlinearity = nonlinearity
        self.pooling_fn = get_pooling_fn(readout)
        act_module = get_nonlinearity(nonlinearity, return_module=True)
        for i in range(num_layers):
            layer_dim = num_input_features if i == 0 else hidden
            conv_update_up = Sequential(
                Linear(layer_dim, hidden),
                act_module(),
                Linear(hidden, hidden),
                act_module())
            conv_update_faces = Sequential(
                Linear(layer_dim, hidden),
                act_module(),
                Linear(hidden, hidden),
                act_module())
            self.convs.append(
                SparseSINConv(up_msg_size=layer_dim, down_msg_size=layer_dim,
                    msg_faces_nn=lambda x: x, msg_up_nn=lambda x1, x2: x1,
                    inp_update_up_nn=None, inp_update_faces_nn=None,
                    train_eps=train_eps, max_dim=self.max_dim,
                    hidden=hidden, act_module=act_module, layer_dim=layer_dim))
        self.jump = JumpingKnowledge(jump_mode) if jump_mode is not None else None
        self.lin1s = torch.nn.ModuleList()
        for _ in range(max_dim + 1):
            if jump_mode == 'cat':
                # These layers don't use a bias. Then, in case a level is not present the output
                # is just zero and it is not given by the biases.
                self.lin1s.append(Linear(num_layers * hidden, final_hidden_multiplier * hidden,
                    bias=False))
            else:
                self.lin1s.append(Linear(hidden, final_hidden_multiplier * hidden))
        self.lin2 = Linear(final_hidden_multiplier * hidden, num_classes)

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
        # print(batch_size)
        # The MP output is of shape [message_passing_dim, batch_size, feature_dim]
        pooled_xs = torch.zeros(self.max_dim + 1, batch_size, xs[0].size(-1),
            device=batch_size.device)
        for i in range(len(xs)):
            # It's very important that size is supplied.
            pooled_xs[i, :, :] = self.pooling_fn(xs[i], data.chains[i].batch, size=batch_size)

        new_xs = []
        for i in range(self.max_dim + 1):
            new_xs.append(pooled_xs[i])
        return new_xs

    def jump_complex(self, jump_xs):
        # Perform JumpingKnowledge at each level of the complex
        xs = []
        for jumpx in jump_xs:
            xs += [self.jump(jumpx)]
        return xs

    def forward(self, data: ComplexBatch, include_partial=False):
        act = get_nonlinearity(self.nonlinearity, return_module=False)

        xs, jump_xs = None, None
        res = {}
        for i, conv in enumerate(self.convs):
            params = data.get_all_chain_params(max_dim=self.max_dim, include_down_features=False)
            start_to_process = 0
            # if i == len(self.convs) - 2:
            #     start_to_process = 1
            # if i == len(self.convs) - 1:
            #     start_to_process = 2
            xs = conv(*params, start_to_process=start_to_process)
            data.set_xs(xs)

            if include_partial:
                for k in range(len(xs)):
                    res[f"layer{i}_{k}"] = xs[k]

            if self.jump_mode is not None:
                if jump_xs is None:
                    jump_xs = [[] for _ in xs]
                for i, x in enumerate(xs):
                    jump_xs[i] += [x]

        if self.jump_mode is not None:
            xs = self.jump_complex(jump_xs)

        xs = self.pool_complex(xs, data)
        # Select the dimensions we want at the end.
        xs = [xs[i] for i in self.readout_dims]

        if include_partial:
            for k in range(len(xs)):
                res[f"pool_{k}"] = xs[k]

        new_xs = []
        for i, x in enumerate(xs):
            new_xs.append(act(self.lin1s[self.readout_dims[i]](x)))

        x = torch.stack(new_xs, dim=0)
        x = x.sum(0)
        x = F.dropout(x, p=self.dropout_rate, training=self.training)

        x = self.lin2(x)

        if include_partial:
            res['out'] = x
            return x, res
        return x

    def __repr__(self):
        return self.__class__.__name__


class EdgeSIN0(torch.nn.Module):
    """
    A variant of SIN0 operating up to edge level. It may optionally ignore triangle features.

    This model is based on
    https://github.com/rusty1s/pytorch_geometric/blob/master/benchmark/kernel/gin.py
    """

    def __init__(self, num_input_features, num_classes, num_layers, hidden,
                 dropout_rate: float = 0.5,
                 jump_mode=None, nonlinearity='relu', include_top_features=True,
                 update_top_features=True,
                 readout='sum'):
        super(EdgeSIN0, self).__init__()

        self.max_dim = 1
        self.include_top_features = include_top_features
        # If the top features are included, then they can be updated by a network.
        self.update_top_features = include_top_features and update_top_features
        self.dropout_rate = dropout_rate
        self.jump_mode = jump_mode
        self.convs = torch.nn.ModuleList()
        self.update_top_nns = torch.nn.ModuleList()
        self.nonlinearity = nonlinearity
        self.pooling_fn = get_pooling_fn(readout)
        conv_nonlinearity = get_nonlinearity(nonlinearity, return_module=True)
        for i in range(num_layers):
            layer_dim = num_input_features if i == 0 else hidden
            v_conv_update = Sequential(
                Linear(layer_dim, hidden),
                conv_nonlinearity(),
                Linear(hidden, hidden),
                conv_nonlinearity(),
                BN(hidden))
            e_conv_update = Sequential(
                Linear(layer_dim, hidden),
                conv_nonlinearity(),
                Linear(hidden, hidden),
                conv_nonlinearity(),
                BN(hidden))
            v_conv_up = Sequential(
                Linear(layer_dim * 2, layer_dim),
                conv_nonlinearity(),
                BN(layer_dim))
            e_conv_down = Sequential(
                Linear(layer_dim * 2, layer_dim),
                conv_nonlinearity(),
                BN(layer_dim))
            e_conv_inp_dim = layer_dim * 2 if include_top_features else layer_dim
            e_conv_up = Sequential(
                Linear(e_conv_inp_dim, layer_dim),
                conv_nonlinearity(),
                BN(layer_dim))
            self.convs.append(
                EdgeSINConv(layer_dim, layer_dim, v_conv_up, e_conv_down, e_conv_up,
                    v_conv_update, e_conv_update, train_eps=False))
            if self.update_top_features and i < num_layers - 1:
                self.update_top_nns.append(Sequential(
                    Linear(layer_dim, hidden),
                    conv_nonlinearity(),
                    Linear(hidden, hidden),
                    conv_nonlinearity(),
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
        for net in self.update_top_nns:
            net.reset_parameters()

    def pool_complex(self, xs, data):
        # All complexes have nodes so we can extract the batch size from chains[0]
        batch_size = data.chains[0].batch.max() + 1
        # The MP output is of shape [message_passing_dim, batch_size, feature_dim]
        pooled_xs = torch.zeros(self.max_dim + 1, batch_size, xs[0].size(-1),
            device=batch_size.device)
        for i in range(len(xs)):
            # It's very important that size is supplied.
            pooled_xs[i, :, :] = self.pooling_fn(xs[i], data.chains[i].batch, size=batch_size)
        return pooled_xs

    def jump_complex(self, jump_xs):
        # Perform JumpingKnowledge at each level of the complex
        xs = []
        for jumpx in jump_xs:
            xs += [self.jump(jumpx)]
        return xs

    def forward(self, data: ComplexBatch):
        model_nonlinearity = get_nonlinearity(self.nonlinearity, return_module=False)
        xs, jump_xs = None, None
        for i, conv in enumerate(self.convs):
            params = data.get_all_chain_params(max_dim=self.max_dim,
                include_top_features=self.include_top_features)
            xs = conv(*params)
            # If we are at the last convolutional layer, we do not need to update after
            # We also check triangle features do indeed exist in this batch before doing this.
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

        x = model_nonlinearity(self.lin1(x))
        x = F.dropout(x, p=self.dropout_rate, training=self.training)
        x = self.lin2(x)
        return x

    def __repr__(self):
        return self.__class__.__name__


class Dummy(torch.nn.Module):
    """
    A dummy simplicial network model.
    No parameters in the convolutional layers.
    Readout at each layer is by summation.
    Outputs are computed by one single linear layer.
    """

    def __init__(self, num_input_features, num_classes, num_layers, max_dim: int = 2,
                 readout='sum'):
        super(Dummy, self).__init__()

        self.max_dim = max_dim
        self.convs = torch.nn.ModuleList()
        self.pooling_fn = get_pooling_fn(readout)
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
        pooled_xs = torch.zeros(self.max_dim + 1, batch_size, xs[0].size(-1),
            device=batch_size.device)
        for i in range(len(xs)):
            # It's very important that size is supplied.
            # Otherwise, if we have complexes with no simplices at certain levels, the wrong
            # shape could be inferred automatically from data.chains[i].batch.
            # This makes sure the output tensor will have the right dimensions.
            pooled_xs[i, :, :] = self.pooling_fn(xs[i], data.chains[i].batch, size=batch_size)
        # Reduce across the levels of the complexes
        x = pooled_xs.sum(dim=0)

        x = self.lin(x)
        return x

    def __repr__(self):
        return self.__class__.__name__
