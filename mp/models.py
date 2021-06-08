import torch
import torch.nn.functional as F

from torch.nn import Linear, Sequential, ReLU, BatchNorm1d as BN, LayerNorm as LN
from torch_geometric.nn import global_mean_pool, global_add_pool, JumpingKnowledge
from mp.layers import (
    SINConv, EdgeSINConv, SparseSINConv, DummySimplicialMessagePassing, OrientedConv)
from mp.nn import get_nonlinearity, get_pooling_fn, pool_complex
from data.complex import Complex, ComplexBatch, ChainBatch


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
        for c, conv in enumerate(self.convs):
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
                 train_eps=False, final_hidden_multiplier: int = 2, use_cofaces=False,
                 readout_dims=(0, 1, 2), final_readout='sum', apply_dropout_before='lin2'):
        super(SparseSIN, self).__init__()

        self.max_dim = max_dim
        if readout_dims is not None:
            self.readout_dims = tuple([dim for dim in readout_dims if dim <= max_dim])
        else:
            self.readout_dims = list(range(max_dim+1))
        self.final_readout = final_readout
        self.dropout_rate = dropout_rate
        self.apply_dropout_before = apply_dropout_before
        self.jump_mode = jump_mode
        self.convs = torch.nn.ModuleList()
        self.nonlinearity = nonlinearity
        self.pooling_fn = get_pooling_fn(readout)
        act_module = get_nonlinearity(nonlinearity, return_module=True)
        for i in range(num_layers):
            layer_dim = num_input_features if i == 0 else hidden
            # Instantiate message / update functions here if you want them to be
            # shared across dimensions. See the example below:
            # conv_update_up = Sequential(
            #     Linear(layer_dim, hidden),
            #     act_module(),
            #     Linear(hidden, hidden),
            #     act_module())
            # conv_update_faces = Sequential(
            #     Linear(layer_dim, hidden),
            #     act_module(),
            #     Linear(hidden, hidden),
            #     act_module())
            self.convs.append(
                SparseSINConv(up_msg_size=layer_dim, down_msg_size=layer_dim,
                    face_msg_size=layer_dim, msg_faces_nn=None, msg_up_nn=None,
                    inp_update_up_nn=None, inp_update_faces_nn=None,
                    train_eps=train_eps, max_dim=self.max_dim,
                    hidden=hidden, act_module=act_module, layer_dim=layer_dim,
                    apply_norm=(num_input_features>1), use_cofaces=use_cofaces))
            # TODO: turn apply_norm into a less hacky trick
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
        self.lin1s.reset_parameters()
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
        for c, conv in enumerate(self.convs):
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
                    res[f"layer{c}_{k}"] = xs[k]

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
            if self.apply_dropout_before == 'lin1':
                x = F.dropout(x, p=self.dropout_rate, training=self.training)
            new_xs.append(act(self.lin1s[self.readout_dims[i]](x)))

        x = torch.stack(new_xs, dim=0)
        
        if self.apply_dropout_before == 'final_readout':
            x = F.dropout(x, p=self.dropout_rate, training=self.training)
        if self.final_readout == 'mean':
            x = x.mean(0)
        elif self.final_readout == 'sum':
            x = x.sum(0)
        else:
            raise NotImplementedError
        if self.apply_dropout_before not in ['lin1', 'final_readout']:
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
        for c, conv in enumerate(self.convs):
            params = data.get_all_chain_params(max_dim=self.max_dim,
                include_top_features=self.include_top_features)
            xs = conv(*params)
            # If we are at the last convolutional layer, we do not need to update after
            # We also check triangle features do indeed exist in this batch before doing this.
            if self.update_top_features and c < len(self.convs) - 1 and 2 in data.chains:
                top_x = self.update_top_nns[c](data.chains[2].x)
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
        for c, conv in enumerate(self.convs):
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


class EdgeOrient(torch.nn.Module):
    """
    A model for edge-defined signals taking edge orientation into account.
    """

    def __init__(self, num_input_features, num_classes, num_layers, hidden,
                 dropout_rate: float = 0.0, jump_mode=None, nonlinearity='id', readout='sum'):
        super(EdgeOrient, self).__init__()

        self.max_dim = 1
        self.dropout_rate = dropout_rate
        self.jump_mode = jump_mode
        self.convs = torch.nn.ModuleList()
        self.nonlinearity = nonlinearity
        self.pooling_fn = get_pooling_fn(readout)
        for i in range(num_layers):
            layer_dim = num_input_features if i == 0 else hidden
            # !!!!! Biases must be set to false. Otherwise, the model is not equivariant !!!!
            update_up = Linear(layer_dim, hidden, bias=False)
            update_down = Linear(layer_dim, hidden, bias=False)
            update = Linear(layer_dim, hidden, bias=False)

            self.convs.append(
                OrientedConv(dim=1, up_msg_size=layer_dim, down_msg_size=layer_dim,
                    update_up_nn=update_up, update_down_nn=update_down, update_nn=update,
                    act_fn=get_nonlinearity(nonlinearity, return_module=False)))
        self.jump = JumpingKnowledge(jump_mode) if jump_mode is not None else None
        self.lin1 = Linear(hidden, hidden)
        self.lin2 = Linear(hidden, num_classes)

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        if self.jump_mode is not None:
            self.jump.reset_parameters()
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()

    def forward(self, data: ChainBatch, include_partial=False):
        x, jump_x = data.x, None
        for c, conv in enumerate(self.convs):
            x = conv(data)
            data.x = x

        cell_pred = x

        # To obtain orientation invariance, we take the absolute value of the features.
        batch_size = data.batch.max() + 1
        x = torch.abs(x)
        x = self.pooling_fn(x, data.batch, size=batch_size)

        # We can use relu here since we applied abs()
        x = torch.relu(self.lin1(x))
        x = F.dropout(x, p=self.dropout_rate, training=self.training)
        x = self.lin2(x)

        if include_partial:
            return x, cell_pred
        return x

    def __repr__(self):
        return self.__class__.__name__


class EdgeMPNN(torch.nn.Module):
    """
    An MPNN operating in the line graph.
    """

    def __init__(self, num_input_features, num_classes, num_layers, hidden,
                 dropout_rate: float = 0.0, jump_mode=None, nonlinearity='relu', readout='sum'):
        super(EdgeMPNN, self).__init__()

        self.max_dim = 1
        self.dropout_rate = dropout_rate
        self.jump_mode = jump_mode
        self.convs = torch.nn.ModuleList()
        self.nonlinearity = nonlinearity
        self.pooling_fn = get_pooling_fn(readout)
        for i in range(num_layers):
            layer_dim = num_input_features if i == 0 else hidden
            # We pass this lambda function to discard upper adjacencies
            update_up = lambda x: 0
            update_down = Linear(layer_dim, hidden, bias=False)
            update = Linear(layer_dim, hidden, bias=False)
            self.convs.append(
                OrientedConv(dim=1, up_msg_size=layer_dim, down_msg_size=layer_dim,
                    update_up_nn=update_up, update_down_nn=update_down, update_nn=update,
                    act_fn=get_nonlinearity(nonlinearity, return_module=False), orient=False))
        self.jump = JumpingKnowledge(jump_mode) if jump_mode is not None else None
        self.lin1 = Linear(hidden, hidden)
        self.lin2 = Linear(hidden, num_classes)

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        if self.jump_mode is not None:
            self.jump.reset_parameters()
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()

    def forward(self, data: ChainBatch, include_partial=False):
        data.x = torch.abs(data.x)
        for c, conv in enumerate(self.convs):
            x = conv(data)
            data.x = x
        cell_pred = x

        batch_size = data.batch.max() + 1
        x = self.pooling_fn(x, data.batch, size=batch_size)

        x = torch.relu(self.lin1(x))
        x = F.dropout(x, p=self.dropout_rate, training=self.training)
        x = self.lin2(x)

        if include_partial:
            return x, cell_pred
        return x

    def __repr__(self):
        return self.__class__.__name__


class MessagePassingAgnostic(torch.nn.Module):
    """
    A model which does not perform any message passing.
    Initial simplicial/cell representations are obtained by applying a dense layer, instead.
    Sort of resembles a 'DeepSets'-likes architecture but on Simplicial/Cell Complexes.
    """
    def __init__(self, num_input_features, num_classes, hidden, dropout_rate: float = 0.5,
                 max_dim: int = 2, nonlinearity='relu', readout='sum'):
        super(MessagePassingAgnostic, self).__init__()

        self.max_dim = max_dim
        self.dropout_rate = dropout_rate
        self.readout_type = readout
        self.act = get_nonlinearity(nonlinearity, return_module=False)
        self.lin0s = torch.nn.ModuleList()
        for dim in range(max_dim + 1):
            self.lin0s.append(Linear(num_input_features, hidden))
        self.lin1 = Linear(hidden, hidden)
        self.lin2 = Linear(hidden, num_classes)

    def reset_parameters(self):
        for lin0 in self.lin0s:
            lin0.reset_parameters()
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()

    def forward(self, data: ComplexBatch):
        
        params = data.get_all_chain_params(max_dim=self.max_dim, include_down_features=False)
        xs = list()
        for dim in range(len(params)):
            x_dim = params[dim].x
            x_dim = self.lin0s[dim](x_dim)
            xs.append(self.act(x_dim))
        pooled_xs = pool_complex(xs, data, self.max_dim, self.readout_type)
        pooled_xs = self.act(self.lin1(pooled_xs))
        x = pooled_xs.sum(dim=0)
        # TODO: eventually remove the following comment
        # NB: as an alternative, we can instead defer the application of lin1
        # to here, after final readout. As default option we instead apply lin1
        # before that, as it is more similiar to SparseSIN as an approach.
        # That was the original implementation used in rebuttal. However,
        # according to May experiments there were no differences in results
        # over the SR benchmark.
        # x = model_nonlinearity(self.lin1(x))
        x = F.dropout(x, p=self.dropout_rate, training=self.training)
        x = self.lin2(x)
        return x

    def __repr__(self):
        return self.__class__.__name__
