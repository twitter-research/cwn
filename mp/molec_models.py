import torch
import torch.nn.functional as F

from torch.nn import Linear, Embedding, Dropout
from torch_geometric.nn import JumpingKnowledge
from mp.layers import InitReduceConv, EmbedVEWithReduce, OGBEmbedVEWithReduce, SparseSINConv
from ogb.graphproppred.mol_encoder import AtomEncoder, BondEncoder
from data.complex import ComplexBatch
from mp.models import get_nonlinearity, get_pooling_fn


class EmbedSparseSIN(torch.nn.Module):
    """
    A simplicial version of GIN with some tailoring to nimbly work on molecules from the ZINC database.

    This model is based on
    https://github.com/rusty1s/pytorch_geometric/blob/master/benchmark/kernel/gin.py
    """

    def __init__(self, atom_types, bond_types, out_size, num_layers, hidden,
                 dropout_rate: float = 0.5, max_dim: int = 2, jump_mode=None, nonlinearity='relu',
                 readout='sum', train_eps=False, final_hidden_multiplier: int = 2,
                 readout_dims=(0, 1, 2), final_readout='sum', apply_dropout_before='lin2',
                 init_reduce='sum', embed_edge=False, embed_dim=None, use_cofaces=False):
        super(EmbedSparseSIN, self).__init__()

        self.max_dim = max_dim
        if readout_dims is not None:
            self.readout_dims = tuple([dim for dim in readout_dims if dim <= max_dim])
        else:
            self.readout_dims = list(range(max_dim+1))

        if embed_dim is None:
            embed_dim = hidden
        self.v_embed_init = Embedding(atom_types, embed_dim)

        self.e_embed_init = None
        if embed_edge:
            self.e_embed_init = Embedding(bond_types, embed_dim)
        self.reduce_init = InitReduceConv(reduce=init_reduce)
        self.init_conv = EmbedVEWithReduce(self.v_embed_init, self.e_embed_init, self.reduce_init)

        self.final_readout = final_readout
        self.dropout_rate = dropout_rate
        self.apply_dropout_before = apply_dropout_before
        self.jump_mode = jump_mode
        self.convs = torch.nn.ModuleList()
        self.nonlinearity = nonlinearity
        self.pooling_fn = get_pooling_fn(readout)
        act_module = get_nonlinearity(nonlinearity, return_module=True)
        for i in range(num_layers):
            layer_dim = embed_dim if i == 0 else hidden
            self.convs.append(
                SparseSINConv(up_msg_size=layer_dim, down_msg_size=layer_dim,
                    face_msg_size=layer_dim, msg_faces_nn=lambda x: x,
                    msg_up_nn=lambda x: x[0], inp_update_up_nn=None,
                    inp_update_faces_nn=None, train_eps=train_eps, max_dim=self.max_dim,
                    hidden=hidden, act_module=act_module, layer_dim=layer_dim,
                    apply_norm=(embed_dim>1), use_cofaces=use_cofaces)) # TODO: turn this into a less hacky trick
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
        self.lin2 = Linear(final_hidden_multiplier * hidden, out_size)

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        if self.jump_mode is not None:
            self.jump.reset_parameters()
        self.init_conv.reset_parameters()
        self.lin1s.reset_parameters()
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

        # Check input node/edge features are scalars.
        assert data.chains[0].x.size(-1) == 1
        if data.chains[1].x is not None:
            assert data.chains[1].x.size(-1) == 1

        # Embed and populate higher-levels
        params = data.get_all_chain_params(max_dim=self.max_dim, include_down_features=False)
        xs = list(self.init_conv(*params))

        # Apply dropout on the input features like all models do on ZINC.
        for i, x in enumerate(xs):
            xs[i] = F.dropout(xs[i], p=self.dropout_rate, training=self.training)

        data.set_xs(xs)

        for i, conv in enumerate(self.convs):
            params = data.get_all_chain_params(max_dim=self.max_dim, include_down_features=False)
            start_to_process = 0
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


class MolhivSparseSIN(torch.nn.Module):
    """
    A simplicial version of GIN with some tailoring to nimbly work on molecules from the ogbg-molhiv dataset

    This model is based on
    https://github.com/rusty1s/pytorch_geometric/blob/master/benchmark/kernel/gin.py
    """

    def __init__(self, out_size, num_layers, hidden,
                 dropout_rate: float = 0.5, max_dim: int = 2, jump_mode=None, nonlinearity='relu',
                 readout='sum', train_eps=False, final_hidden_multiplier: int = 2,
                 readout_dims=(0, 1, 2), final_readout='sum', apply_dropout_before='lin2',
                 init_reduce='sum', embed_edge=False, embed_dim=None, use_cofaces=False):
        super(MolhivSparseSIN, self).__init__()

        self.max_dim = max_dim
        if readout_dims is not None:
            self.readout_dims = tuple([dim for dim in readout_dims if dim <= max_dim])
        else:
            self.readout_dims = list(range(max_dim+1))

        if embed_dim is None:
            embed_dim = hidden
        self.v_embed_init = AtomEncoder(embed_dim)

        self.e_embed_init = None
        if embed_edge:
            self.e_embed_init = BondEncoder(embed_dim)
        self.reduce_init = InitReduceConv(reduce=init_reduce)
        self.init_conv = OGBEmbedVEWithReduce(self.v_embed_init, self.e_embed_init, self.reduce_init)

        self.final_readout = final_readout
        self.dropout_rate = dropout_rate
        self.apply_dropout_before = apply_dropout_before
        self.jump_mode = jump_mode
        self.convs = torch.nn.ModuleList()
        self.nonlinearity = nonlinearity
        self.pooling_fn = get_pooling_fn(readout)
        act_module = get_nonlinearity(nonlinearity, return_module=True)
        for i in range(num_layers):
            layer_dim = embed_dim if i == 0 else hidden
            self.convs.append(
                SparseSINConv(up_msg_size=layer_dim, down_msg_size=layer_dim,
                    face_msg_size=layer_dim, msg_faces_nn=lambda x: x,
                    msg_up_nn=lambda x: x[0], inp_update_up_nn=None,
                    inp_update_faces_nn=None, train_eps=train_eps, max_dim=self.max_dim,
                    hidden=hidden, act_module=act_module, layer_dim=layer_dim,
                    apply_norm=(embed_dim>1), use_cofaces=use_cofaces)) # TODO: turn this into a less hacky trick
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
        self.lin2 = Linear(final_hidden_multiplier * hidden, out_size)

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        if self.jump_mode is not None:
            self.jump.reset_parameters()
        self.init_conv.reset_parameters()
        self.lin1s.reset_parameters()
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

        # Check input node/edge features are scalars.
        # assert data.chains[0].x.size(-1) == 1, data.chains[0].x.size(-1)
        # if data.chains[1].x is not None:
        #     assert data.chains[1].x.size(-1) == 1

        # Embed and populate higher-levels
        params = data.get_all_chain_params(max_dim=self.max_dim, include_down_features=False)
        xs = list(self.init_conv(*params))

        # Apply dropout on the input features like all models do on ZINC.
        for i, x in enumerate(xs):
            xs[i] = F.dropout(xs[i], p=self.dropout_rate, training=self.training)

        data.set_xs(xs)

        for i, conv in enumerate(self.convs):
            params = data.get_all_chain_params(max_dim=self.max_dim, include_down_features=False)
            start_to_process = 0
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

