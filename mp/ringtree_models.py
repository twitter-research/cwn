import torch

from mp.layers import SparseSINConv
from mp.nn import get_nonlinearity
from data.complex import ComplexBatch
from torch.nn import Linear, Sequential, BatchNorm1d as BN
from torch_geometric.nn import GINConv


class RingTreeSparseSIN(torch.nn.Module):
    """
    A simplicial version of GIN.

    This model is based on
    https://github.com/rusty1s/pytorch_geometric/blob/master/benchmark/kernel/gin.py
    """

    def __init__(self, num_input_features, num_classes, num_layers, hidden,
                 max_dim: int = 2, nonlinearity='relu', train_eps=False, use_cofaces=False):
        super(RingTreeSparseSIN, self).__init__()

        self.max_dim = max_dim
        self.convs = torch.nn.ModuleList()
        self.nonlinearity = nonlinearity
        act_module = get_nonlinearity(nonlinearity, return_module=True)
        for i in range(num_layers):
            layer_dim = num_input_features if i == 0 else hidden
            self.convs.append(
                SparseSINConv(up_msg_size=layer_dim, down_msg_size=layer_dim,
                    face_msg_size=layer_dim, msg_faces_nn=None, msg_up_nn=None,
                    inp_update_up_nn=None, inp_update_faces_nn=None,
                    train_eps=train_eps, max_dim=self.max_dim,
                    hidden=hidden, act_module=act_module, layer_dim=layer_dim,
                    apply_norm=(num_input_features>1), use_cofaces=use_cofaces))
        self.lin1 = Linear(hidden, num_classes)

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        self.lin1.reset_parameters()

    def forward(self, data: ComplexBatch, include_partial=False):
        xs = None
        res = {}
        for c, conv in enumerate(self.convs):
            params = data.get_all_chain_params(max_dim=self.max_dim, include_down_features=False)
            xs = conv(*params)
            data.set_xs(xs)

            if include_partial:
                for k in range(len(xs)):
                    res[f"layer{c}_{k}"] = xs[k]

        x = xs[0]
        # Extract the target node from each graph
        mask = data.nodes.mask
        x = self.lin1(x[mask])

        if include_partial:
            res['out'] = x
            return x, res

        return x

    def __repr__(self):
        return self.__class__.__name__


class RingTreeGIN(torch.nn.Module):
    def __init__(self, num_features, num_layers, hidden, num_classes, nonlinearity='relu'):
        super(RingTreeGIN, self).__init__()
        self.nonlinearity = nonlinearity
        conv_nonlinearity = get_nonlinearity(nonlinearity, return_module=True)
        self.conv1 = GINConv(
            Sequential(
                Linear(num_features, hidden),
                BN(hidden),
                conv_nonlinearity(),
                Linear(hidden, hidden),
                BN(hidden),
                conv_nonlinearity(),
            ), train_eps=True)
        self.convs = torch.nn.ModuleList()
        for i in range(num_layers - 1):
            self.convs.append(
                GINConv(
                    Sequential(
                        Linear(hidden, hidden),
                        BN(hidden),
                        conv_nonlinearity(),
                        Linear(hidden, hidden),
                        BN(hidden),
                        conv_nonlinearity(),
                    ), train_eps=True))
        self.lin1 = Linear(hidden, num_classes)

    def reset_parameters(self):
        self.conv1.reset_parameters()
        for conv in self.convs:
            conv.reset_parameters()
        self.lin1.reset_parameters()

    def forward(self, data):
        x, edge_index, mask = data.x, data.edge_index, data.mask
        x = self.conv1(x, edge_index)
        for conv in self.convs:
            x = conv(x, edge_index)
        # Select the target node of each graph in the batch
        x = x[mask]
        x = self.lin1(x)
        return x

    def __repr__(self):
        return self.__class__.__name__

