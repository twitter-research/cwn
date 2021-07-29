import torch

from mp.layers import SparseCINConv
from mp.nn import get_nonlinearity, get_graph_norm
from data.complex import ComplexBatch
from torch.nn import Linear, Sequential
from torch_geometric.nn import GINConv


class RingSparseCIN(torch.nn.Module):
    """
    A simple cellular version of GIN employed for Ring experiments.

    This model is based on
    https://github.com/rusty1s/pytorch_geometric/blob/master/benchmark/kernel/gin.py
    """

    def __init__(self, num_input_features, num_classes, num_layers, hidden,
                 max_dim: int = 2, nonlinearity='relu', train_eps=False, use_coboundaries=False,
                 graph_norm='id'):
        super(RingSparseCIN, self).__init__()

        self.max_dim = max_dim
        self.convs = torch.nn.ModuleList()
        self.nonlinearity = nonlinearity
        self.init_layer = Linear(num_input_features, num_input_features)
        act_module = get_nonlinearity(nonlinearity, return_module=True)
        self.graph_norm = get_graph_norm(graph_norm)

        for i in range(num_layers):
            layer_dim = num_input_features if i == 0 else hidden
            self.convs.append(
                SparseCINConv(up_msg_size=layer_dim, down_msg_size=layer_dim,
                    boundary_msg_size=layer_dim, passed_msg_boundaries_nn=None, passed_msg_up_nn=None,
                    passed_update_up_nn=None, passed_update_boundaries_nn=None,
                    train_eps=train_eps, max_dim=self.max_dim,
                    hidden=hidden, act_module=act_module, layer_dim=layer_dim,
                    graph_norm=self.graph_norm, use_coboundaries=use_coboundaries))
        self.lin1 = Linear(hidden, num_classes)

    def reset_parameters(self):
        self.init_layer.reset_parameters()
        for conv in self.convs:
            conv.reset_parameters()
        self.lin1.reset_parameters()

    def forward(self, data: ComplexBatch, include_partial=False):
        xs = None
        res = {}

        data.nodes.x = self.init_layer(data.nodes.x)
        for c, conv in enumerate(self.convs):
            params = data.get_all_cochain_params(max_dim=self.max_dim, include_down_features=False)
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


class RingGIN(torch.nn.Module):
    def __init__(self, num_features, num_layers, hidden, num_classes, nonlinearity='relu',
                 graph_norm='bn'):
        super(RingGIN, self).__init__()
        self.nonlinearity = nonlinearity
        conv_nonlinearity = get_nonlinearity(nonlinearity, return_module=True)
        self.init_linear = Linear(num_features, num_features)
        self.graph_norm = get_graph_norm(graph_norm)

        # BN is needed to make GIN work empirically beyond 2 layers for the ring experiments.
        self.conv1 = GINConv(
            Sequential(
                Linear(num_features, hidden),
                self.graph_norm(hidden),
                conv_nonlinearity(),
                Linear(hidden, hidden),
                self.graph_norm(hidden),
                conv_nonlinearity(),
            ), train_eps=False)

        self.convs = torch.nn.ModuleList()
        for i in range(num_layers - 1):
            self.convs.append(
                GINConv(
                    Sequential(
                        Linear(hidden, hidden),
                        self.graph_norm(hidden),
                        conv_nonlinearity(),
                        Linear(hidden, hidden),
                        self.graph_norm(hidden),
                        conv_nonlinearity(),
                    ), train_eps=False))
        self.lin1 = Linear(hidden, num_classes)

    def reset_parameters(self):
        self.init_linear.reset_parameters()
        self.conv1.reset_parameters()
        for conv in self.convs:
            conv.reset_parameters()
        self.lin1.reset_parameters()

    def forward(self, data):
        act = get_nonlinearity(self.nonlinearity, return_module=False)
        x, edge_index, mask = data.x, data.edge_index, data.mask
        x = self.init_linear(x)
        x = act(self.conv1(x, edge_index))
        for conv in self.convs:
            x = conv(x, edge_index)
        # Select the target node of each graph in the batch
        x = x[mask]
        x = self.lin1(x)
        return x

    def __repr__(self):
        return self.__class__.__name__

