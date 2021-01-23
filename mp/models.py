import torch
import torch.nn.functional as F

from torch.nn import Linear, Sequential, ReLU, BatchNorm1d as BN
from torch_geometric.nn import global_mean_pool
from mp.layers import SINConv
from data.complex import Complex, ComplexBatch


class SIN(torch.nn.Module):
    """
    A simplicial version of GIN.

    This model is based on
    https://github.com/rusty1s/pytorch_geometric/blob/master/benchmark/kernel/gin.py
    """
    def __init__(self, num_input_features, num_classes, num_layers, hidden):
        super(SIN, self).__init__()

        self.convs = torch.nn.ModuleList()
        for i in range(num_layers):
            input_dim = num_input_features if i == 0 else hidden
            conv_update = Sequential(
                Linear(input_dim, hidden),
                ReLU(),
                Linear(hidden, hidden),
                ReLU(),
                BN(hidden))
            conv_up = Sequential(
                Linear(input_dim * 2, hidden),
                ReLU(),
                BN(hidden))
            conv_down = Sequential(
                Linear(input_dim * 2, hidden),
                ReLU(),
                BN(hidden))
            self.convs.append(
                SINConv(conv_up, conv_down, conv_update, train_eps=False))
        self.lin1 = Linear(hidden, hidden)
        self.lin2 = Linear(hidden, num_classes)

    def reset_parameters(self):
        self.conv1.reset_parameters()
        for conv in self.convs:
            conv.reset_parameters()
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()

    def forward(self, input_data: ComplexBatch):
        data = input_data
        xs = None

        for i, conv in enumerate(self.convs):
            params = data.get_all_chain_params()
            xs = conv(*params)
            # Create a new batch with the new features from the previous layer.
            if i < len(self.convs) - 1:
                data = ComplexBatch.from_features(xs, data)

        # All complexes have nodes so we can extract the batch size from chains[0]
        batch_size = data.chains[0].batch.max() + 1
        # The MP output is of shape [complex_dim, batch_size, feature_dim]
        # We assume that all layers have the same feature size.
        pooled_xs = torch.empty(data.max_dim, batch_size, xs[0].size(-1))
        for i in range(len(xs)):
            # It's very important that size is supplied.
            # Otherwise, if we have complexes with no simplices at certain levels, the wrong
            # shape could be inferred automatically from data.chains[i].batch.
            # This makes sure the output tensor will have the right dimensions.
            pooled_xs[i, :, :] = global_mean_pool(xs[i], data.chains[i].batch, size=batch_size)
        # Reduce across the levels of the complexes
        x = pooled_xs.sum(dim=0)

        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin2(x)
        return F.log_softmax(x, dim=-1)

    def __repr__(self):
        return self.__class__.__name__
