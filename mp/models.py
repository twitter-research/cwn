import torch

from typing import Callable
from torch import Tensor
from mp.smp import ChainMessagePassing, ChainMessagePassingParams
from torch_geometric.nn.inits import reset


class DummyChainMessagePassing(ChainMessagePassing):
    """This is a dummy parameter-free message passing model used for testing."""

    def message_up(self, up_x_j: Tensor, up_attr: Tensor) -> Tensor:
        # (num_up_adj, x_feature_dim) + (num_up_adj, up_feat_dim)
        # We assume the feature dim is the same across al levels
        return up_x_j + up_attr

    def message_down(self, down_x_j: Tensor, down_attr: Tensor) -> Tensor:
        # (num_down_adj, x_feature_dim) + (num_down_adj, down_feat_dim)
        # We assume the feature dim is the same across al levels
        return down_x_j + down_attr

    def forward(self, chain: ChainMessagePassingParams):
        return self.propagate(chain.up_index, chain.down_index, x=chain.x,
                              up_attr=chain.kwargs['up_attr'], down_attr=chain.kwargs['down_attr'])


class DummySimplicialMessagePassing(torch.nn.Module):
    def __init__(self, ):
        super(DummySimplicialMessagePassing, self).__init__()
        self.vertex_mp = DummyChainMessagePassing()
        self.edge_mp = DummyChainMessagePassing()
        self.triangle_mp = DummyChainMessagePassing()

    def forward(self, v_params, e_params, t_params):
        x_out = self.vertex_mp.forward(v_params)
        e_out = self.edge_mp.forward(e_params)
        t_out = self.triangle_mp.forward(t_params)
        return x_out, e_out, t_out


class SINChainConv(ChainMessagePassing):
    """This is a dummy parameter-free message passing model used for testing."""
    def __init__(self, msg_nn: Callable, update_nn: Callable,
                 eps: float = 0., train_eps: bool = False):
        super(SINChainConv, self).__init__()
        self.msg_nn = msg_nn
        self.update_nn = update_nn
        self.initial_eps = eps
        if train_eps:
            self.eps = torch.nn.Parameter(torch.Tensor([eps]))
        else:
            self.register_buffer('eps', torch.Tensor([eps]))
        self.reset_parameters()

    def forward(self, chain: ChainMessagePassingParams):
        out = self.propagate(chain.up_index, chain.down_index, x=chain.x,
                             up_attr=chain.kwargs['up_attr'], down_attr=chain.kwargs['down_attr'])
        out += (1 + self.eps) * chain.x
        return self.update_nn(out)

    def reset_parameters(self):
        reset(self.msg_nn)
        reset(self.update_nn)
        self.eps.data.fill_(self.initial_eps)

    def __message__(self, x_j, attr):
        x = torch.cat([x_j, attr], dim=-1)
        return self.msg_nn(x)

    def message_up(self, up_x_j: Tensor, up_attr: Tensor) -> Tensor:
        return self.__message__(up_x_j, up_attr)

    def message_down(self, down_x_j: Tensor, down_attr: Tensor) -> Tensor:
        return self.__message__(down_x_j, down_attr)


class SINConv(torch.nn.Module):
    def __init__(self, msg_nn: Callable, update_nn: Callable, eps: float = 0.,
                 train_eps: bool = False):
        super(SINConv, self).__init__()
        self.vertex_mp = SINChainConv(msg_nn, update_nn, eps, train_eps)
        self.edge_mp = SINChainConv(msg_nn, update_nn, eps, train_eps)
        self.triangle_mp = SINChainConv(msg_nn, update_nn, eps, train_eps)

    def forward(self, v_params, e_params, t_params):
        x_out = self.vertex_mp.forward(v_params)
        e_out = self.edge_mp.forward(e_params)
        t_out = self.triangle_mp.forward(t_params)
        return x_out, e_out, t_out

