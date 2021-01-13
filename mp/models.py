from abc import ABC

from torch import Tensor
from mp.smp import ChainMessagePassing, SimplicialMessagePassing


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


class DummySimplicialMessagePassing(SimplicialMessagePassing):
    def __init__(self, vertex_mp: DummyChainMessagePassing, edge_mp: DummyChainMessagePassing,
                 triangle_mp: DummyChainMessagePassing):
        super(DummySimplicialMessagePassing, self).__init__(
            vertex_mp, edge_mp, triangle_mp)

    def forward(self, v_params, e_params, t_params):
        return self.propagate(v_params, e_params, t_params)

