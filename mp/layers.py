import torch

from typing import Callable, Optional
from torch import Tensor
from mp.smp import ChainMessagePassing, ChainMessagePassingParams
from torch_geometric.nn.inits import reset
from torch.nn import Linear, Sequential, BatchNorm1d as BN
from data.complex import Chain
from torch_scatter import scatter


class DummyChainMessagePassing(ChainMessagePassing):
    """This is a dummy parameter-free message passing model used for testing."""
    def __init__(self, up_msg_size, down_msg_size, face_msg_size=None,
                 use_face_msg=False, use_down_msg=True):
        super(DummyChainMessagePassing, self).__init__(up_msg_size, down_msg_size,
                                                       face_msg_size=face_msg_size,
                                                       use_face_msg=use_face_msg,
                                                       use_down_msg=use_down_msg)

    def message_up(self, up_x_j: Tensor, up_attr: Tensor) -> Tensor:
        # (num_up_adj, x_feature_dim) + (num_up_adj, up_feat_dim)
        # We assume the feature dim is the same across al levels
        return up_x_j + up_attr

    def message_down(self, down_x_j: Tensor, down_attr: Tensor) -> Tensor:
        # (num_down_adj, x_feature_dim) + (num_down_adj, down_feat_dim)
        # We assume the feature dim is the same across al levels
        return down_x_j + down_attr

    def forward(self, chain: ChainMessagePassingParams):
        up_out, down_out, face_out = self.propagate(chain.up_index, chain.down_index, 
                                                    chain.face_index, x=chain.x,
                                                    up_attr=chain.kwargs['up_attr'],
                                                    down_attr=chain.kwargs['down_attr'],
                                                    face_attr=chain.kwargs['face_attr'])
        # down or face will be zero if one of them is not used.
        return chain.x + up_out + down_out + face_out


class DummySimplicialMessagePassing(torch.nn.Module):
    def __init__(self, input_dim=1, max_dim: int = 2, use_face_msg=False, use_down_msg=True):
        super(DummySimplicialMessagePassing, self).__init__()
        self.max_dim = max_dim
        self.mp_levels = torch.nn.ModuleList()
        for dim in range(max_dim+1):
            mp = DummyChainMessagePassing(input_dim, input_dim, face_msg_size=input_dim,
                                          use_face_msg=use_face_msg, use_down_msg=use_down_msg)
            self.mp_levels.append(mp)
    
    def forward(self, *chain_params: ChainMessagePassingParams):
        assert len(chain_params) <= self.max_dim+1

        out = []
        for dim in range(len(chain_params)):
            out.append(self.mp_levels[dim].forward(chain_params[dim]))
        return out


class SINChainConv(ChainMessagePassing):
    """This is a dummy parameter-free message passing model used for testing."""
    def __init__(self, up_msg_size: int, down_msg_size: int,
                 msg_up_nn: Callable, msg_down_nn: Callable, update_nn: Callable,
                 eps: float = 0., train_eps: bool = False):
        super(SINChainConv, self).__init__(up_msg_size, down_msg_size, use_face_msg=False)
        self.msg_up_nn = msg_up_nn
        self.msg_down_nn = msg_down_nn
        self.update_nn = update_nn
        self.initial_eps = eps
        if train_eps:
            self.eps = torch.nn.Parameter(torch.Tensor([eps]))
        else:
            self.register_buffer('eps', torch.Tensor([eps]))
        self.reset_parameters()

    def forward(self, chain: ChainMessagePassingParams):
        out_up, out_down, _ = self.propagate(chain.up_index, chain.down_index, 
                                             None, x=chain.x,
                                             up_attr=chain.kwargs['up_attr'],
                                             down_attr=chain.kwargs['down_attr'])

        # TODO: This is probably not injective, so we should do something else.
        out_up += (1 + self.eps) * chain.x
        out_down += (1 + self.eps) * chain.x
        return self.update_nn(out_up + out_down)

    def reset_parameters(self):
        reset(self.msg_up_nn)
        reset(self.msg_down_nn)
        reset(self.update_nn)
        self.eps.data.fill_(self.initial_eps)

    def message_up(self, up_x_j: Tensor, up_attr: Tensor) -> Tensor:
        if up_attr is not None:
            x = torch.cat([up_x_j, up_attr], dim=-1)
            return self.msg_up_nn(x)
        else:
            return self.msg_up_nn(up_x_j)

    def message_down(self, down_x_j: Tensor, down_attr: Tensor) -> Tensor:
        x = torch.cat([down_x_j, down_attr], dim=-1)
        return self.msg_down_nn(x)


class SINConv(torch.nn.Module):
    def __init__(self, up_msg_size: int, down_msg_size: int,
                 msg_up_nn: Callable, msg_down_nn: Callable, update_nn: Callable,
                 eps: float = 0., train_eps: bool = False, max_dim: int = 2):
        super(SINConv, self).__init__()
        self.max_dim = max_dim
        self.mp_levels = torch.nn.ModuleList()
        for dim in range(max_dim+1):
            mp = SINChainConv(up_msg_size, down_msg_size,
                              msg_up_nn, msg_down_nn, update_nn, eps, train_eps)
            self.mp_levels.append(mp)

    def forward(self, *chain_params: ChainMessagePassingParams):
        assert len(chain_params) <= self.max_dim+1

        out = []
        for dim in range(len(chain_params)):
            out.append(self.mp_levels[dim].forward(chain_params[dim]))
        return out


class EdgeSINConv(torch.nn.Module):
    """
    SIN convolutional layer which performs chain message passing only
    _up to_ 1-dimensional simplices (edges).
    """
    def __init__(self, up_msg_size: int, down_msg_size: int,
                 v_msg_up_nn: Callable, e_msg_down_nn: Callable, e_msg_up_nn: Callable,
                 v_update_nn: Callable, e_update_nn: Callable, eps: float = 0., train_eps=False):
        super(EdgeSINConv, self).__init__()
        self.max_dim = 1
        self.mp_levels = torch.nn.ModuleList()

        v_mp = SINChainConv(up_msg_size, down_msg_size,
                            v_msg_up_nn, lambda *args: None, v_update_nn, eps, train_eps)
        e_mp = SINChainConv(up_msg_size, down_msg_size,
                            e_msg_up_nn, e_msg_down_nn, e_update_nn, eps, train_eps)
        self.mp_levels.extend([v_mp, e_mp])

    def forward(self, *chain_params: ChainMessagePassingParams):
        assert len(chain_params) <= self.max_dim+1

        out = []
        for dim in range(len(chain_params)):
            out.append(self.mp_levels[dim].forward(chain_params[dim]))
        return out


class SparseSINChainConv(ChainMessagePassing):
    """This is a SIN Chain layer that operates of faces and upper adjacent simplices."""
    def __init__(self, dim: int, up_msg_size: int, down_msg_size: int, face_msg_size: Optional[int],
                 msg_up_nn: Callable, msg_faces_nn: Callable, update_up_nn: Callable, update_faces_nn,
                 combine_nn: Callable, eps: float = 0., train_eps: bool = False):
        super(SparseSINChainConv, self).__init__(up_msg_size, down_msg_size, face_msg_size=face_msg_size,
                                                 use_down_msg=False)
        self.dim = dim
        self.msg_up_nn = msg_up_nn
        self.msg_faces_nn = msg_faces_nn
        self.update_up_nn = update_up_nn
        self.update_faces_nn = update_faces_nn
        self.combine_nn = combine_nn
        self.initial_eps = eps
        if train_eps:
            self.eps1 = torch.nn.Parameter(torch.Tensor([eps]))
            self.eps2 = torch.nn.Parameter(torch.Tensor([eps]))
        else:
            self.register_buffer('eps1', torch.Tensor([eps]))
            self.register_buffer('eps2', torch.Tensor([eps]))
        self.reset_parameters()

    def forward(self, chain: ChainMessagePassingParams):
        out_up, _, out_faces = self.propagate(chain.up_index, chain.down_index, 
                                              chain.face_index, x=chain.x,
                                              up_attr=chain.kwargs['up_attr'],
                                              face_attr=chain.kwargs['face_attr'])

        # As in GIN, we can learn an injective update function for each multi-set
        out_up += (1 + self.eps1) * chain.x
        out_faces += (1 + self.eps2) * chain.x
        out_up = self.update_up_nn(out_up)
        out_faces = self.update_faces_nn(out_faces)

        # We need to combine the two such that the output is injective
        # Because the cross product of countable spaces is countable, then such a function exists.
        # And we can learn it with another MLP.
        return self.combine_nn(torch.cat([out_up, out_faces], dim=-1))

    def reset_parameters(self):
        reset(self.msg_up_nn)
        reset(self.msg_faces_nn)
        reset(self.update_up_nn)
        reset(self.update_faces_nn)
        reset(self.combine_nn)
        self.eps1.data.fill_(self.initial_eps)
        self.eps2.data.fill_(self.initial_eps)

    def message_up(self, up_x_j: Tensor, up_attr: Tensor) -> Tensor:
        # return self.msg_up_nn(torch.cat([up_x_j, up_attr], dim=-1))
        return self.msg_up_nn((up_x_j, up_attr))
    
    def message_faces(self, face_x_j: Tensor) -> Tensor:
        return self.msg_up_faces(faces_x_j)

    # def message_and_aggregate_faces(self, face_attr: Tensor) -> Tensor:
    #     shape = face_attr.size()
    #     x = face_attr.view(shape[0] * shape[1], -1)
    #     x = self.msg_faces_nn(x)
    #     return x.view(shape).sum(1)

class Catter(torch.nn.Module):
    def __init__(self):
        super(Catter, self).__init__()
    def forward(self, x):
        return torch.cat(x, dim=-1)
    
class SparseSINConv(torch.nn.Module):
    """A simplicial version of GIN which performs message passing from  simplicial upper
    neighbors and faces, but not from lower neighbors (hence why "Sparse")
    """

    def __init__(self, up_msg_size: int, down_msg_size: int, face_msg_size: Optional[int],
                 msg_up_nn: Callable, msg_faces_nn: Callable, inp_update_up_nn: Callable,
                 inp_update_faces_nn: Callable, eps: float = 0., train_eps: bool = False,
                 max_dim: int = 2, apply_norm=True, **kwargs):
        super(SparseSINConv, self).__init__()
        self.max_dim = max_dim
        self.mp_levels = torch.nn.ModuleList()
        for dim in range(max_dim+1):
            if msg_up_nn is None:
                msg_up_nn = Sequential(
                        Catter(),
                        Linear(kwargs['layer_dim'] * 2, kwargs['layer_dim']),
                        kwargs['act_module']()
                )
                # if apply_norm:
                #     msg_up_nn = Sequential(
                #         Catter(),
                #         Linear(kwargs['layer_dim'] * 2, kwargs['hidden']),
                #         BN(kwargs['hidden']),
                #         kwargs['act_module'](),
                #         Linear(kwargs['hidden'], kwargs['layer_dim']),
                #         BN(kwargs['layer_dim']),
                #         kwargs['act_module']()
                #     )
                # else:
                #     msg_up_nn = Sequential(
                #         Catter(),
                #         Linear(kwargs['layer_dim'] * 2, kwargs['hidden']),
                #         kwargs['act_module'](),
                #         Linear(kwargs['hidden'], kwargs['layer_dim']),
                #         kwargs['act_module']()
                #     )
            if msg_faces_nn is None:
                msg_faces_nn = lambda x: x
                # if apply_norm:
                #     msg_faces_nn = Sequential(
                #         Linear(kwargs['layer_dim'], kwargs['hidden']),
                #         BN(kwargs['hidden']),
                #         kwargs['act_module'](),
                #         Linear(kwargs['hidden'], kwargs['layer_dim']),
                #         BN(kwargs['layer_dim']),
                #         kwargs['act_module']()
                #     )
                # else:
                #     msg_faces_nn = Sequential(
                #         Linear(kwargs['layer_dim'], kwargs['hidden']),
                #         kwargs['act_module'](),
                #         Linear(kwargs['hidden'], kwargs['layer_dim']),
                #         kwargs['act_module']()
                #     )
            # ---
            if inp_update_up_nn is None:
                if apply_norm:
                    update_up_nn = Sequential(
                        Linear(kwargs['layer_dim'], kwargs['hidden']),
                        BN(kwargs['hidden']),
                        kwargs['act_module'](),
                        Linear(kwargs['hidden'], kwargs['hidden']),
                        BN(kwargs['hidden']),
                        kwargs['act_module']()
                    )
                else:
                    update_up_nn = Sequential(
                        Linear(kwargs['layer_dim'], kwargs['hidden']),
                        kwargs['act_module'](),
                        Linear(kwargs['hidden'], kwargs['hidden']),
                        kwargs['act_module']()
                    )
            else:
                update_up_nn = inp_update_up_nn
            if inp_update_faces_nn is None:
                if apply_norm:
                    update_faces_nn = Sequential(
                        Linear(kwargs['layer_dim'], kwargs['hidden']),
                        BN(kwargs['hidden']),
                        kwargs['act_module'](),
                        Linear(kwargs['hidden'], kwargs['hidden']),
                        BN(kwargs['hidden']),
                        kwargs['act_module']()
                    )
                else:
                    update_faces_nn = Sequential(
                        Linear(kwargs['layer_dim'], kwargs['hidden']),
                        kwargs['act_module'](),
                        Linear(kwargs['hidden'], kwargs['hidden']),
                        kwargs['act_module']()
                    )
            else:
                update_faces_nn = inp_update_faces_nn
            if apply_norm:
                combine_nn = Sequential(
                    Linear(kwargs['hidden']*2, kwargs['hidden']),
                    BN(kwargs['hidden']),
                    kwargs['act_module']())
            else:
                combine_nn = Sequential(
                    Linear(kwargs['hidden']*2, kwargs['hidden']),
                    kwargs['act_module']())
            mp = SparseSINChainConv(dim, up_msg_size, down_msg_size, face_msg_size=face_msg_size,
                msg_up_nn=msg_up_nn, msg_faces_nn=msg_faces_nn, update_up_nn=update_up_nn,
                update_faces_nn=update_faces_nn, combine_nn=combine_nn, eps=eps,
                train_eps=train_eps)
            self.mp_levels.append(mp)

    def forward(self, *chain_params: ChainMessagePassingParams, start_to_process=0):
        assert len(chain_params) <= self.max_dim+1

        out = []
        for dim in range(len(chain_params)):
            if dim < start_to_process:
                out.append(chain_params[dim].x)
            else:
                out.append(self.mp_levels[dim].forward(chain_params[dim]))
        return out


# TODO(Cris): Add tests on small complexes for orientation equivariance and invariance.
class OrientedConv(ChainMessagePassing):
    def __init__(self, dim: int, up_msg_size: int, down_msg_size: int,
                 update_up_nn: Callable, update_down_nn: Callable, update_nn: Callable, act_fn,
                 orient=True):
        super(OrientedConv, self).__init__(up_msg_size, down_msg_size, use_face_msg=False)
        self.dim = dim
        self.update_up_nn = update_up_nn
        self.update_down_nn = update_down_nn
        self.update_nn = update_nn
        self.act_fn = act_fn
        self.orient = orient

    def forward(self, chain: Chain):
        assert len(chain.upper_orient) == chain.upper_index.size(1)
        assert len(chain.lower_orient) == chain.lower_index.size(1)
        assert chain.upper_index.max() < len(chain.x)
        assert chain.lower_index.max() < len(chain.x)

        out_up, out_down, _ = self.propagate(chain.upper_index, chain.lower_index, None, x=chain.x,
            up_attr=chain.upper_orient.view(-1, 1), down_attr=chain.lower_orient.view(-1, 1))

        out_up = self.update_up_nn(out_up)
        out_down = self.update_down_nn(out_down)
        x = self.update_nn(chain.x)
        return self.act_fn(x + out_up + out_down)

    def reset_parameters(self):
        reset(self.update_up_nn)
        reset(self.update_down_nn)
        reset(self.update_nn)

    # TODO: As a temporary hack, we pass the orientation through the up and down attributes.
    def message_up(self, up_x_j: Tensor, up_attr: Tensor) -> Tensor:
        if self.orient:
            return up_x_j * up_attr
        return up_x_j

    def message_down(self, down_x_j: Tensor, down_attr: Tensor) -> Tensor:
        if self.orient:
            return down_x_j * down_attr
        return down_x_j


class InitReduceConv(torch.nn.Module):

    def __init__(self, reduce='add'):
        """

        Args:
            reduce (str): Way to aggregate faces. Can be "sum, add, mean, min, max"
        """
        super(InitReduceConv, self).__init__()
        self.reduce = reduce

    def forward(self, face_x, face_index):
        features = face_x.index_select(0, face_index[0])
        out_size = face_index[1, :].max() + 1
        return scatter(features, face_index[1], dim=0, dim_size=out_size, reduce=self.reduce)


class EmbedVEWithReduce(torch.nn.Module):

    def __init__(self,
                 v_embed_layer: torch.nn.Embedding,
                 e_embed_layer: Optional[torch.nn.Embedding],
                 init_reduce: InitReduceConv):
        """

        Args:
            v_embed_layer: Layer to embed the integer features of the vertices
            e_embed_layer: Layer (potentially None) to embed the integer features of the edges.
            init_reduce: Layer to initialise the 2D cell features and potentially the edge features.
        """
        super(EmbedVEWithReduce, self).__init__()
        self.v_embed_layer = v_embed_layer
        self.e_embed_layer = e_embed_layer
        self.init_reduce = init_reduce

    def forward(self,  *chain_params: ChainMessagePassingParams):
        assert 2 <= len(chain_params) <= 3
        v_params = chain_params[0]
        e_params = chain_params[1]
        c_params = chain_params[2] if len(chain_params) == 3 else None

        assert v_params.x is not None
        # The embedding layer expects integers so we convert the tensor to int.
        assert v_params.x.size(-1) == 1
        vx = self.v_embed_layer(v_params.x.squeeze().to(dtype=torch.long))

        if e_params.x is None:
            ex = self.init_reduce(vx, e_params.face_index)
        else:
            # The embedding layer expects integers so we convert the tensor to int.
            assert e_params.x.size(-1) == 1
            ex = self.e_embed_layer(e_params.x.squeeze().to(dtype=torch.long))

        if c_params is not None:
            # We divide by two in case this was obtained from node aggregation.
            # The division should not do any harm if this is an aggregation of learned embeddings.
            cx = self.init_reduce(ex, c_params.face_index) / 2.

            return vx, ex, cx

        return vx, ex

    def reset_parameters(self):
        reset(self.v_embed_layer)
        reset(self.e_embed_layer)
