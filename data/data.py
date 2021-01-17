import torch

from torch import Tensor
from mp.smp import ChainMessagePassingParams
from torch_geometric.typing import Adj


class Chain(object):
    """
        Class representing a chain of k-dim simplices.
    """
    def __init__(self, dim: int, x: Tensor, upper_index: Adj = None, lower_index: Adj = None,
                 shared_faces: Tensor = None, shared_cofaces: Tensor = None, mapping: Tensor = None,
                 y=None):
        """
            Constructs a `dim`-chain.
            - `dim`: dim of the simplices in the chain
            - `x`: feature matrix, shape [num_simplices, num_features]; may not be available
            - `upper_index`: upper adjacency, matrix, shape [2, num_upper_connections];
            may not be available, e.g. when `dim` is the top level dim of a complex
            - `lower_index`: lower adjacency, matrix, shape [2, num_lower_connections];
            may not be available, e.g. when `dim` is the top level dim of a complex
            - `y`: labels over simplices in the chain, shape [num_simplices,]
        """
        self.dim = dim
        self.x = x
        self.upper_index = upper_index
        self.lower_index = lower_index
        self.y = y
        self.shared_faces = shared_faces
        self.shared_cofaces = shared_cofaces
        self.oriented = False
        self._hodge_laplacian = None
        # TODO: Figure out what to do with mapping.
        self.mapping = mapping

    def orient(self, arbitrary=None):
        """
            Enforces orientation to the chain.
            If `arbitrary` orientation is provided, it enforces that. Otherwise the canonical one
            is enforced.
        """
        raise NotImplementedError
        # TODO: what is the impact of this on lower/upper signals?
        # ...
        # ...
        # self.lower_orientation = ...  # <- shape [1, num_lower_connections], content: +/- 1.0
        # self.upper_orientation = ...  # <- shape [1, num_upper_connections], content: +/- 1.0
        # self.oriented = True
        # return

    def get_hodge_laplacian(self):
        """
            Returns the Hodge Laplacian.
            Orientation is required; if not present, the chain will first be oriented according
            to the canonical ordering.
        """
        raise NotImplementedError
        # if self._hodge_laplacian is None:  # caching
        #     if not self.oriented:
        #         self.orient()
        #     self._hodge_laplacian = ...
        #     # ^^^ here we need to perform two sparse matrix multiplications
        #     # -- we can leverage on torch_sparse
        #     # indices of the sparse matrices are self.lower_index and self.upper_index,
        #     # their values are those in
        #     # self.lower_orientation and self.upper_orientation
        # return self._hodge_laplacian

    def initialize_features(self, strategy='constant'):
        """
            Routine to initialize simplex-wise features based on the provided `strategy`.
        """
        raise NotImplementedError
        # self.x = ...
        # return


class Complex(object):
    """
        Class representing an attributed simplicial complex.
    """

    def __init__(self, nodes: Chain, edges: Chain, triangles: Chain, y=None):

        # TODO: This needs some data checking to check that these chains are consistent together
        self.nodes = nodes
        self.edges = edges
        self.triangles = triangles
        self.chains = {0: self.nodes, 1: self.edges, 2: self.triangles}
        self.max_dim = 2
        # ^^^ these will be abstracted later allowing for generic orders;
        # in that case they will be provided as an array (or similar) and min and
        # max orders will be set automatically, according to what passed
        # NB: nodes, edges, triangles are objects of class `Chain` with orders 0, 1, 2
        self.y = y  # complex-wise label for complex-level tasks
        return

    def get_chain_params(self, dim) -> ChainMessagePassingParams:
        """
            Conveniently returns all necessary input parameters to perform higher-dim
            neural message passing at the specified `dim`.
        """
        if dim in self.chains:
            simplices = self.chains[dim]
            x = simplices.x
            if simplices.upper_index is not None:
                upper_index = simplices.upper_index
                upper_features = self.chains[dim + 1].x
                if upper_features is not None:
                    upper_features = torch.index_select(upper_features, 0,
                                                        self.chains[dim].shared_cofaces)
            else:
                upper_index = None
                upper_features = None
            if simplices.lower_index is not None:
                lower_index = simplices.lower_index
                lower_features = self.chains[dim - 1].x
                if lower_features is not None:
                    lower_features = torch.index_select(lower_features, 0,
                                                        self.chains[dim].shared_faces)
            else:
                lower_index = None
                lower_features = None
            inputs = ChainMessagePassingParams(x, upper_index, lower_index,
                                               up_attr=upper_features, down_attr=lower_features)
        else:
            raise NotImplementedError(
                'Dim {} is not present in the complex or not yet supported.'.format(dim))
        return inputs

    def get_labels(self, dim=None):
        """
            Returns target labels.
            If `dim`==k (integer in [0, self.max_dim]) then the labels over
            k-simplices are returned.
            In the case `dim` is None the complex-wise label is returned.
        """
        if dim is None:
            y = self.y
        else:
            if dim in self.chains:
                y = self.chains[dim].y
            else:
                raise NotImplementedError(
                    'Dim {} is not present in the complex or not yet supported.'.format(dim))
        return y

