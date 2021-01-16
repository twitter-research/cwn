"""
Input: Simplicial complex of dimension d
Output: k-dim Laplacians up to dimension d
"""


import numpy as np
from scipy import sparse
from scipy.sparse import coo_matrix
from random import shuffle
import torch
import torch.nn as nn
import scipy.sparse as sp
import scipy.sparse.linalg as spl

import time


def build_boundaries(simplices):
    """Build the boundary operators from a list of simplices.
    Parameters
    ----------
    simplices: list of dictionaries
        List of dictionaries, one per dimension d. The size of the dictionary
        is the number of d-simplices. The dictionary's keys are sets (of size d
        + 1) of the 0-simplices that constitute the d-simplices. The
        dictionary's values are the indexes of the simplices in the boundary
        and Laplacian matrices.
    Returns
    -------
    boundaries: list of sparse matrices
       List of boundary operators, one per dimension: i-th boundary is in i-th position
    """
    boundaries = list()
    for d in range(1, len(simplices)):
        idx_simplices, idx_faces, values = [], [], []
        for simplex, idx_simplex in simplices[d].items():
            for i, left_out in enumerate(np.sort(list(simplex))):
                idx_simplices.append(idx_simplex)
                values.append((-1)**i)
                face = simplex.difference({left_out})
                idx_faces.append(simplices[d-1][face])
        assert len(values) == (d+1) * len(simplices[d])
        boundary = coo_matrix((values, (idx_faces, idx_simplices)),
                                     dtype=np.float32,
                                     shape=(len(simplices[d-1]), len(simplices[d])))
        boundaries.append(boundary)
    return boundaries


def build_laplacians(boundaries):
    """Build the Laplacian operators from the boundary operators.
    Parameters
    ----------
    boundaries: list of sparse matrices
       List of boundary operators, one per dimension.
    Returns
    -------
    laplacians: list of sparse matrices
       List of Laplacian operators, one per dimension: laplacian of degree i is in the i-th position
    """
    laplacians = list()
    up = coo_matrix(boundaries[0] @ boundaries[0].T)
    laplacians.append(up)
    for d in range(len(boundaries)-1):
        down = boundaries[d].T @ boundaries[d]
        up = boundaries[d+1] @ boundaries[d+1].T
        laplacians.append(coo_matrix(down + up))
    down = boundaries[-1].T @ boundaries[-1]
    laplacians.append(coo_matrix(down))
    return laplacians

def normalize(L, half_interval = False):
    assert(sp.isspmatrix(L))
    M = L.shape[0]
    assert(M == L.shape[1])
    topeig = spl.eigsh(L, k=1, which="LM", return_eigenvectors = False)[0]   
    #print("Topeig = %f" %(topeig))

    ret = L.copy()
    if half_interval:
        ret *= 1.0/topeig
    else:
        ret *= 2.0/topeig
        ret.setdiag(ret.diagonal(0) - np.ones(M), 0)

    return ret

def assemble(K, L, x):
    (B, C_in, M) = x.shape
    assert(L.shape[0] == M)
    assert(L.shape[0] == L.shape[1])
    assert(K > 0)
    
    X = []
    for b in range(0, B):
        X123 = []
        for c_in in range(0, C_in):
            X23 = []
            X23.append(x[b, c_in, :].unsqueeze(1)) # Constant, k = 0 term.

            if K > 1:
                X23.append(L.mm(X23[0]))
            for k in range(2, K):
                X23.append(2*(L.mm(X23[k-1])) - X23[k-2])

            X23 = torch.cat(X23, 1)
            assert(X23.shape == (M, K))
            X123.append(X23.unsqueeze(0))

        X123 = torch.cat(X123, 0)
        assert(X123.shape == (C_in, M, K))
        X.append(X123.unsqueeze(0))

    X = torch.cat(X, 0)
    assert(X.shape == (B, C_in, M, K))

    return X

def coo2tensor(A):
    assert(sp.isspmatrix_coo(A))
    idxs = torch.LongTensor(np.vstack((A.row, A.col)))
    vals = torch.FloatTensor(A.data)
    return torch.sparse_coo_tensor(idxs, vals, size = A.shape, requires_grad = False)

class SimplicialConvolution(nn.Module):
    def __init__(self, K, C_in, C_out, enable_bias = True, variance = 1.0, groups = 1):
        assert groups == 1, "Only groups = 1 is currently supported."
        super().__init__()

        assert(C_in > 0)
        assert(C_out > 0)
        assert(K > 0)
        
        self.C_in = C_in
        self.C_out = C_out
        self.K = K
        self.enable_bias = enable_bias

        self.theta = nn.parameter.Parameter(variance*torch.randn((self.C_out, self.C_in, self.K)))
        if self.enable_bias:
            self.bias = nn.parameter.Parameter(torch.zeros((1, self.C_out, 1)))
        else:
            self.bias = 0.0
            
    def forward(self, L, x):
        assert(len(L.shape) == 2)
        assert(L.shape[0] == L.shape[1])
                
        (B, C_in, M) = x.shape
     
        assert(M == L.shape[0])
        assert(C_in == self.C_in)

        X = assemble(self.K, L, x)
        y = torch.einsum("bimk,oik->bom", (X, self.theta))
        assert(y.shape == (B, self.C_out, M))

        return y + self.bias

# This class does not yet implement the
# Laplacian-power-pre/post-composed with the coboundary. It can be
# simulated by just adding more layers anyway, so keeping it simple
# for now.
#
# Note: You can use this for a adjoints of coboundaries too. Just feed
# a transposed D.
class Coboundary(nn.Module):
    def __init__(self, C_in, C_out, enable_bias = True, variance = 1.0):
        super().__init__()

        assert(C_in > 0)
        assert(C_out > 0)

        self.C_in = C_in
        self.C_out = C_out
        self.enable_bias = enable_bias

        self.theta = nn.parameter.Parameter(variance*torch.randn((self.C_out, self.C_in)))
        if self.enable_bias:
            self.bias = nn.parameter.Parameter(torch.zeros((1, self.C_out, 1)))
        else:
            self.bias = 0.0

    def forward(self, D, x):
        assert(len(D.shape) == 2)
        
        (B, C_in, M) = x.shape
        
        assert(D.shape[1] == M)
        assert(C_in == self.C_in)
        
        N = D.shape[0]

        # This is essentially the equivalent of chebyshev.assemble for
        # the convolutional modules.
        X = []
        for b in range(0, B):
            X12 = []
            for c_in in range(0, self.C_in):
                X12.append(D.mm(x[b, c_in, :].unsqueeze(1)).transpose(0,1)) # D.mm(x[b, c_in, :]) has shape Nx1
            X12 = torch.cat(X12, 0)

            assert(X12.shape == (self.C_in, N))
            X.append(X12.unsqueeze(0))

        X = torch.cat(X, 0)
        assert(X.shape == (B, self.C_in, N))
                   
        y = torch.einsum("oi,bin->bon", (self.theta, X))
        assert(y.shape == (B, self.C_out, N))

        return y + self.bias