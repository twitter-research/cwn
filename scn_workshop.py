"""
Input: Simplicial complex of dimension d
Output: k-order Laplacians up to dimension d
"""


import numpy as np
from scipy import sparse
from scipy.sparse import coo_matrix
from random import shuffle

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