"""
Author: Nicholas Glaze, Rice ECE (nkg2 at rice.edu)

Synthetic dataset generation; to generate a dataset, edit the function call in __main__ and run this file. The dataset
    will be saved into two folders: trajectory_data_1hop_ + your_folder_suffix, and trajectory_data_2hop_ + suffix.
    -Generating a dataset also generates a pdf with a cool pic of your graph!

If you want to use your own data, it'd be helpful to read this, and generate a synthetic one to better understand the
    format.

Description of dataset; your dataset should have all of these files:
trajectory_data_1hop/
    -B1.npy: B1 incidence matrix (nodes-edges); generate with incidence_matrices()
    -B2.npy: B2 incidence matrix (edges-faces); generate with incidence_matrices()
    -flows_in.npy: array of flows, each with dimension (n_edges) representing each path; 1 if this edge is traversed
        "forward" (lower # node -> higher # node), -1 if traversed in "reverse", 0 if not traversed
        -convert path (list of nodes) to flow with path_to_flow()
    -G_undir.pkl: undirected networkx graph representing the dataset's graphs
    -last_nodes.npy: the last node in each trajectory prefix; we forecast the step from this node to one of its neighbors
    -rev_flows_in.npy: same as flows_in, but reversed path direction -- (1,2,3) becomes (3,2,1)
    -rev_last_nodes.npy: same as last_nodes, but for reversed paths
    -rev_target_nodes.npy: same, for reversed paths
    -rev_targets.npy: same as targets, for reversed paths
    -target_nodes.npy: the correct suffix node for each trajectory; we're trying to predict this one
    -targets: for each path, a vector of dimension (max_degree) representing which neighbor is the correct suffix
        -neighbors are ordered by increasing node number
    -test_mask.npy: vector of length n_trajectories; 1 if this trajectory is in the test set, else 0
    -train_mask.npy: same, for training set
trajectory_data_2hop/
    -Pretty much the same, but for predicting the second "hop" after the known prefix. My code doesn't actually do any
    multi-hop predictions atm, so you can just copy-paste your 1hop data to the 2-hop folder, and it should work fine.

"""


import numpy as np
import networkx as nx
import os
import matplotlib.pyplot as plt

from scipy.spatial import Delaunay


def strip_paths(paths):
    """
    Remove repeated edges
    """
    res_all = []
    for path in paths:
        res = []
        for node in path:
            if len(res) < 2:
                res.append(node)
                continue

            if node == res[-2]:
                res.pop()
                continue
            else:
                res.append(node)
        res_all.append(res)
    return res_all


def color_faces(G, V, coords, faces, filename='graph_faces.pdf', paths=None):
    """
    Saves a plot of the graph, with faces colored in
    """
    for f in np.array(faces):
        plt.gca().add_patch(plt.Polygon(coords[f], facecolor=(173/256,216/256,240/256, 0.4), ec='k', linewidth=0.3))

    nx.draw_networkx(G, with_labels=False,
                      width=0.3,
                      node_size=0, pos=dict(zip(V, coords)))

    if paths:
        coords_dict = {i: xy for i, xy in enumerate(coords)}
        for path in paths:
            edges = [(path[i], path[i+1]) for i in range(len(path) - 1)]
            nx.draw_networkx_edges(G.to_directed(), pos=coords_dict, edgelist=edges, edge_color='black', width=1.5,
                                   arrows=True, arrowsize=7, node_size=3)
    plt.savefig(filename)


def random_SC_graph(n, holes=True):
    """
    Randomly generates a graph of simplicial complexes, made up of n nodes.
    Graph has holes in top left and bottom right regions.

    :param n: # of nodes in graph

    Returns:
        NetworkX DiGraph object G
        Sorted list of nodes V
        Sorted list of edges E
        Map  (edge tuples -> indices in E) edge_to_idx
        List of faces
        List of valid node indexes (nodes not in either hole)

    """
    np.random.seed(1)
    coords = np.random.rand(n,2)

    # sort nodes to be ordered from bottom-left to top-right
    diagonal_coordinates = np.sum(coords, axis=1)  # y = -x + c, compute c
    diagonal_idxs = np.argsort(diagonal_coordinates)  # sort by c: origin comes first, upper-right comes last
    coords = coords[diagonal_idxs]  # apply sort to original coordinates

    np.random.seed(1030)
    tri = Delaunay(coords)

    valid_idxs = np.where((np.linalg.norm(coords - [1/4, 3/4], axis=1) > 1/8) \
                          & (np.linalg.norm(coords - [3/4, 1/4], axis=1) > 1/8))[0]

    if not holes:
        valid_idxs = np.array(range(len(coords)))
    faces = np.array(sorted([sorted(t) for t in tri.simplices if np.in1d(t, valid_idxs).all()]))

    # SC matrix construction
    G = nx.OrderedDiGraph()
    G.add_nodes_from(np.arange(n)) # add nodes that are excluded to keep indexing easy
    E = []
    for f in faces:
        [a,b,c] = sorted(f)
        E.append((a,b))
        E.append((b,c))
        E.append((a,c))

    V = np.array(G.nodes)
    E = np.array(sorted(set(E)))

    for e in E:
        G.add_edge(*e)


    edge_to_idx = {tuple(E[i]): i for i in range(len(E))}
    print('Average degree:', np.average([G.degree[node] for node in range(n)]))
    print('Nodes:', len(V), 'Edges:', len(E))

    return G, V, E, faces, edge_to_idx, coords, valid_idxs


def incidence_matrices(G, V, E, faces, edge_to_idx):
    """
    Returns incidence matrices B1 and B2

    :param G: NetworkX DiGraph
    :param V: list of nodes
    :param E: list of edges
    :param faces: list of faces in G

    Returns B1 (|V| x |E|) and B2 (|E| x |faces|)
    B1[i][j]: -1 if node is is tail of edge j, 1 if node is head of edge j, else 0 (tail -> head) (smaller -> larger)
    B2[i][j]: 1 if edge i appears sorted in face j, -1 if edge i appears reversed in face j, else 0; given faces with sorted node order
    """
    B1 = np.array(nx.incidence_matrix(G, nodelist=V, edgelist=E, oriented=True).todense())
    B2 = np.zeros([len(E),len(faces)])

    for f_idx, face in enumerate(faces): # face is sorted
        edges = [face[:-1], face[1:], [face[0], face[2]]]
        e_idxs = [edge_to_idx[tuple(e)] for e in edges]

        B2[e_idxs[:-1], f_idx] = 1
        B2[e_idxs[-1], f_idx] = -1
    return B1, B2


def faces_from_B2(B2, E):
    """
    Given a B2 matrix, returns the list of faces.
    """
    faces_B2 = []
    for j in range(B2.shape[1]):
        edge_idxs = np.where(B2[:, j] != 0)
        edges = E[edge_idxs]
        nodes = set()
        for e in edges:
            for n in e:
                nodes.add(n)
        faces_B2.append(tuple(sorted(nodes)))
    return faces_B2


def split_paths(paths, truncate_paths=True, suffix_size=2):
    """
    Truncates paths (if indicated), then splits each into prefix + suffix
    """
    if truncate_paths:
        paths_truncated = [p[:4 + np.random.choice(range(2, len(p) - 4))] for p in paths]
    else:
        paths_truncated = paths

    prefixes = [p[:-suffix_size] for p in paths_truncated]
    suffixes = [p[-suffix_size:] for p in paths_truncated]
    last_nodes = [p[-1] for p in prefixes]

    return prefixes, suffixes, last_nodes


def conditional_incidence_matrix(B1, Nv, D):
    '''
    B1: numpy array
    Nv: row indices of B1 to extract
    D: max degree, for zero padding
    '''
    B_cond = np.zeros([D,B1.shape[1]])
    B_cond[:len(Nv),:] = B1[Nv]
    return B_cond


def generate_Bconds(G_undir, B1, last_nodes, max_degree):
    """
    Generates the conditional incidence matrix for each "last node" in a path, padded to the size of the max degree
    """
    B_conds = []
    for n in last_nodes:
        n_nbrs = np.array(sorted(G_undir[n]))
        B_cond = conditional_incidence_matrix(B1, n_nbrs, max_degree)
        B_conds.append(B_cond)
    return B_conds


def neighborhood(G, v):
    '''
    G: networkx undirected graph
    v: node label
    '''
    return np.array(sorted(G[v]))


def neighborhood_to_onehot(Nv, w, D):
    '''
    Nv: numpy array
    w: integer, presumably present in Nv
    D: max degree, for zero padding
    '''
    onehot = (Nv==w).astype(np.float)
    onehot_final = np.zeros(D)
    onehot_final[:onehot.shape[0]] = onehot
    return np.array([onehot_final]).T


def flow_to_path(flow, E, last_node):
    """
    Given a flow vector and the last node in the path, returns the path
    """
    # get edges in path
    path = [last_node]
    edges = set()
    for i in np.where(flow != 0)[0]:
        if flow[i] == 1:
            edges.add(E[i])
        elif flow[i] == -1:
            edges.add(E[i][::-1])
    # order edges
    cur_node = last_node
    while edges:
        next_edge = None
        for e in edges:
            if e[1] == cur_node:
                next_edge = e
        if next_edge is None:
            raise ValueError
        path.append(next_edge[0])
        edges.remove(next_edge)
        cur_node = next_edge[0]

    path[0] = int(path[0])
    return path[::-1]


def path_to_flow(path, edge_to_idx, m):
    '''
    path: list of nodes
    E_lookup: dictionary mapping edge tuples to indices
    m: number of edges
    '''
    l = len(path)
    f = np.zeros([m,1])
    for j in range(l-1):
        v0 = path[j]
        v1 = path[j+1]
        if v0 < v1:
            k = edge_to_idx[(v0,v1)]
            f[k] += 1
        else:
            k = edge_to_idx[(v1,v0)]
            f[k] -= 1
    return f
