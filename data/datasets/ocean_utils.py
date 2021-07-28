"""
Based on
- https://github.com/nglaze00/SCoNe_GCN/blob/master/ocean_drifters_data/buoy_data.py
- https://github.com/nglaze00/SCoNe_GCN/blob/master/trajectory_analysis/synthetic_data_gen.py

MIT License

Copyright (c) 2021 Nicholas Glaze
Copyright (c) 2021 The CWN Project Authors

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

import networkx as nx
import numpy as np
import h5py
import os.path as osp
import matplotlib.pyplot as plt
import data.datasets.flow_utils as fu

from definitions import ROOT_DIR
from tqdm import tqdm


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


def path_to_flow(path, edge_to_idx, m):
    '''Instantiates a 1-cochain from a path, accounting for the edge orientation.
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


def incidence_matrices(G, V, E, faces, edge_to_idx):
    """
    Returns incidence matrices B1 and B2

    :param G: NetworkX DiGraph
    :param V: list of nodes
    :param E: list of edges
    :param faces: list of faces in G

    Returns B1 (|V| x |E|) and B2 (|E| x |faces|)
    B1[i][j]: -1 if node i is tail of edge j, 1 if node i is head of edge j, else 0 (tail -> head) (smaller -> larger)
    B2[i][j]: 1 if edge i appears sorted in face j, -1 if edge i appears reversed in face j, else 0; given faces with sorted node order
    """
    B1 = np.array(nx.incidence_matrix(G, nodelist=V, edgelist=E, oriented=True).todense())
    B2 = np.zeros([len(E),len(faces)])

    for f_idx, face in enumerate(faces):  # face is sorted
        edges = [face[:-1], face[1:], [face[0], face[2]]]
        e_idxs = [edge_to_idx[tuple(e)] for e in edges]

        B2[e_idxs[:-1], f_idx] = 1
        B2[e_idxs[-1], f_idx] = -1
    return B1, B2


def strip_paths(paths):
    """
    Remove edges which are sequentially repeated in a path, e.g. [a, b, c, d, c, d, e, f] -> [a, b, c, d, e, f]
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
    plt.figure()
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

    plt.scatter(x=[np.mean(coords[:, 0]) - 0.03], y=[np.mean(coords[:, 1])])
    plt.savefig(filename)


def orientation(p1, p2, p3):
    # to find the orientation of
    # an ordered triplet (p1,p2,p3)
    # function returns the following values:
    # 0 : Colinear points
    # 1 : Clockwise points
    # 2 : Counterclockwise
    val = (float(p2[1] - p1[1]) * (p3[0] - p2[0])) - (float(p2[0] - p1[0]) * (p3[1] - p2[1]))
    if val > 0:
        # Clockwise orientation
        return 0
    elif val < 0:
        # Counterclockwise orientation
        return 1
    else:
        print(p1, p2, p3)
        raise ValueError('Points should not be collinear')


def extract_label(path, coords):
    # This is the center of madagascar. We will use it to compute the clockwise orientation
    # of the flow.
    center = [np.mean(coords[:, 0]) - 0.03, np.mean(coords[:, 1])]
    return orientation(center, coords[path[0]], coords[path[-1]])


def load_ocean_dataset(train_orient='default', test_orient='default'):
    raw_dir = osp.join(ROOT_DIR, 'datasets', 'OCEAN', 'raw')
    raw_filename = osp.join(raw_dir, 'dataBuoys.jld2')

    f = h5py.File(raw_filename, 'r')

    # elist (edge list)
    edge_list = f['elist'][:] - 1 # 1-index -> 0-index

    # tlist (triangle list)
    face_list = f['tlist'][:] - 1

    # print("Faces", np.shape(face_list))

    # NodeToHex (map node id <-> hex coords) # nodes are 1-indexed in data source
    node_hex_map = [tuple(f[x][()]) for x in f['NodeToHex'][:]]
    hex_node_map = {tuple(hex_coords): node for node, hex_coords in enumerate(node_hex_map)}

    # coords
    hex_coords = np.array([tuple(x) for x in f['HexcentersXY'][()]])

    # nodes
    traj_nodes = [[f[x][()] - 1 for x in f[ref][()]] for ref in f['TrajectoriesNodes'][:]]

    # generate graph + faces
    G = nx.Graph()
    G.add_edges_from([(edge_list[0][i], edge_list[1][i]) for i in range(len(edge_list[0]))])

    V, E = np.array(sorted(G.nodes)), np.array([sorted(x) for x in sorted(G.edges)])
    faces = np.array(sorted([[face_list[j][i] for j in range(3)] for i in range(len(face_list[0]))]))

    edge_to_idx = {tuple(e): i for i, e in enumerate(E)}
    coords = hex_coords
    valid_idxs = np.arange(len(coords))

    # B1, B2
    B1, B2 = incidence_matrices(G, V, E, faces, edge_to_idx)

    # Trajectories
    G_undir = G.to_undirected()
    stripped_paths = strip_paths(traj_nodes)
    paths = [path for path in stripped_paths if len(path) >= 5]
    new_paths = []
    for path in paths:
        new_path = path if path[-1] != path[0] else path[:-1]
        new_paths.append(new_path)
    paths = new_paths
    print("Max length path", max([len(path) for path in paths]))

    # Print graph info
    print(np.mean([len(G[i]) for i in V]))
    print('# nodes: {}, # edges: {}, # faces: {}'.format(*B1.shape, B2.shape[1]))
    print('# paths: {}, # paths with prefix length >= 3: {}'.format(len(traj_nodes), len(paths)))

    # Save graph image to file
    # filename = osp.join(raw_dir, 'madagascar_graph_faces_paths.pdf')
    # color_faces(G, V, coords, faces_from_B2(B2, E), filename=filename, paths=[paths[100]])

    # train / test masks
    np.random.seed(1)
    train_mask = np.asarray([1] * round(len(paths) * 0.8) + [0] * round(len(paths) * 0.2))
    np.random.shuffle(train_mask)
    test_mask = 1 - train_mask

    flows = np.array([path_to_flow(p, edge_to_idx, len(E)) for p in paths])
    labels = np.array([extract_label(p, coords) for p in paths], dtype=int)
    print("Label 14", labels[100])

    # avg_clock = np.array([coords[i] for i, _ in enumerate(paths) if labels[i] == 0])
    # print("Average clockwise position", np.mean(avg_clock[:, 0]), np.mean(avg_clock[:, 1]))

    print('Flows', np.shape(flows))
    print('Train samples:', sum(train_mask))
    print('Test samples:', sum(test_mask))

    assert len(labels) == len(train_mask)

    print('Train Clockwise', sum(1 - labels[train_mask.astype(bool)]))
    print('Train Anticlockwise', sum(labels[train_mask.astype(bool)]))
    print('Test Clockwise', sum(1 - labels[test_mask.astype(bool)]))
    print('Test Anticlockwise', sum(labels[test_mask.astype(bool)]))

    assert B1.shape[1] == B2.shape[0]
    num_edges = B1.shape[1]

    train_cochains, test_cochains = [], []
    for i in tqdm(range(len(flows)), desc='Processing dataset'):
        if train_mask[i] == 1:
            T2 = fu.get_orient_matrix(num_edges, train_orient)
            cochain = fu.build_cochain(B1, B2, T2, flows[i], labels[i], G_undir)
            train_cochains.append(cochain)
        else:
            T2 = fu.get_orient_matrix(num_edges, test_orient)
            cochain = fu.build_cochain(B1, B2, T2, flows[i], labels[i], G_undir)
            test_cochains.append(cochain)

    return train_cochains, test_cochains, G_undir
