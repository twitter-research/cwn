"""
Author: Nicholas Glaze, Rice ECE (nkg2 at rice.edu)

Code for converting ocean drifter data from Schaub's format to ours.
"""

import h5py
import os.path as osp

from data.datasets.scone_utils import *
from definitions import ROOT_DIR

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


def generate_dataset():
    dataset_folder = 'buoy'
    raw_path = osp.join(ROOT_DIR, 'datasets', 'MADAG', 'raw', 'dataBuoys.jld2')

    f = h5py.File(raw_path, 'r')
    print(f.keys())

    ### Load arrays from file

    ## Graph

    # elist (edge list)
    edge_list = f['elist'][:] - 1 # 1-index -> 0-index

    # tlist (triangle list)
    face_list = f['tlist'][:] - 1

    # NodeToHex (map node id <-> hex coords) # nodes are 1-indexed in data source
    node_hex_map = [tuple(f[x][()]) for x in f['NodeToHex'][:]]
    hex_node_map = {tuple(hex_coords): node for node, hex_coords in enumerate(node_hex_map)}


    ## trajectories

    # coords
    hex_coords = np.array([tuple(x) for x in f['HexcentersXY'][()]])

    # nodes
    traj_nodes = [[f[x][()] - 1 for x in f[ref][()]] for ref in f['TrajectoriesNodes'][:]]

    #### Convert to SCoNe dataset

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
    paths = [path[-10:] for path in stripped_paths if len(path) >= 5]

    # Print graph info
    print(np.mean([len(G[i]) for i in V]))
    print('# nodes: {}, # edges: {}, # faces: {}'.format(*B1.shape, B2.shape[1]))
    print('# paths: {}, # paths with prefix length >= 3: {}'.format(len(traj_nodes), len(paths)))

    rev_paths = [path[::-1] for path in paths]

    # Save graph image to file
    color_faces(G, V, coords, faces_from_B2(B2, E), filename='madagascar_graph_faces_paths.pdf', paths=[paths[1], paths[48], paths[125]])

    # train / test masks
    np.random.seed(1)
    train_mask = np.asarray([1] * round(len(paths) * 0.8) + [0] * round(len(paths) * 0.2))
    np.random.shuffle(train_mask)
    test_mask = 1 - train_mask

    max_degree = np.max([deg for n, deg in G_undir.degree()])

    ## Consolidate dataset

    # forward
    prefix_flows_1hop, targets_1hop, last_nodes_1hop, suffixes_1hop, \
        prefix_flows_2hop, targets_2hop, last_nodes_2hop, suffixes_2hop = path_dataset(G_undir, E, edge_to_idx,
                                                                                    paths, max_degree, include_2hop=True,
                                                                                    truncate_paths=False)

    print('Train samples:', sum(train_mask))
    print('Test samples:', sum(test_mask))
