"""
Author: Nicholas Glaze, Rice ECE (nkg2 at rice.edu)

Code for converting ocean drifter data from Schaub's format to ours.
"""

import h5py
import os.path as osp

from data.datasets.scone_utils import *
from data.datasets.flow_utils import *
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


def load_madagascar_dataset():
    raw_dir = osp.join(ROOT_DIR, 'datasets', 'OCEAN', 'raw')
    raw_filename = osp.join(raw_dir, 'dataBuoys.jld2')

    f = h5py.File(raw_filename, 'r')
    print(f.keys())

    ### Load arrays from file

    ## Graph

    # elist (edge list)
    edge_list = f['elist'][:] - 1 # 1-index -> 0-index

    # tlist (triangle list)
    face_list = f['tlist'][:] - 1

    print("Faces", np.shape(face_list))

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
    filename = osp.join(raw_dir, 'madagascar_graph_faces_paths.pdf')
    color_faces(G, V, coords, faces_from_B2(B2, E), filename=filename, paths=[paths[100]])

    # train / test masks
    np.random.seed(1)
    train_mask = np.asarray([1] * round(len(paths) * 0.8) + [0] * round(len(paths) * 0.2))
    np.random.shuffle(train_mask)
    test_mask = 1 - train_mask

    flows = np.array([path_to_flow(p, edge_to_idx, len(E)) for p in paths])
    labels = np.array([extract_label(p, coords) for p in paths], dtype=np.int)
    print("Label 14", labels[100])

    # avg_clock = np.array([coords[i] for i, _ in enumerate(paths) if labels[i] == 0])
    # print("Average clockwise position", np.mean(avg_clock[:, 0]), np.mean(avg_clock[:, 1]))

    print('Flows', np.shape(flows))
    print('Train samples:', sum(train_mask))
    print('Test samples:', sum(test_mask))

    assert len(labels) == len(train_mask)

    print('Train Clockwise', sum(1 - labels[train_mask.astype(np.bool)]))
    print('Train Anticlockwise', sum(labels[train_mask.astype(np.bool)]))
    print('Test Clockwise', sum(1 - labels[test_mask.astype(np.bool)]))
    print('Test Anticlockwise', sum(labels[test_mask.astype(np.bool)]))

    lower_index, lower_orient = extract_adj_from_boundary(B1, G_undir)
    upper_index, upper_orient = extract_adj_from_boundary(B2.T, G_undir)
    index_dict = {
        'lower_index': lower_index,
        'lower_orient': lower_orient,
        'upper_index': upper_index,
        'upper_orient': upper_orient,
    }
    flows = torch.tensor(flows, dtype=torch.float32)

    train_chains, test_chains = [], []
    for i in range(len(flows)):
        chain = Chain(dim=1, x=flows[i], **index_dict, y=torch.tensor([labels[i]]))
        if train_mask[i] == 1:
            train_chains.append(chain)
        else:
            test_chains.append(chain)

    return train_chains, test_chains, G_undir

