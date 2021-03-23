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


def generate_random_walks(G, points, valid_idxs, m=1000):
    """
    Generates m random walks over the valid nodes in G.

    trajectories will look like one of the following:
        BEGIN -> A0 -> B0 -> END (top left regions)
        BEGIN -> A1 -> B1 -> END (middle regions)
        BEGIN -> A2 -> B2 -> END (bottom right regions

    :param G: NetworkX digraph
    :param points: list of (x, y) points that make up the graph's nodes
    :param valid_idxs: list of valid node indexes in
    :param E: sorted list of edges in E
    :param edge_to_idx: map (edge tuple -> index
    :param m: # of walks to generate

    Returns:
        paths: List of walks (each walk is a list of nodes)
        flows: E x m matrix:
            index i,j is 1 if flow j contains edge e in the direction of increasing node #
            i,j is -1 if decreasing node #
            else 0
    """
    points_valid = points[valid_idxs]

    # "partition" node space
    # 0: middle
    # 1: upper
    # 2: lower
    BEGIN = valid_idxs[np.sum(points_valid, axis=1) < 1 / 4]
    END = valid_idxs[np.sum(points_valid, axis=1) > 7 / 4]

    A012 = valid_idxs[(np.sum(points_valid, axis=1) > 1 / 4) & (np.sum(points_valid, axis=1) < 1)]
    A0 = A012[(points[A012, 1] - points[A012, 0] < 1 / 2) & (points[A012, 1] - points[A012, 0] > -1 / 2)]
    A1 = A012[points[A012, 1] - points[A012, 0] > 1 / 2]
    A2 = A012[points[A012, 1] - points[A012, 0] < -1 / 2]

    B012 = valid_idxs[(np.sum(points_valid, axis=1) < 7 / 4) & (np.sum(points_valid, axis=1) > 1)]
    B0 = B012[(points[B012, 1] - points[B012, 0] < 1 / 2) & (points[B012, 1] - points[B012, 0] > -1 / 2)]
    B1_ = B012[points[B012, 1] - points[B012, 0] > 1 / 2]
    B2_ = B012[points[B012, 1] - points[B012, 0] < -1 / 2]

    paths = []
    G_undir = G.to_undirected()
    i = 0
    while len(paths) < m:
        v_begin = np.random.choice(BEGIN)
        if i % 3 == 0:
            v_1 = np.random.choice(A0)
            v_2 = np.random.choice(B0)
        elif i % 3 == 1:
            v_1 = np.random.choice(A1)
            v_2 = np.random.choice(B1_)
        else:
            v_1 = np.random.choice(A2)
            v_2 = np.random.choice(B2_)
        v_end = np.random.choice(END)

        path = nx.shortest_path(G_undir, v_begin, v_1)[:-1] + \
               nx.shortest_path(G_undir, v_1, v_2)[:-1] + \
               nx.shortest_path(G_undir, v_2, v_end)
        if len(path) == len(set(path)):
            paths.append(path)
            i += 1

    return G_undir, paths


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


def path_dataset(G_undir, E, edge_to_idx, paths, max_degree, include_2hop=True, truncate_paths=True):
    """
    Builds necessary matrices for 1-hop and 2-hop learning, from a list of paths
    """
    # 1-hop
    prefixes_1hop, suffixes, last_nodes = split_paths(paths, truncate_paths=truncate_paths,
                                                      suffix_size=(2 if include_2hop else 1))
    suffixes_1hop = [s[0] for s in suffixes]
    prefix_flows = np.array([path_to_flow(p, edge_to_idx, len(E)) for p in prefixes_1hop])

    targets = np.array(
        [neighborhood_to_onehot(neighborhood(G_undir, prefix[-1]), suffix, max_degree) for prefix, suffix in
         zip(prefixes_1hop, suffixes_1hop)])

    if not include_2hop:
        return prefix_flows, targets, last_nodes, suffixes_1hop, [], [], [], []

    # 2-hop
    prefixes_2hop = [np.concatenate([p, [s]]) for p, s in zip(prefixes_1hop, suffixes_1hop)]
    suffixes_2hop = [s[1] for s in suffixes]
    last_nodes_2hop = [s[0] for s in suffixes]
    prefix_flows_2hop = np.array([path_to_flow(p, edge_to_idx, len(E)) for p in prefixes_2hop])

    targets_2hop = np.array(
        [neighborhood_to_onehot(neighborhood(G_undir, prefix[-1]), suffix, max_degree) for prefix, suffix in
         zip(prefixes_2hop, suffixes_2hop)])

    return prefix_flows, targets, last_nodes, suffixes_1hop, prefix_flows_2hop, targets_2hop, last_nodes_2hop, suffixes_2hop


def generate_dataset(n, m, folder, holes=True):
    # generate graph
    G, V, E, faces, edge_to_idx, coords, valid_idxs = random_SC_graph(n, holes=holes)



    # B1, B2
    B1, B2 = incidence_matrices(G, V, E, faces, edge_to_idx)
    G_undir, paths = generate_random_walks(G, coords, valid_idxs, m=m)
    rev_paths = [path[::-1] for path in paths]

    # Save image of graph to file
    color_faces(G.to_undirected(), V, coords, faces, filename='synthetic_graph_faces_paths.pdf',
                paths=[paths[11], paths[7][:-1], paths[18][:-1]])


    # train / test masks
    train_mask = np.asarray([1] * int(len(paths) * 0.8) + [0] * int(len(paths) * 0.2))
    np.random.shuffle(train_mask)
    test_mask = 1 - train_mask


    max_degree = np.max([deg for n, deg in G_undir.degree()])
    print(max_degree)

    # forward
    prefix_flows_1hop, targets_1hop, last_nodes_1hop, suffixes_1hop, \
        prefix_flows_2hop, targets_2hop, last_nodes_2hop, suffixes_2hop = path_dataset(G_undir, E, edge_to_idx, paths, max_degree)

    # reversed
    rev_prefix_flows_1hop, rev_targets_1hop, rev_last_nodes_1hop, rev_suffixes_1hop, \
        rev_prefix_flows_2hop, rev_targets_2hop, rev_last_nodes_2hop, rev_suffixes_2hop = path_dataset(G_undir, E, edge_to_idx, rev_paths, max_degree)

    dataset_1hop = [prefix_flows_1hop, B1, B2, targets_1hop, train_mask, test_mask, G_undir, coords, last_nodes_1hop,
                    suffixes_1hop, rev_prefix_flows_1hop, rev_targets_1hop, rev_last_nodes_1hop, rev_suffixes_1hop]
    dataset_2hop = [prefix_flows_2hop, B1, B2, targets_2hop, train_mask, test_mask, G_undir, coords, last_nodes_2hop,
                    suffixes_2hop, rev_prefix_flows_2hop, rev_targets_2hop, rev_last_nodes_2hop, rev_suffixes_2hop]

    # save datasets
    folder_1hop = 'trajectory_data_1hop_' + folder
    folder_2hop = 'trajectory_data_2hop_' + folder
    try:
        os.mkdir(folder_1hop), os.mkdir(folder_2hop)
    except:
        pass

    filenames = ('flows_in', 'B1', 'B2', 'targets', 'train_mask', 'test_mask', 'G_undir', 'coords', 'last_nodes', 'target_nodes', 'rev_flows_in', 'rev_targets', 'rev_last_nodes', 'rev_target_nodes')
    for arr_1hop, arr_2hop, filename in zip(dataset_1hop, dataset_2hop, filenames):
        if filename == 'G_undir':
            nx.readwrite.gpickle.write_gpickle(G_undir, os.path.join(folder_1hop, filename + '.pkl'))
            nx.readwrite.gpickle.write_gpickle(G_undir, os.path.join(folder_2hop, filename + '.pkl'))
        else:
            np.save(os.path.join(folder_1hop, filename + '.npy'), arr_1hop)
            np.save(os.path.join(folder_2hop, filename + '.npy'), arr_2hop)


def load_dataset(folder):
    """
    Loads training data from trajectory_data folder
    """
    file_paths = [os.path.join(folder, ar + '.npy') for ar in ('flows_in', 'B1', 'B2', 'targets', 'train_mask',
                                                               'test_mask', 'G_undir', 'last_nodes', 'target_nodes')]
    G_undir = nx.readwrite.gpickle.read_gpickle(file_paths[6][:-4] + '.pkl')
    remap = {node: int(node) for node in G_undir.nodes}
    G_undir = nx.relabel_nodes(G_undir, remap)
    # print(B_matrices[0][10])

    try:
        prefixes = np.load(folder + '/prefixes.npy')
    except:
        prefixes = None

    return np.load(file_paths[0]), [np.load(p) for p in file_paths[1:3]], np.load(file_paths[3]), \
           np.load(file_paths[4]), np.load(file_paths[5]), G_undir, np.load(file_paths[7]), np.load(file_paths[8])


def to_rnn_format(folder, prefixes_file=None):
    """
    Converts dataset to the format used by this repo https://github.com/wuhao5688/RNN-TrajModel
    """
    # load paths + graph
    G_undir = nx.readwrite.gpickle.read_gpickle(folder + '/G_undir.pkl')
    remap = {node: int(node) for node in G_undir.nodes}
    G_undir = nx.relabel_nodes(G_undir, remap)
    E = list(G_undir.edges)

    flows, last_nodes, target_nodes, train_mask, test_mask = [np.load(folder + '/' + name + '.npy') for name in
                                                              ('flows_in', 'last_nodes', 'target_nodes', 'train_mask',
                                                               'test_mask')]

    if not prefixes_file:
        prefixes = [flow_to_path(flow, E, last_node) for flow, last_node in zip(flows, last_nodes)]
    else:
        prefixes = np.load(folder + '/' + prefixes_file, allow_pickle=True)
    paths = [prefix + [target] for prefix, target in zip(prefixes, target_nodes)]
    coords = [[0, 0]] * len(G_undir.nodes)

    # save nodes + edges
    # nodes
    content = ''
    for i, c in enumerate(coords):
        content += str(i) + '\t' + str(c[0]) + '\t' + str(c[1]) + '\n'

    f = open(folder + '/nodeOSM.txt', 'w')
    f.write(content)
    f.close()

    # edges (graph is directed, so add one for both directions
    content = ''
    edge_to_id = {}
    E_directed = list(sorted(E + [e[::-1] for e in E]))

    for i, (e0, e1) in enumerate(E_directed):
        content += str(i) + '\t' + str(e0) + '\t' + str(e1) + '\t2\t' + str(coords[e0][0]) + '\t' + str(coords[e0][1]) + '\t' + str(coords[e1][0]) + '\t' + str(coords[e1][1]) + '\n'
        edge_to_id[(e0, e1)] = i
    f = open(folder + '/edgeOSM.txt', 'w')
    f.write(content)
    f.close()

    def build_content(paths):
        content = ''
        for path in paths:
            traj = [edge_to_id[(path[i], path[i + 1])] for i in range(len(path) - 1)]
            for e_id in traj:
                content += str(e_id) + ','
            content += '\n'
        return content

    # trajectories
    train_paths = [paths[i] for i in range(len(paths)) if train_mask[i] == 1]
    train_paths_trans = [paths[i] for i in range(len(paths)) if train_mask[i] == 1 and i % 3 == 1]
    test_paths_standard = [paths[i] for i in range(len(paths)) if test_mask[i] == 1]
    test_paths_rev = [path[::-1] for path in test_paths_standard]
    test_paths_trans = [paths[i] for i in range(len(paths)) if test_mask[i] == 1 and i % 3 == 2]

    f = open(folder + '/trajs.txt', 'w')
    f.write(build_content(train_paths) + build_content(test_paths_standard))
    f.close()
    f = open(folder + '/trajs_rev.txt', 'w')
    f.write(build_content(train_paths) + build_content(test_paths_rev))
    f.close()
    f = open(folder + '/trajs_trans.txt', 'w')
    f.write(build_content(train_paths_trans) + build_content(test_paths_trans))
    f.close()


if __name__ == '__main__':
    folder_suffix = 'synthetic' # make this whatever you want
    generate_dataset(400, 1000, folder_suffix)
    # to_rnn_format('trajectory_data_1hop_' + folder_suffix, prefixes_file=None)
