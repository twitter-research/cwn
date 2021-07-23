import numpy as np
import random
import torch
import networkx as nx
import itertools

from scipy.spatial import Delaunay
from scipy import sparse
from data.complex import Cochain
from data.parallel import ProgressParallel
from joblib import delayed


def is_inside_rectangle(x, rect):
    return rect[0, 0] <= x[0] <= rect[1, 0] and rect[0, 1] <= x[1] <= rect[1, 1]


def sample_point_from_rect(points, rect):
    samples = []
    for i in range(len(points)):
        if is_inside_rectangle(points[i], rect):
            samples.append(i)

    return random.choice(samples)


def create_hole(points, triangles, hole):
    kept_triangles = []
    removed_vertices = set()

    # Find the points and triangles to remove
    for i in range(len(triangles)):
        simplex = triangles[i]
        assert len(simplex) == 3
        xs = points[simplex]

        remove_triangle = False
        for j in range(3):
            vertex = simplex[j]
            if is_inside_rectangle(xs[j], hole):
                remove_triangle = True
                removed_vertices.add(vertex)

        if not remove_triangle:
            kept_triangles.append(i)

    # Remove the triangles and points inside the holes
    triangles = triangles[np.array(kept_triangles)]

    # Remove the points that are not part of any triangles anymore.
    # This can happen in some very rare cases
    for i in range(len(points)):
        if np.sum(triangles == i) == 0:
            removed_vertices.add(i)

    points = np.delete(points, list(removed_vertices), axis=0)

    # Renumber the indices of the triangles' vertices
    for vertex in sorted(removed_vertices, reverse=True):
        triangles[triangles >= vertex] -= 1

    return points, triangles


def create_graph_from_triangulation(points, triangles):
    # Create a graph from from this containing only the non-removed triangles
    G = nx.Graph()
    edge_idx = 0
    edge_to_tuple = {}
    tuple_to_edge = {}

    for i in range(len(triangles)):
        vertices = triangles[i]
        for j in range(3):
            if vertices[j] not in G:
                G.add_node(vertices[j], point=points[vertices[j]])

            for v1, v2 in itertools.combinations(vertices, 2):
                if not G.has_edge(v1, v2):
                    G.add_edge(v1, v2, index=edge_idx)
                    edge_to_tuple[edge_idx] = (min(v1, v2), max(v1, v2))
                    tuple_to_edge[(min(v1, v2), max(v1, v2))] = edge_idx
                    edge_idx += 1
                assert G.has_edge(v2, v1)

    G.graph['edge_to_tuple'] = edge_to_tuple
    G.graph['tuple_to_edge'] = tuple_to_edge
    G.graph['points'] = points
    G.graph['triangles'] = triangles
    return G


def extract_boundary_matrices(G: nx.Graph):
    """Compute the boundary and co-boundary matrices for the edges of the complex. """
    edge_to_tuple = G.graph['edge_to_tuple']
    tuple_to_edge = G.graph['tuple_to_edge']
    triangles = G.graph['triangles']

    B1 = np.zeros((G.number_of_nodes(), G.number_of_edges()), dtype=float)
    for edge_id in range(G.number_of_edges()):
        nodes = edge_to_tuple[edge_id]
        min_node = min(nodes)
        max_node = max(nodes)
        B1[min_node, edge_id] = -1
        B1[max_node, edge_id] = 1

    assert np.all(np.sum(np.abs(B1), axis=-1) > 0)
    assert np.all(np.sum(np.abs(B1), axis=0) == 2)
    assert np.all(np.sum(B1, axis=0) == 0)

    def extract_edge_and_orientation(triangle, i):
        assert i <= 2
        n1 = triangle[i]
        if i < 2:
            n2 = triangle[i + 1]
        else:
            n2 = triangle[0]

        if n1 < n2:
            orientation = 1
        else:
            orientation = -1

        return tuple_to_edge[(min(n1, n2), max(n1, n2))], orientation

    B2 = np.zeros((G.number_of_edges(), len(triangles)), dtype=float)
    for i in range(len(triangles)):
        edge1, orientation1 = extract_edge_and_orientation(triangles[i], 0)
        edge2, orientation2 = extract_edge_and_orientation(triangles[i], 1)
        edge3, orientation3 = extract_edge_and_orientation(triangles[i], 2)
        assert edge1 != edge2 and edge1 != edge3 and edge2 != edge3

        B2[edge1, i] = orientation1
        B2[edge2, i] = orientation2
        B2[edge3, i] = orientation3

    assert np.all(np.sum(np.abs(B2), axis=0) == 3)
    assert np.all(np.sum(np.abs(B2), axis=-1) > 0)
    return B1, B2


def generate_trajectory(start_rect, end_rect, ckpt_rect, G: nx.Graph):
    points = G.graph['points']
    tuple_to_edge = G.graph['tuple_to_edge']

    start_vertex = sample_point_from_rect(points, start_rect)
    end_vertex = sample_point_from_rect(points, end_rect)
    ckpt_vertex = sample_point_from_rect(points, ckpt_rect)

    x = np.zeros((len(tuple_to_edge), 1))

    vertex = start_vertex
    end_point = points[end_vertex]
    ckpt_point = points[ckpt_vertex]

    path = [vertex]
    explored = set()

    ckpt_reached = False

    while vertex != end_vertex:
        explored.add(vertex)
        if vertex == ckpt_vertex:
            ckpt_reached = True

        nv = np.array([nghb for nghb in G.neighbors(vertex)
                       if nghb not in explored])
        if len(nv) == 0:
            # If we get stuck because everything around was explored
            # Then just try to generate another trajectory.
            return generate_trajectory(start_rect, end_rect, ckpt_rect, G)
        npoints = points[nv]

        if ckpt_reached:
            dist = np.sum((npoints - end_point[None, :]) ** 2, axis=-1)
        else:
            dist = np.sum((npoints - ckpt_point[None, :]) ** 2, axis=-1)

        # prob = softmax(-dist**2)
        # vertex = nv[np.random.choice(len(prob), p=prob)]
        coin_toss = np.random.uniform()

        if coin_toss < 0.1:
            vertex = nv[np.random.choice(len(dist))]
        else:
            vertex = nv[np.argmin(dist)]

        path.append(vertex)

        # Set the flow value according to the orientation
        if path[-2] < path[-1]:
            x[tuple_to_edge[(path[-2], path[-1])], 0] = 1
        else:
            x[tuple_to_edge[(path[-1], path[-2])], 0] = -1

    return x, path


def extract_adj_from_boundary(B, G=None):
    A = sparse.csr_matrix(B.T).dot(sparse.csr_matrix(B))

    n = A.shape[0]
    if G is not None:
        assert n == G.number_of_edges()

    # Subtract self-loops, which we do not count.
    connections = A.count_nonzero() - np.sum(A.diagonal() != 0)

    index = torch.empty((2, connections), dtype=torch.long)
    orient = torch.empty(connections)

    connection = 0
    cA = A.tocoo()
    for i, j, v in zip(cA.row, cA.col, cA.data):
        if j >= i:
            continue
        assert v == 1 or v == -1, print(v)

        index[0, connection] = i
        index[1, connection] = j
        orient[connection] = np.sign(v)

        index[0, connection + 1] = j
        index[1, connection + 1] = i
        orient[connection + 1] = np.sign(v)

        connection += 2

    assert connection == connections
    return index, orient


def build_cochain(B1, B2, T2, x, class_id, G=None):
    # Change the orientation of the boundary matrices
    B1 = sparse.csr_matrix(B1).dot(sparse.csr_matrix(T2)).toarray()
    B2 = sparse.csr_matrix(T2).dot(sparse.csr_matrix(B2)).toarray()

    # Extract the adjacencies in pyG edge_index format.
    lower_index, lower_orient = extract_adj_from_boundary(B1, G)
    upper_index, upper_orient = extract_adj_from_boundary(B2.T, G)
    index_dict = {
        'lower_index': lower_index,
        'lower_orient': lower_orient,
        'upper_index': upper_index,
        'upper_orient': upper_orient,
    }

    # Change the orientation of the features
    x = sparse.csr_matrix(T2).dot(sparse.csr_matrix(x)).toarray()
    x = torch.tensor(x, dtype=torch.float32)

    return Cochain(dim=1, x=x, **index_dict, y=torch.tensor([class_id]))


def generate_flow_cochain(class_id, G, B1, B2, T2):
    assert 0 <= class_id <= 1

    # Define the start, midpoint and and stop regions for the trajectories.
    start_rect = np.array([[0.0, 0.8], [0.2, 1.0]])
    end_rect = np.array([[0.8, 0.0], [1.0, 0.2]])
    bot_ckpt_rect = np.array([[0.0, 0.0], [0.2, 0.2]])
    top_ckpt_rect = np.array([[0.8, 0.8], [1.0, 1.0]])
    ckpts = [bot_ckpt_rect, top_ckpt_rect]

    # Generate flow
    x, _ = generate_trajectory(start_rect, end_rect, ckpts[class_id], G)

    return build_cochain(B1, B2, T2, x, class_id, G)


def get_orient_matrix(size, orientation):
    """Creates a change of orientation operator of the specified size."""
    if orientation == 'default':
        return np.identity(size)
    elif orientation == 'random':
        diag = 2*np.random.randint(0, 2, size=size) - 1
        return np.diag(diag).astype(np.long)
    else:
        raise ValueError(f'Unsupported orientation {orientation}')


def load_flow_dataset(num_points=1000, num_train=1000, num_test=200,
                      train_orientation='default', test_orientation='default', n_jobs=2):
    points = np.random.uniform(low=-0.05, high=1.05, size=(num_points, 2))
    tri = Delaunay(points)

    # Double check each point appears in some triangle.
    for i in range(len(points)):
        assert np.sum(tri.simplices == i) > 0

    hole1 = np.array([[0.2, 0.2], [0.4, 0.4]])
    hole2 = np.array([[0.6, 0.6], [0.8, 0.8]])

    points, triangles = create_hole(points, tri.simplices, hole1)

    # Double check each point appears in some triangle.
    for i in range(len(points)):
        assert np.sum(triangles == i) > 0

    points, triangles = create_hole(points, triangles, hole2)

    # Double check each point appears in some triangle.
    for i in range(len(points)):
        assert np.sum(triangles == i) > 0

    assert np.min(triangles) == 0
    assert np.max(triangles) == len(points) - 1

    G = create_graph_from_triangulation(points, triangles)
    assert G.number_of_nodes() == len(points)

    B1, B2 = extract_boundary_matrices(G)
    classes = 2

    assert B1.shape[1] == B2.shape[0]
    num_edges = B1.shape[1]

    # Process these in parallel because it's slow
    samples_per_class = num_train // classes
    parallel = ProgressParallel(n_jobs=n_jobs, use_tqdm=True, total=num_train)
    train_samples = parallel(delayed(generate_flow_cochain)(
        class_id=min(i // samples_per_class, 1), G=G, B1=B1, B2=B2,
        T2=get_orient_matrix(num_edges, train_orientation)) for i in range(num_train))

    samples_per_class = num_test // classes
    parallel = ProgressParallel(n_jobs=n_jobs, use_tqdm=True, total=num_test)
    test_samples = parallel(delayed(generate_flow_cochain)(
        class_id=min(i // samples_per_class, 1), G=G, B1=B1, B2=B2,
        T2=get_orient_matrix(num_edges, test_orientation)) for i in range(num_test))

    return train_samples, test_samples, G






