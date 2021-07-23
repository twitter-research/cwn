import matplotlib.pyplot as plt
import itertools as it

import numpy as np
import torch
import gudhi as gd
import itertools
import graph_tool as gt
import graph_tool.topology as top
import networkx as nx

from tqdm import tqdm
from data.complex import Chain, Complex
from typing import List, Dict, Optional, Union
from torch import Tensor
from torch_geometric.typing import Adj
from torch_scatter import scatter
from data.parallel import ProgressParallel
from joblib import delayed


def get_nx_graph(ptg_graph):
    edge_list = ptg_graph.edge_index.numpy().T
    G = nx.Graph()
    G.add_nodes_from(list(range(ptg_graph.num_nodes)))
    G.add_edges_from(edge_list)
    return G


def color_mapper(val, ncolors, max_c=16):
    if val <= max_c:
        return ncolors[int((val / max_c) * (len(ncolors)-1))]
    else:
        return [0.0, 0.0, 0.0]


def draw_legend(colors):
    plt.figure(figsize=(2,2), dpi=100)
    plt.box(False)
    plt.title('Color legend')
    for k, kind in enumerate(sorted(colors)):
        plt.plot([], [], 'o', color=colors[kind], label=kind, markersize=5)
    plt.xticks([])
    plt.yticks([])
    plt.legend(loc='center')
    plt.show()
    plt.close()
    return


def get_boundaries(simplex):
    '''
        Given a k-simplex as an iterable of nodes, returns all its boundaries (contained (k-1)-simplices).
        boundaries are returned as a set of ordered tuples.
    '''
    k = len(simplex)
    if k == 1:
        raise ValueError("0-simplices do not have boundaries.")
    return set([tuple(sorted(comb)) for comb in it.combinations(simplex, k-1)])


def lower_adj(a, b, min_k=1, all_simplices=None):
    '''
        Returns True if simplices a, b are lower-adjacent.
        Simplices a, b are lower-adjacent when:
            - their size (k) is larger than 1;
            - they have the same dimension;
            - the share one (k-1)-boundary.
        Optionally, it is possible to also enforce the shared
        co-boundary to be present in a set of know simplices (`all_simplices`).
    '''
    assert len(a) == len(b)
    k = len(a)
    intersection = tuple(sorted(set(a) & set(b)))
    result = len(intersection) == k - 1 and k > min_k
    if all_simplices is not None:
        result = result and intersection in all_simplices
    return result


def get_simplex_upper_adjs(simplex, all_facets, all_simplices, non_facets, upper_adjacencies, upper_adjacencies_labeled):
    '''
        Recursive function to compute upper adjacencies between the boundaries of a given simplex.
        Parameters:
            - `simplex`: input simplex as an iterable of nodes
            - `all_facets`: a set of facets for the complex the simplex belongs to
            - `all_simplices`: set updated during the recursion to keep track of all simplices encountered
            - `non_facets`: set updated during the recursion to keep track of all simplices encountered which are not facets
            - `upper_adjacencies`: dictionary of upper adjacencies updated during recursion
    '''
    # keep track of the simplices encountered;
    # ... and those which do not correspond to any facet
    nodes = tuple(sorted(simplex))
    all_simplices.add(nodes)
    if nodes not in all_facets:
        non_facets.add(nodes)

    # stop recursion if nodes have been reached
    k = len(nodes)
    if k == 1:
        return all_simplices, non_facets, upper_adjacencies, upper_adjacencies_labeled

    # get boundaries of the input simplex: all of them are
    # considered to be upper adjacent w.r.t present simplex
    boundaries = get_boundaries(nodes)
    for boundary in boundaries:

        # add adjacencies to all other boundaries in the simplex
        if boundary not in upper_adjacencies[k-1]:
            upper_adjacencies[k-1][boundary] = set()
            upper_adjacencies_labeled[k-1][boundary] = dict()
        upper_adjacencies[k-1][boundary] |= boundaries - {boundary}
        for neighbor in boundaries - {boundary}:
            upper_adjacencies_labeled[k-1][boundary][neighbor] = nodes

        # recur down w.r.t. present face
        all_simplices, non_facets, upper_adjacencies, upper_adjacencies_labeled = get_simplex_upper_adjs(boundary, all_facets, all_simplices, non_facets, upper_adjacencies, upper_adjacencies_labeled)

    return all_simplices, non_facets, upper_adjacencies, upper_adjacencies_labeled


def compute_upper_adjs(all_facets, all_facets_by_size, max_size):
    '''
        Computes upper adjacencies for all simplices in a complex.
        The complex is specified in terms of its facets (top-level simplices).
        Here the idea is to iterate on the facets and, for each of them, to recursively compute the upper adjacencies between the constituent simplices.
        Parameters:
            - `all_facets`: the set of facets constituting the complex; each facet is an ordered tuple containing nodes
            - `all_facets_by_size`: dictionary of facets constituting the complex; the dictionary is indexed by the number of nodes in each facet
            - `max_size`: maximum number of nodes for facets to be considered here; e.g. if `max_size` equals 3, then facets are considered only up to triangles
    '''
    # initialize dictionary of adjacencies
    upper_adjacencies = dict()
    upper_adjacencies_labeled = dict()
    for k in range(1, max_size):
        upper_adjacencies[k] = dict()
        upper_adjacencies_labeled[k] = dict()
    all_simplices = set()
    non_facets = set()

    # iterate over sizes
    for k in range(1, max_size+1):
        facets = all_facets_by_size[k]

        # iterate over facets of size k
        for facet in facets:
            all_simplices, non_facets, upper_adjacencies, upper_adjacencies_labeled = get_simplex_upper_adjs(facet, all_facets, all_simplices, non_facets, upper_adjacencies, upper_adjacencies_labeled)

    return all_simplices, non_facets, upper_adjacencies, upper_adjacencies_labeled


def compute_lower_adjs(all_simplices, all_simplices_by_size, max_size):
    '''
        Computes lower adjacencies for all simplices in a complex.
        The complex is specified in terms of all contained simplices -- parameter `all_simplices`, where `all_simplices_by_size` is simply a dictionary of simplices constituting the complex indexed by the number of nodes in each of them.
        Here the idea is to iterate on the facets and, for each of them, to recursively compute the upper adjacencies between the constituent simplices.
        Lastly, `max_size` is maximum number of nodes for simplices to be considered here; e.g. if `max_size` equals 3, then simplices are considered only up to triangles.
    '''
    # initialize dictionary of adjacencies
    lower_adjacencies = dict()
    for k in range(2, max_size+1):
        lower_adjacencies[k] = dict()

    # iterate over sizes
    for k in range(1, max_size+1):
        simplices = all_simplices_by_size[k]

        # iterate over simplices of size k: for each (ordered) couple
        # test wether they are lower adjacent
        for p in range(len(simplices)):
            for pp in range(p+1, len(simplices)):
                nodes_p = simplices[p]
                nodes_pp = simplices[pp]
                if lower_adj(nodes_p, nodes_pp):
                    if nodes_p not in lower_adjacencies[k]:
                        lower_adjacencies[k][nodes_p] = set()
                    if nodes_pp not in lower_adjacencies[k]:
                        lower_adjacencies[k][nodes_pp] = set()
                    # add both directions
                    lower_adjacencies[k][nodes_p].add(nodes_pp)
                    lower_adjacencies[k][nodes_pp].add(nodes_p)

    return lower_adjacencies


def compute_connectivity(all_facets, all_facets_by_size, max_size):
    '''
        Computes the lower and upper connectivities in a complex.
        The complex is specified in terms of its facets (top-level simplices).
        Parameters:
            - `all_facets`: the set of facets constituting the complex; each facet is an ordered tuple containing nodes
            - `all_facets_by_size`: dictionary of facets constituting the complex; the dictionary is indexed by the number of nodes in each facet
            - `max_size`: maximum number of nodes for facets to be considered here; e.g. if `max_size` equals 3, then facets are considered only up to triangles
    '''
    # 1. compute upper adjacencies starting from the facets
    all_simplices, non_facets, upper_adjacencies, upper_adjacencies_labeled = compute_upper_adjs(all_facets, all_facets_by_size, max_size)

    # 2. group all simplices by their size
    all_simplices_by_size = dict()
    for simplex in all_simplices:
        k = len(simplex)
        if k not in all_simplices_by_size:
            all_simplices_by_size[k] = list()
        all_simplices_by_size[k].append(simplex)

    # 3. compute lower_adjacencies
    lower_adjacencies = compute_lower_adjs(all_simplices, all_simplices_by_size, max_size)

    return upper_adjacencies, lower_adjacencies, all_simplices, all_simplices_by_size, upper_adjacencies_labeled


def get_adj_index(simplices, connectivity, size):
    '''
        Transforms a connectivity dictionary into an PyG-like edge index (torch.LongTensor).
        Additionally, it builds dictionaries to map original node tuples to indices and vice-versa.
    '''
    index = {simplex: s for s, simplex in enumerate(simplices)}
    rev_index = {s: simplex for s, simplex in enumerate(simplices)}
    seen = set()
    edges = list()
    for simplex in simplices:
        s = index[simplex]
        neighbors = connectivity.get(simplex, [])
        for neighbor in neighbors:
            edge = tuple(sorted((s, index[neighbor])))
            if not edge in seen:
                edges.append([s, index[neighbor]])
                edges.append([index[neighbor], s])
                seen.add(edge)
    mapping = torch.LongTensor(len(simplices), size)
    for s in rev_index:
        mapping[s] = torch.LongTensor(list(rev_index[s]))
    if len(edges) == 0:
        edges = torch.LongTensor([[],[]])
    else:
        edges = torch.LongTensor(edges).transpose(1,0)
    return edges, mapping


def generate_complex(attributes, labels, upper_indices, lower_indices, mappings, upper_adjs, min_order, max_order):

    # generate mappings nodes -> simplex index
    rev_mappings = dict()
    for order in range(min_order, max_order+1):
        current_rev_map = dict()
        current_map = mappings[order]
        for key in range(current_map.shape[0]):
            current_rev_map[tuple(current_map[key].numpy())] = key
        rev_mappings[order] = current_rev_map

    shared_boundaries = dict()
    shared_coboundaries = dict()
    shared_boundaries[min_order] = None
    shared_coboundaries[max_order] = None
    for order in range(min_order, max_order):

        shared = list()
        lower = lower_indices[order+1].numpy().T
        for link in lower:
            a, b = link
            nodes_a = set(mappings[order+1][a].numpy().tolist())
            nodes_b = set(mappings[order+1][b].numpy().tolist())
            shared_boundary = rev_mappings[order][tuple(sorted(nodes_a & nodes_b))]
            shared.append(shared_boundary)
        shared_boundaries[order+1] = torch.LongTensor(shared)

        shared = list()
        upper = upper_indices[order].numpy().T
        for link in upper:
            a, b = link
            nodes_a = tuple(mappings[order][a].numpy().tolist())
            nodes_b = tuple(mappings[order][b].numpy().tolist())
            shared_coboundary = rev_mappings[order+1][upper_adjs[order+1][nodes_a][nodes_b]]
            shared.append(shared_coboundary)
        shared_coboundaries[order] = torch.LongTensor(shared)

    chains = list()
    for k in range(min_order, max_order+1):
        try:
            y = labels[k]
        except TypeError:
            y = None
        chains.append(Chain(k, x=attributes[k], y=y, upper_index=upper_indices[k], lower_index=lower_indices[k], mapping=mappings[k], shared_boundaries=shared_boundaries[k], shared_coboundaries=shared_coboundaries[k]))

    try:
        _ = labels.keys()
        y = labels
    except AttributeError:
        y = None
    return Complex(*chains, y=y)


def pyg_to_simplex_tree(edge_index: Tensor, size: int):
    """Constructs a simplex tree from a PyG graph.

    Args:
        edge_index: The edge_index of the graph (a tensor of shape [2, num_edges])
        size: The number of nodes in the graph.
    """
    st = gd.SimplexTree()
    # Add vertices to the simplex.
    for v in range(size):
        st.insert([v])

    # Add the edges to the simplex.
    edges = edge_index.numpy()
    for e in range(edges.shape[1]):
        edge = [edges[0][e], edges[1][e]]
        st.insert(edge)

    return st


def get_simplex_boundaries(simplex):
    boundaries = itertools.combinations(simplex, len(simplex) - 1)
    return [tuple(boundary) for boundary in boundaries]


def build_tables(simplex_tree, size):
    complex_dim = simplex_tree.dimension()
    # Each of these data structures has a separate entry per dimension.
    id_maps = [{} for _ in range(complex_dim+1)] # simplex -> id
    simplex_tables = [[] for _ in range(complex_dim+1)] # matrix of simplices
    boundaries_tables = [[] for _ in range(complex_dim+1)]

    simplex_tables[0] = [[v] for v in range(size)]
    id_maps[0] = {tuple([v]): v for v in range(size)}

    for simplex, _ in simplex_tree.get_simplices():
        dim = len(simplex) - 1
        if dim == 0:
            continue

        # Assign this simplex the next unused ID
        next_id = len(simplex_tables[dim])
        id_maps[dim][tuple(simplex)] = next_id
        simplex_tables[dim].append(simplex)

    return simplex_tables, id_maps


def extract_boundaries_and_coboundaries_from_simplex_tree(simplex_tree, id_maps, complex_dim: int):
    """Build two maps simplex -> its coboundaries and simplex -> its boundaries"""
    # The extra dimension is added just for convenience to avoid treating it as a special case.
    boundaries = [{} for _ in range(complex_dim+2)]  # simplex -> boundaries
    coboundaries = [{} for _ in range(complex_dim+2)]  # simplex -> coboundaries
    boundaries_tables = [[] for _ in range(complex_dim+1)]

    for simplex, _ in simplex_tree.get_simplices():
        # Extract the relevant boundary and coboundary maps
        simplex_dim = len(simplex) - 1
        level_coboundaries = coboundaries[simplex_dim]
        level_boundaries = boundaries[simplex_dim + 1]

        # Add the boundaries of the simplex to the boundaries table
        if simplex_dim > 0:
            boundaries_ids = [id_maps[simplex_dim-1][boundary] for boundary in get_simplex_boundaries(simplex)]
            boundaries_tables[simplex_dim].append(boundaries_ids)

        # This operation should be roughly be O(dim_complex), so that is very efficient for us.
        # For details see pages 6-7 https://hal.inria.fr/hal-00707901v1/document
        simplex_coboundaries = simplex_tree.get_cofaces(simplex, codimension=1)
        for coboundary, _ in simplex_coboundaries:
            assert len(coboundary) == len(simplex) + 1

            if tuple(simplex) not in level_coboundaries:
                level_coboundaries[tuple(simplex)] = list()
            level_coboundaries[tuple(simplex)].append(tuple(coboundary))

            if tuple(coboundary) not in level_boundaries:
                level_boundaries[tuple(coboundary)] = list()
            level_boundaries[tuple(coboundary)].append(tuple(simplex))

    return boundaries_tables, boundaries, coboundaries


def build_adj(boundaries: List[Dict], coboundaries: List[Dict], id_maps: List[Dict], complex_dim: int,
              include_down_adj: bool):
    """Builds the upper and lower adjacency data structures of the complex

    Args:
        boundaries: A list of dictionaries of the form
            boundaries[dim][simplex] -> List[simplex] (the boundaries)
        coboundaries: A list of dictionaries of the form
            coboundaries[dim][simplex] -> List[simplex] (the coboundaries)
        id_maps: A dictionary from simplex -> simplex_id
    """
    def initialise_structure():
        return [[] for _ in range(complex_dim+1)]

    upper_indexes, lower_indexes = initialise_structure(), initialise_structure()
    all_shared_boundaries, all_shared_coboundaries = initialise_structure(), initialise_structure()

    # Go through all dimensions of the complex
    for dim in range(complex_dim+1):
        # Go through all the simplices at that dimension
        for simplex, id in id_maps[dim].items():
            # Add the upper adjacent neighbours from the level below
            if dim > 0:
                for boundary1, boundary2 in itertools.combinations(boundaries[dim][simplex], 2):
                    id1, id2 = id_maps[dim - 1][boundary1], id_maps[dim - 1][boundary2]
                    upper_indexes[dim - 1].extend([[id1, id2], [id2, id1]])
                    all_shared_coboundaries[dim - 1].extend([id, id])

            # Add the lower adjacent neighbours from the level above
            if include_down_adj and dim < complex_dim and simplex in coboundaries[dim]:
                for coboundary1, coboundary2 in itertools.combinations(coboundaries[dim][simplex], 2):
                    id1, id2 = id_maps[dim + 1][coboundary1], id_maps[dim + 1][coboundary2]
                    lower_indexes[dim + 1].extend([[id1, id2], [id2, id1]])
                    all_shared_boundaries[dim + 1].extend([id, id])

    return all_shared_boundaries, all_shared_coboundaries, lower_indexes, upper_indexes


def construct_features(vx: Tensor, cell_tables, init_method: str) -> List:
    """Combines the features of the component vertices to initialise the cell features"""
    features = [vx]
    for dim in range(1, len(cell_tables)):
        aux_1 = []
        aux_0 = []
        for c, cell in enumerate(cell_tables[dim]):
            aux_1 += [c for _ in range(len(cell))]
            aux_0 += cell
        node_cell_index = torch.LongTensor([aux_0, aux_1])
        in_features = vx.index_select(0, node_cell_index[0])
        features.append(scatter(in_features, node_cell_index[1], dim=0,
                                dim_size=len(cell_tables[dim]), reduce=init_method))

    return features


def extract_labels(y, size):
    v_y, complex_y = None, None
    if y is None:
        return v_y, complex_y

    y_shape = list(y.size())

    if y_shape[0] == 1:
        # This is a label for the whole graph (for graph classification).
        # We will use it for the complex.
        complex_y = y
    else:
        # This is a label for the vertices of the complex.
        assert y_shape[0] == size
        v_y = y

    return v_y, complex_y


def generate_chain(dim, x, all_upper_index, all_lower_index,
                   all_shared_boundaries, all_shared_coboundaries, simplex_tables, boundaries_tables,
                   complex_dim, y=None):
    """Builds a Chain given all the adjacency data extracted from the complex."""
    if dim == 0:
        assert len(all_lower_index[dim]) == 0
        assert len(all_shared_boundaries[dim]) == 0

    num_simplices_down = len(simplex_tables[dim-1]) if dim > 0 else None
    num_simplices_up = len(simplex_tables[dim+1]) if dim < complex_dim else 0

    up_index = (torch.tensor(all_upper_index[dim], dtype=torch.long).t()
                if len(all_upper_index[dim]) > 0 else None)
    down_index = (torch.tensor(all_lower_index[dim], dtype=torch.long).t()
                  if len(all_lower_index[dim]) > 0 else None)
    shared_coboundaries = (torch.tensor(all_shared_coboundaries[dim], dtype=torch.long)
                      if len(all_shared_coboundaries[dim]) > 0 else None)
    shared_boundaries = (torch.tensor(all_shared_boundaries[dim], dtype=torch.long)
                    if len(all_shared_boundaries[dim]) > 0 else None)
    
    boundary_index = None
    if len(boundaries_tables[dim]) > 0:
        boundary_index = [list(), list()]
        for s, simplex in enumerate(boundaries_tables[dim]):
            for boundary in simplex:
                boundary_index[1].append(s)
                boundary_index[0].append(boundary)
        boundary_index = torch.LongTensor(boundary_index)
        
    if num_simplices_down is None:
        assert shared_boundaries is None
    if num_simplices_up == 0:
        assert shared_coboundaries is None

    if up_index is not None:
        assert up_index.size(1) == shared_coboundaries.size(0)
        assert num_simplices_up == shared_coboundaries.max() + 1
    if down_index is not None:
        assert down_index.size(1) == shared_boundaries.size(0)
        assert num_simplices_down >= shared_boundaries.max() + 1

    return Chain(dim=dim, x=x, upper_index=up_index,
                 lower_index=down_index, shared_coboundaries=shared_coboundaries,
                 shared_boundaries=shared_boundaries, y=y, num_simplices_down=num_simplices_down,
                 num_simplices_up=num_simplices_up, boundary_index=boundary_index)


def compute_clique_complex_with_gudhi(x: Tensor, edge_index: Adj, size: int,
                                      expansion_dim: int = 2, y: Tensor = None,
                                      include_down_adj=True,
                                      init_method: str = 'sum') -> Complex:
    """Generates a clique complex of a pyG graph via gudhi.

    Args:
        x: The feature matrix for the nodes of the graph
        edge_index: The edge_index of the graph (a tensor of shape [2, num_edges])
        size: The number of nodes in the graph
        expansion_dim: The dimension to expand the simplex to.
        y: Labels for the graph nodes or a label for the whole graph.
        include_down_adj: Whether to add down adj in the complex or not
        init_method: How to initialise features at higher levels.
    """
    assert x is not None
    assert isinstance(edge_index, Tensor)  # Support only tensor edge_index for now

    # Creates the gudhi-based simplicial complex
    simplex_tree = pyg_to_simplex_tree(edge_index, size)
    simplex_tree.expansion(expansion_dim)  # Computes the clique complex up to the desired dim.
    complex_dim = simplex_tree.dimension()  # See what is the dimension of the complex now.

    # Builds tables of the simplicial complexes at each level and their IDs
    simplex_tables, id_maps = build_tables(simplex_tree, size)

    # Extracts the boundaries and coboundaries of each simplex in the complex
    boundaries_tables, boundaries, co_boundaries = (
        extract_boundaries_and_coboundaries_from_simplex_tree(simplex_tree, id_maps, complex_dim))

    # Computes the adjacencies between all the simplexes in the complex
    shared_boundaries, shared_coboundaries, lower_idx, upper_idx = build_adj(boundaries, co_boundaries, id_maps,
                                                                   complex_dim, include_down_adj)

    # Construct features for the higher dimensions
    # TODO: Make this handle edge features as well and add alternative options to compute this.
    xs = construct_features(x, simplex_tables, init_method)

    # Initialise the node / complex labels
    v_y, complex_y = extract_labels(y, size)

    chains = []
    for i in range(complex_dim+1):
        y = v_y if i == 0 else None
        chain = generate_chain(i, xs[i], upper_idx, lower_idx, shared_boundaries, shared_coboundaries,
                               simplex_tables, boundaries_tables, complex_dim=complex_dim, y=y)
        chains.append(chain)

    return Complex(*chains, y=complex_y, dimension=complex_dim)


def convert_graph_dataset_with_gudhi(dataset, expansion_dim: int, include_down_adj=True,
                                     init_method: str = 'sum'):
    # TODO(Cris): Add parallelism to this code like in the cell complex conversion code.
    dimension = -1
    complexes = []
    num_features = [None for _ in range(expansion_dim+1)]

    for data in tqdm(dataset):
        complex = compute_clique_complex_with_gudhi(data.x, data.edge_index, data.num_nodes,
            expansion_dim=expansion_dim, y=data.y, include_down_adj=include_down_adj,
            init_method=init_method)
        if complex.dimension > dimension:
            dimension = complex.dimension
        for dim in range(complex.dimension + 1):
            if num_features[dim] is None:
                num_features[dim] = complex.chains[dim].num_features
            else:
                assert num_features[dim] == complex.chains[dim].num_features
        complexes.append(complex)

    return complexes, dimension, num_features[:dimension+1]


# ---- support for rings as cells

def get_rings(edge_index, max_k=7):
    if isinstance(edge_index, torch.Tensor):
        edge_index = edge_index.numpy()

    edge_list = edge_index.T
    graph_gt = gt.Graph(directed=False)
    graph_gt.add_edge_list(edge_list)
    gt.stats.remove_self_loops(graph_gt)
    gt.stats.remove_parallel_edges(graph_gt)
    # We represent rings with their original node ordering
    # so that we can easily read out the boundaries
    # The use of the `sorted_rings` set allows to discard
    # different isomorphisms which are however associated
    # to the same original ring – this happens due to the intrinsic
    # symmetries of cycles
    rings = set()
    sorted_rings = set()
    for k in range(3, max_k+1):
        pattern = nx.cycle_graph(k)
        pattern_edge_list = list(pattern.edges)
        pattern_gt = gt.Graph(directed=False)
        pattern_gt.add_edge_list(pattern_edge_list)
        sub_isos = top.subgraph_isomorphism(pattern_gt, graph_gt, induced=True, subgraph=True,
                                           generator=True)
        sub_iso_sets = map(lambda isomorphism: tuple(isomorphism.a), sub_isos)
        for iso in sub_iso_sets:
            if tuple(sorted(iso)) not in sorted_rings:
                rings.add(iso)
                sorted_rings.add(tuple(sorted(iso)))
    rings = list(rings)
    return rings


def build_tables_with_rings(edge_index, simplex_tree, size, max_k):
    
    # Build simplex tables and id_maps up to edges by conveniently
    # invoking the code for simplicial complexes
    simplex_tables, id_maps = build_tables(simplex_tree, size)
    
    # Find rings in the graph
    rings = get_rings(edge_index, max_k=max_k)
    
    if len(rings) > 0:
        # Extend the tables with rings as 2-cells
        id_maps += [{}]
        simplex_tables += [[]]
        assert len(simplex_tables) == 3, simplex_tables
        for cell in rings:
            next_id = len(simplex_tables[2])
            id_maps[2][cell] = next_id
            simplex_tables[2].append(list(cell))

    return simplex_tables, id_maps


def get_ring_boundaries(ring):
    boundaries = list()
    for n in range(len(ring)):
        a = n
        if n + 1 == len(ring):
            b = 0
        else:
            b = n + 1
        # We represent the boundaries in lexicographic order
        # so to be compatible with 0- and 1- dim cells
        # extracted as simplices with gudhi
        boundaries.append(tuple(sorted([ring[a], ring[b]])))
    return sorted(boundaries)


def extract_boundaries_and_coboundaries_with_rings(simplex_tree, id_maps):
    """Build two maps: cell -> its coboundaries and cell -> its boundaries"""

    # Find boundaries and coboundaries up to edges by conveniently
    # invoking the code for simplicial complexes
    assert simplex_tree.dimension() <= 1
    boundaries_tables, boundaries, coboundaries = extract_boundaries_and_coboundaries_from_simplex_tree(
                                            simplex_tree, id_maps, simplex_tree.dimension())
    
    assert len(id_maps) <= 3
    if len(id_maps) == 3:
        # Extend tables with boundary and coboundary information of rings
        boundaries += [{}]
        coboundaries += [{}]
        boundaries_tables += [[]]
        for cell in id_maps[2]:
            cell_boundaries = get_ring_boundaries(cell)
            boundaries[2][cell] = list()
            boundaries_tables[2].append([])
            for boundary in cell_boundaries:
                assert boundary in id_maps[1], boundary
                boundaries[2][cell].append(boundary)
                if boundary not in coboundaries[1]:
                    coboundaries[1][boundary] = list()
                coboundaries[1][boundary].append(cell)
                boundaries_tables[2][-1].append(id_maps[1][boundary])
    
    return boundaries_tables, boundaries, coboundaries


def compute_ring_2complex(x: Union[Tensor, np.ndarray], edge_index: Union[Tensor, np.ndarray],
                          edge_attr: Optional[Union[Tensor, np.ndarray]],
                          size: int, y: Optional[Union[Tensor, np.ndarray]] = None, max_k: int = 7,
                          include_down_adj=True, init_method: str = 'sum',
                          init_edges=True, init_rings=False) -> Complex:
    """Generates a ring 2-complex of a pyG graph via graph-tool.

    Args:
        x: The feature matrix for the nodes of the graph (shape [num_vertices, num_v_feats])
        edge_index: The edge_index of the graph (a tensor of shape [2, num_edges])
        edge_attr: The feature matrix for the edges of the graph (shape [num_edges, num_e_feats])
        size: The number of nodes in the graph
        y: Labels for the graph nodes or a label for the whole graph.
        max_k: maximum length of rings to look for.
        include_down_adj: Whether to add down adj in the complex or not
        init_method: How to initialise features at higher levels.
    """
    assert x is not None
    assert isinstance(edge_index, np.ndarray) or isinstance(edge_index, Tensor)

    # For parallel processing with joblib we need to pass numpy arrays as inputs
    # Therefore, we convert here everything back to a tensor.
    if isinstance(x, np.ndarray):
        x = torch.tensor(x)
    if isinstance(edge_index, np.ndarray):
        edge_index = torch.tensor(edge_index)
    if isinstance(edge_attr, np.ndarray):
        edge_attr = torch.tensor(edge_attr)
    if isinstance(y, np.ndarray):
        y = torch.tensor(y)

    # Creates the gudhi-based simplicial complex up to edges
    simplex_tree = pyg_to_simplex_tree(edge_index, size)
    assert simplex_tree.dimension() <= 1
    if simplex_tree.dimension() == 0:
        assert edge_index.size(1) == 0

    # Builds tables of the cellular complexes at each level and their IDs
    simplex_tables, id_maps = build_tables_with_rings(edge_index, simplex_tree, size, max_k)
    assert len(id_maps) <= 3
    complex_dim = len(id_maps)-1

    # Extracts the boundaries and coboundaries of each simplex in the complex
    boundaries_tables, boundaries, co_boundaries = extract_boundaries_and_coboundaries_with_rings(simplex_tree, id_maps)

    # Computes the adjacencies between all the simplexes in the complex;
    # here we force complex dimension to be 2
    shared_boundaries, shared_coboundaries, lower_idx, upper_idx = build_adj(boundaries, co_boundaries, id_maps,
                                                                   complex_dim, include_down_adj)
    
    # Construct features for the higher dimensions
    xs = [x, None, None]
    constructed_features = construct_features(x, simplex_tables, init_method)
    if simplex_tree.dimension() == 0:
        assert len(constructed_features) == 1
    if init_rings and len(constructed_features) > 2:
        xs[2] = constructed_features[2]
    
    if init_edges and simplex_tree.dimension() >= 1:
        if edge_attr is None:
            xs[1] = constructed_features[1]
        # If we have edge-features we simply use them for 1-cells
        else:
            # If edge_attr is a list of scalar features, make it a matrix
            if edge_attr.dim() == 1:
                edge_attr = edge_attr.view(-1, 1)
            # Retrieve feats and check edge features are undirected
            ex = dict()
            for e, edge in enumerate(edge_index.numpy().T):
                canon_edge = tuple(sorted(edge))
                edge_id = id_maps[1][canon_edge]
                edge_feats = edge_attr[e]
                if edge_id in ex:
                    assert torch.equal(ex[edge_id], edge_feats)
                else:
                    ex[edge_id] = edge_feats

            # Build edge feature matrix
            max_id = max(ex.keys())
            edge_feats = []
            assert len(simplex_tables[1]) == max_id + 1
            for id in range(max_id + 1):
                edge_feats.append(ex[id])
            xs[1] = torch.stack(edge_feats, dim=0)
            assert xs[1].dim() == 2
            assert xs[1].size(0) == len(id_maps[1])
            assert xs[1].size(1) == edge_attr.size(1)

    # Initialise the node / complex labels
    v_y, complex_y = extract_labels(y, size)

    chains = []
    for i in range(complex_dim + 1):
        y = v_y if i == 0 else None
        chain = generate_chain(i, xs[i], upper_idx, lower_idx, shared_boundaries, shared_coboundaries,
                               simplex_tables, boundaries_tables, complex_dim=complex_dim, y=y)
        chains.append(chain)

    return Complex(*chains, y=complex_y, dimension=complex_dim)


def convert_graph_dataset_with_rings(dataset, max_ring_size=7, include_down_adj=False,
                                     init_method: str = 'sum', init_edges=True, init_rings=False,
                                     n_jobs=1):
    dimension = -1
    num_features = [None, None, None]

    def maybe_convert_to_numpy(x):
        if isinstance(x, Tensor):
            return x.numpy()
        return x

    # Process the dataset in parallel
    parallel = ProgressParallel(n_jobs=n_jobs, use_tqdm=True, total=len(dataset))
    # It is important we supply a numpy array here. tensors seem to slow joblib down significantly.
    complexes = parallel(delayed(compute_ring_2complex)(
        maybe_convert_to_numpy(data.x), maybe_convert_to_numpy(data.edge_index),
        maybe_convert_to_numpy(data.edge_attr),
        data.num_nodes, y=maybe_convert_to_numpy(data.y), max_k=max_ring_size,
        include_down_adj=include_down_adj, init_method=init_method,
        init_edges=init_edges, init_rings=init_rings) for data in dataset)

    # NB: here we perform additional checks to verify the order of complexes
    # corresponds to that of input graphs after _parallel_ conversion
    for c, complex in enumerate(complexes):

        # Handle dimension and number of features
        if complex.dimension > dimension:
            dimension = complex.dimension
        for dim in range(complex.dimension + 1):
            if num_features[dim] is None:
                num_features[dim] = complex.chains[dim].num_features
            else:
                assert num_features[dim] == complex.chains[dim].num_features

        # Validate against graph
        graph = dataset[c]
        assert torch.equal(complex.y, graph.y)
        assert torch.equal(complex.chains[0].x, graph.x)
        if complex.dimension >= 1:
            assert complex.chains[1].x.size(0) == (graph.edge_index.size(1) // 2)

    return complexes, dimension, num_features[:dimension+1]
