import itertools
import torch
import networkx as nx

from torch_geometric.utils import convert
from torch_geometric.data import Data

def check_edge_index_are_the_same(upper_index, edge_index):
    # These two tensors should have the same content but in different order.
    assert upper_index.size() == edge_index.size()
    num_edges = edge_index.size(1)

    edge_set1 = set()
    edge_set2 = set()
    for i in range(num_edges):
        e1, e2 = edge_index[0, i].item(), edge_index[1, i].item()
        edge1 = tuple(sorted([e1, e2]))
        edge_set1.add(edge1)

        e1, e2 = upper_index[0, i].item(), upper_index[1, i].item()
        edge2 = tuple(sorted([e1, e2]))
        edge_set2.add(edge2)

    assert edge_set1 == edge_set2


def get_table(boundary_index):
    elements = boundary_index.size(1)
    id_to_cell = dict()
    for i in range(elements):
        cell_id = boundary_index[1, i].item()
        boundary = boundary_index[0, i].item()
        if cell_id not in id_to_cell:
            id_to_cell[cell_id] = []
        id_to_cell[cell_id].append(boundary)
    return id_to_cell


def check_edge_attr_are_the_same(boundary_index, ex, edge_index, edge_attr):
    # The maximum node that has an edge must be the same.
    assert boundary_index[0, :].max() == edge_index.max()
    # The number of edges present in both tensors should be the same.
    assert boundary_index.size(1) == edge_index.size(1)

    id_to_edge = get_table(boundary_index)

    edge_to_id = dict()
    for edge_idx, edge in id_to_edge.items():
        edge_to_id[tuple(sorted(edge))] = edge_idx

    edges = boundary_index.size(1)
    for i in range(edges):
        e1, e2 = edge_index[0, i].item(), edge_index[1, i].item()
        edge = tuple(sorted([e1, e2]))

        edge_attr1 = ex[edge_to_id[edge]].squeeze()
        edge_attr2 = edge_attr[i].squeeze()
        
        # NB: edge feats may be multidimensional, so we cannot
        # generally use the `==` operator here
        assert torch.equal(edge_attr1, edge_attr2)


def get_rings(n, edge_index, max_ring):
    x = torch.zeros((n, 1))
    data = Data(x, edge_index=edge_index)
    graph = convert.to_networkx(data)

    def is_cycle_edge(i1, i2, cycle):
        if i2 == i1 + 1:
            return True
        if i1 == 0 and i2 == len(cycle) - 1:
            return True
        return False

    def is_chordless(cycle):
        for (i1, v1), (i2, v2) in itertools.combinations(enumerate(cycle), 2):
            if not is_cycle_edge(i1, i2, cycle) and graph.has_edge(v1, v2):
                return False
        return True

    nx_rings = set()
    for cycle in nx.simple_cycles(graph):
        # Because we need to use a DiGraph for this method, it will also return each edge
        # as a cycle. So we skip these together with cycles above the maximum length.
        if len(cycle) <= 2 or len(cycle) > max_ring:
            continue
        # We skip the cycles with chords
        if not is_chordless(cycle):
            continue
        # Store the cycle in a canonical form
        nx_rings.add(tuple(sorted(cycle)))

    return nx_rings


def get_complex_rings(r_boundary_index, e_boundary_index):
    # Construct the edge and ring tables
    id_to_ring = get_table(r_boundary_index)
    id_to_edge = get_table(e_boundary_index)

    rings = set()
    for ring, edges in id_to_ring.items():
        # Compose the two tables to extract the vertices in the ring.
        vertices = [vertex for edge in edges for vertex in id_to_edge[edge]]
        # Eliminate duplicates.
        vertices = set(vertices)
        # Store the ring in sorted order.
        rings.add(tuple(sorted(vertices)))
    return rings


def compare_complexes(yielded, expected, include_down_adj):
    
    assert yielded.dimension == expected.dimension
    assert torch.equal(yielded.y, expected.y)
    for dim in range(expected.dimension + 1):
        y_chain = yielded.chains[dim]
        e_chain = expected.chains[dim]
        assert y_chain.num_simplices == e_chain.num_simplices
        assert y_chain.num_simplices_up == e_chain.num_simplices_up
        assert y_chain.num_simplices_up == e_chain.num_simplices_up
        assert y_chain.num_simplices_down == e_chain.num_simplices_down, dim
        assert torch.equal(y_chain.x, e_chain.x)
        if dim > 0:
            assert torch.equal(y_chain.boundary_index, e_chain.boundary_index)
            if include_down_adj:
                if y_chain.lower_index is None:
                    assert e_chain.lower_index is None
                    assert y_chain.shared_boundaries is None
                    assert e_chain.shared_boundaries is None
                else:
                    assert torch.equal(y_chain.lower_index, e_chain.lower_index)
                    assert torch.equal(y_chain.shared_boundaries, e_chain.shared_boundaries)
        else:
            assert y_chain.boundary_index is None and e_chain.boundary_index is None
            assert y_chain.lower_index is None and e_chain.lower_index is None
            assert y_chain.shared_boundaries is None and e_chain.shared_boundaries is None
        if dim < expected.dimension:
            if y_chain.upper_index is None:
                assert e_chain.upper_index is None
                assert y_chain.shared_coboundaries is None
                assert e_chain.shared_coboundaries is None
            else:
                assert torch.equal(y_chain.upper_index, e_chain.upper_index)
                assert torch.equal(y_chain.shared_coboundaries, e_chain.shared_coboundaries)
        else:
            assert y_chain.upper_index is None and e_chain.upper_index is None
            assert y_chain.shared_coboundaries is None and e_chain.shared_coboundaries is None
    

def compare_complexes_without_2feats(yielded, expected, include_down_adj):
    
    assert yielded.dimension == expected.dimension
    assert torch.equal(yielded.y, expected.y)
    for dim in range(expected.dimension + 1):
        y_chain = yielded.chains[dim]
        e_chain = expected.chains[dim]
        assert y_chain.num_simplices == e_chain.num_simplices
        assert y_chain.num_simplices_up == e_chain.num_simplices_up
        assert y_chain.num_simplices_up == e_chain.num_simplices_up
        assert y_chain.num_simplices_down == e_chain.num_simplices_down, dim
        if dim > 0:
            assert torch.equal(y_chain.boundary_index, e_chain.boundary_index)
            if include_down_adj:
                if y_chain.lower_index is None:
                    assert e_chain.lower_index is None
                    assert y_chain.shared_boundaries is None
                    assert e_chain.shared_boundaries is None
                else:
                    assert torch.equal(y_chain.lower_index, e_chain.lower_index)
                    assert torch.equal(y_chain.shared_boundaries, e_chain.shared_boundaries)
        else:
            assert y_chain.boundary_index is None and e_chain.boundary_index is None
            assert y_chain.lower_index is None and e_chain.lower_index is None
            assert y_chain.shared_boundaries is None and e_chain.shared_boundaries is None
        if dim < expected.dimension:
            if y_chain.upper_index is None:
                assert e_chain.upper_index is None
                assert y_chain.shared_coboundaries is None
                assert e_chain.shared_coboundaries is None
            else:
                assert torch.equal(y_chain.upper_index, e_chain.upper_index)
                assert torch.equal(y_chain.shared_coboundaries, e_chain.shared_coboundaries)
        else:
            assert y_chain.upper_index is None and e_chain.upper_index is None
            assert y_chain.shared_coboundaries is None and e_chain.shared_coboundaries is None
        if dim != 2:
            assert torch.equal(y_chain.x, e_chain.x)
        else:
            assert y_chain.x is None and e_chain.x is None