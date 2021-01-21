import torch
from torch_geometric.data import Data
from data.utils import compute_connectivity, get_adj_index, compute_clique_complex_with_gudhi
from data.ogbg_ppa_utils import draw_ppa_ego, extract_complex
from torch_sparse import coalesce
import numpy as np
import pytest

# Define here below the `house graph` and the expected connectivity to be constructed.
# The `house graph` (3-2-4 is a filled triangle):
#    4
#   / \
#  3---2
#  |   |
#  0---1 
#
#    .
#   4 5
#  . 2 .
#  3   1
#  . 0 . 
# 
#    .
#   /0\
#  .---.
#  |   |
#  .---. 

# if you want to visualize it, uncomment the following line
# draw_ppa_ego(edge_index.numpy().T, from_edge_list=True, with_labels=True)

@pytest.fixture
def house_edge_index():
    return torch.tensor([[0, 0, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4],
                               [1, 3, 0, 2, 1, 3, 4, 0, 2, 4, 2, 3]], dtype=torch.long)

@pytest.fixture
def house_node_upper_adjacency():
    expected_node_upper = {
        (0,): {(1,), (3,)},
        (1,): {(0,), (2,)},
        (2,): {(1,), (3,), (4,)},
        (3,): {(0,), (2,), (4,)},
        (4,): {(2,), (3,)}}
    tuples_to_nodes = {
       (0,): 0,
       (1,): 1,
       (2,): 2,
       (3,): 3,
       (4,): 4,}
    expected_node_upper_index = torch.tensor([[0, 0, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4],
                                              [1, 3, 0, 2, 1, 3, 4, 0, 2, 4, 2, 3]], dtype=torch.long)
    return expected_node_upper, expected_node_upper_index, tuples_to_nodes

@pytest.fixture
def house_edge_upper_adjacency():
    expected_edge_upper = {
        (2,3): {(2,4), (3,4)},
        (2,4): {(2,3), (3,4)},
        (3,4): {(2,3), (2,4)}}
    tuples_to_edges = {
       (0,1): 0,
       (1,2): 1,
       (2,3): 2,
       (0,3): 3,
       (3,4): 4,
       (2,4): 5}
    expected_edge_upper_index = torch.tensor([[2, 2, 4, 4, 5, 5],
                                              [4, 5, 2, 5, 2, 4]], dtype=torch.long)
    return expected_edge_upper, expected_edge_upper_index, tuples_to_edges

@pytest.fixture
def house_edge_lower_adjacency():
    expected_edge_lower = {
        (0,1): {(1,2), (0,3)},
        (1,2): {(0,1), (2,3), (2,4)},
        (2,3): {(1,2), (2,4), (0,3), (3,4)},
        (0,3): {(0,1), (2,3), (3,4)},
        (2,4): {(1,2), (2,3), (3,4)},
        (3,4): {(0,3), (2,3), (2,4)}}
    tuples_to_edges = {
       (0,1): 0,
       (1,2): 1,
       (2,3): 2,
       (0,3): 3,
       (3,4): 4,
       (2,4): 5}
    expected_edge_lower_index = torch.tensor([[0, 0, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 4, 4,  4, 5, 5, 5],
                                              [1, 3, 0, 2, 5, 1, 3, 4, 5, 0, 2, 4, 2, 3, 5, 1, 2, 4]], dtype=torch.long)
    return expected_edge_lower, expected_edge_lower_index, tuples_to_edges

@pytest.fixture
def house_facets():
    all_facets = {(2,3,4), (0,1), (1,2), (0,3)}
    all_facets_by_size = {
        3: {(2,3,4)},
        2: {(0,1), (1,2), (0,3)},
        1: set()}
    max_size = 3
    return all_facets, all_facets_by_size, max_size

@pytest.fixture
def yielded_connectivity(house_facets):
    all_facets, all_facets_by_size, max_size = house_facets
    upper_adjacencies, lower_adjacencies, all_simplices, all_simplices_by_size, _ = compute_connectivity(all_facets,  all_facets_by_size, max_size)
    return upper_adjacencies, lower_adjacencies, all_simplices, all_simplices_by_size
    
# Test the extraction of higher-dim connectivity

def validate_adj_dict(yielded, expected):
    for simplex in yielded:
        assert simplex in expected
        assert yielded[simplex] == expected[simplex]
    assert len(yielded) == len(expected)

def validate_index(yielded, expected, yielded_mapping, expected_mapping):
    
    # simplex -> tuple -> simplex
    mapping = {simplex: expected_mapping[tuple(yielded_mapping[simplex].numpy())] for simplex in range(yielded_mapping.shape[0])}
    size = torch.max(yielded).item()+1
    
    # coalesce
    coalesced = coalesce(yielded, None, size, size)[0]
    translated = list()
    
    # translate in terms of the expected mapping
    for edge in coalesced.numpy().T:
        a, b = edge
        translated.append([mapping[a], mapping[b]])
    translated = torch.LongTensor(translated)
    
    # coalesce back again
    translated = coalesce(translated.transpose(1,0), None, size, size)[0]
    translated = coalesce(translated, None, size, size)[0]
    
    # validate
    assert np.all(np.equal(translated.numpy(), expected.numpy())), '{} vs. {}'.format(translated, expected)

    
def test_node_upper_adj(yielded_connectivity, house_node_upper_adjacency):
    
    expected_node_upper, expected_node_upper_index, tuples_to_nodes = house_node_upper_adjacency
    upper_adjacencies, lower_adjacencies, all_simplices, all_simplices_by_size  = yielded_connectivity
    
    validate_adj_dict(upper_adjacencies[1], expected_node_upper)
    node_upper_adj, node_mappings = get_adj_index(all_simplices_by_size[1],  upper_adjacencies[1], 1)
    validate_index(node_upper_adj, expected_node_upper_index, node_mappings, tuples_to_nodes)

    return


def test_edge_upper_adj(yielded_connectivity, house_edge_upper_adjacency):
    
    expected_edge_upper, expected_edge_upper_index, tuples_to_edges = house_edge_upper_adjacency
    upper_adjacencies, lower_adjacencies, all_simplices, all_simplices_by_size  = yielded_connectivity

    validate_adj_dict(upper_adjacencies[2], expected_edge_upper)
    edge_upper_adj, edge_mappings = get_adj_index(all_simplices_by_size[2],  upper_adjacencies[2], 2)
    validate_index(edge_upper_adj, expected_edge_upper_index, edge_mappings, tuples_to_edges)
    
    return
    
    
def test_edge_lower_adj(yielded_connectivity, house_edge_lower_adjacency):
    
    expected_edge_lower, expected_edge_lower_index, tuples_to_edges = house_edge_lower_adjacency
    upper_adjacencies, lower_adjacencies, all_simplices, all_simplices_by_size  = yielded_connectivity
    
    validate_adj_dict(lower_adjacencies[2], expected_edge_lower)
    edge_lower_adj, edge_mappings = get_adj_index(all_simplices_by_size[2],  lower_adjacencies[2], 2)
    validate_index(edge_lower_adj, expected_edge_lower_index, edge_mappings, tuples_to_edges)

    return


def test_clique_complex(house_edge_index, house_node_upper_adjacency, house_edge_upper_adjacency, house_edge_lower_adjacency):

    # Test the overall construction of a clique-Complex object from a ppa egonet -like structure

    house = Data(edge_index=house_edge_index, edge_attr=torch.ones((house_edge_index.shape[1]), 7))
    house.num_nodes = house_edge_index.max().item() + 1

    house_complex, upper, lower = extract_complex(house, 0.0, 0.0, max_size=3)
    expected_node_upper, expected_node_upper_index, tuples_to_nodes = house_node_upper_adjacency
    expected_edge_upper, expected_edge_upper_index, tuples_to_edges = house_edge_upper_adjacency
    expected_edge_lower, expected_edge_lower_index, _ = house_edge_lower_adjacency
    
    validate_adj_dict(upper[1], expected_node_upper)  # <- node upper adjacency
    validate_adj_dict(upper[2], expected_edge_upper)  # <- edge upper adjacency
    validate_adj_dict(lower[2], expected_edge_lower)  # <- edge lower adjacency

    nodes = house_complex.chains[0]
    edges = house_complex.chains[1]
    validate_index(nodes.upper_index, expected_node_upper_index, nodes.mapping, tuples_to_nodes)  # <- node upper index
    validate_index(edges.upper_index, expected_edge_upper_index, edges.mapping, tuples_to_edges)  # <- edge upper index
    validate_index(edges.lower_index, expected_edge_lower_index, edges.mapping, tuples_to_edges)  # <- edge lower index
    
    return


def test_gudhi_clique_complex(house_edge_index):
    house = Data(edge_index=house_edge_index, x=torch.range(0, 4).view(5, 1))
    house.num_nodes = house_edge_index.max().item() + 1

    house_complex = compute_clique_complex_with_gudhi(house.x, house.edge_index, house.num_nodes)
    
    v_params = house_complex.get_chain_params(dim=0)
    assert torch.equal(v_params.x, house.x)
    assert v_params.down_index is None

    expected_v_up_index = torch.tensor([[0, 1, 0, 3, 1, 2, 2, 3, 2, 4, 3, 4],
                                        [1, 0, 3, 0, 2, 1, 3, 2, 4, 2, 4, 3]], dtype=torch.long)
    assert torch.equal(v_params.up_index, expected_v_up_index)

    expected_v_up_attr = torch.tensor([[1], [1], [3], [3], [3], [3],
                                       [5], [5], [6], [6], [7], [7]], dtype=torch.float)
    assert torch.equal(v_params.kwargs['up_attr'], expected_v_up_attr)
    assert v_params.kwargs['down_attr'] is None

    e_params = house_complex.get_chain_params(dim=1)
    expected_e_x = torch.tensor([[1], [3], [3], [5], [6], [7]], dtype=torch.float)
    assert torch.equal(e_params.x, expected_e_x)

    expected_e_up_index = torch.tensor([[3, 4, 3, 5, 4, 5],
                                        [4, 3, 5, 3, 5, 4]], dtype=torch.long)
    assert torch.equal(e_params.up_index, expected_e_up_index)

    expected_e_up_attr = torch.tensor([[9], [9], [9], [9], [9], [9]], dtype=torch.float)
    assert torch.equal(e_params.kwargs['up_attr'], expected_e_up_attr)

    expected_e_down_index = torch.tensor([[0, 1, 0, 2, 2, 3, 2, 4, 3, 4, 1, 3, 1, 5, 3, 5, 4, 5],
                                          [1, 0, 2, 0, 3, 2, 4, 2, 4, 3, 3, 1, 5, 1, 5, 3, 5, 4]],
                                         dtype=torch.long)
    assert torch.equal(e_params.down_index, expected_e_down_index)
    expected_e_down_attr = torch.tensor([[0], [0], [1], [1], [2], [2], [2], [2], [2], [2],
                                         [3], [3], [3], [3], [3], [3], [4], [4]],
                                        dtype=torch.float)
    assert torch.equal(e_params.kwargs['down_attr'], expected_e_down_attr)

    t_params = house_complex.get_chain_params(dim=2)
    expected_t_x = torch.tensor([[9]], dtype=torch.float)
    assert torch.equal(t_params.x, expected_t_x)
    assert t_params.down_index is None
    assert t_params.up_index is None





