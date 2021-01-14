import torch
from .utils import compute_connectivity, get_adj_index
from .ogbg_ppa_utils import draw_ppa_ego, extract_complex
from torch_sparse import coalesce
import copy
from ogb.graphproppred import PygGraphPropPredDataset
from torch_geometric.data import DataLoader
import numpy as np
import pytest
import os
from ..definitions import ROOT_DIR

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
    
# Test the extraction of higher-order connectivity

def validate_adj_dict(yielded, expected):
    for simplex in yielded:
        assert simplex in expected
        assert yielded[simplex] == expected[simplex]
    assert len(yielded) == len(expected)

def validate_index(yielded, expected, yielded_mapping, expected_mapping):
    
    # simplex -> tuple -> simplex
    mapping = {simplex: expected_mapping[yielded_mapping[simplex]] for simplex in yielded_mapping}
    size = torch.max(yielded).item()+1
    
    # coalesce
    # coalesced = coalesce(yielded.transpose(1,0), None, size, size)[0]
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
    node_upper_adj, _, node_mappings = get_adj_index(all_simplices_by_size[1],  upper_adjacencies[1])
    validate_index(node_upper_adj, expected_node_upper_index, node_mappings, tuples_to_nodes)

    return


def test_edge_upper_adj(yielded_connectivity, house_edge_upper_adjacency):
    
    expected_edge_upper, expected_edge_upper_index, tuples_to_edges = house_edge_upper_adjacency
    upper_adjacencies, lower_adjacencies, all_simplices, all_simplices_by_size  = yielded_connectivity

    validate_adj_dict(upper_adjacencies[2], expected_edge_upper)
    edge_upper_adj, _, edge_mappings = get_adj_index(all_simplices_by_size[2],  upper_adjacencies[2])
    validate_index(edge_upper_adj, expected_edge_upper_index, edge_mappings, tuples_to_edges)
    
    return
    
    
def test_edge_lower_adj(yielded_connectivity, house_edge_lower_adjacency):
    
    expected_edge_lower, expected_edge_lower_index, tuples_to_edges = house_edge_lower_adjacency
    upper_adjacencies, lower_adjacencies, all_simplices, all_simplices_by_size  = yielded_connectivity
    
    validate_adj_dict(lower_adjacencies[2], expected_edge_lower)
    edge_lower_adj, _, edge_mappings = get_adj_index(all_simplices_by_size[2],  lower_adjacencies[2])
    validate_index(edge_lower_adj, expected_edge_lower_index, edge_mappings, tuples_to_edges)

    return


def test_clique_complex(house_edge_index, house_node_upper_adjacency, house_edge_upper_adjacency, house_edge_lower_adjacency):
    # Test the overall construction of a clique-Complex object from a ppa egonet -like structure

    d_name = 'ogbg-ppa'
    dataset = PygGraphPropPredDataset(name=d_name, root=os.path.join(ROOT_DIR,'data_exploration/dataset'))
    split_idx = dataset.get_idx_split() 
    train_loader = DataLoader(dataset[split_idx["train"]], batch_size=1, shuffle=False, )

    for ego in train_loader: break
    house = copy.deepcopy(ego)
    house.edge_index = house_edge_index
    house.edge_attr = torch.ones((house_edge_index.shape[1]), house.edge_attr.shape[1])
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