import torch
from torch_geometric.data import Data
from data.utils import compute_connectivity, get_adj_index, compute_clique_complex_with_gudhi, convert_graph_dataset_with_gudhi, compute_ring_2complex_with_graphtool_and_gudhi
from data.ogbg_ppa_utils import draw_ppa_ego, extract_complex
from torch_sparse import coalesce
from data.complex import ComplexBatch
import numpy as np
import pytest

# TODO: Gudhi does not preserve the order of the edges in edge_index. It uses a lexicographic order
# Once we care about edge_features at initialisation, we need to make the order the same.

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
    '''
        4
       / \
      3---2
      |   |
      0---1 
    
        .
       5 4
      . 3 .
      1   2
      . 0 . 
     
        .
       /0\
      .---.
      |   |
      .---. 
    '''
    house = Data(edge_index=house_edge_index, x=torch.range(0, 4).view(5, 1), y=torch.tensor([1]))
    house.num_nodes = house_edge_index.max().item() + 1

    house_complex = compute_clique_complex_with_gudhi(house.x, house.edge_index, house.num_nodes,
                                                      y=house.y)

    # Check the number of simplices
    assert house_complex.nodes.num_simplices_down is None
    assert house_complex.nodes.num_simplices_up == 6
    assert house_complex.edges.num_simplices_down == 5
    assert house_complex.edges.num_simplices_up == 1
    assert house_complex.triangles.num_simplices_down == 6
    assert house_complex.triangles.num_simplices_up == 0

    # Check the returned parameters
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
    assert v_params.kwargs['face_attr'] is None

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

    assert torch.equal(e_params.kwargs['face_attr'], house.x)
    assert list(e_params.kwargs['face_index'].size()) == [2, 2*house_complex.edges.num_simplices]
    assert torch.equal(e_params.kwargs['face_index'][1], torch.LongTensor([0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5]))
    assert torch.equal(e_params.kwargs['face_index'][0], torch.LongTensor([0, 1, 0, 3, 1, 2, 2, 3, 2, 4, 3, 4]))

    t_params = house_complex.get_chain_params(dim=2)
    expected_t_x = torch.tensor([[9]], dtype=torch.float)
    assert torch.equal(t_params.x, expected_t_x)
    assert t_params.down_index is None
    assert t_params.up_index is None
    assert torch.equal(t_params.kwargs['face_attr'], expected_e_x)
    assert list(t_params.kwargs['face_index'].size()) == [2, 3*house_complex.triangles.num_simplices]
    assert torch.equal(t_params.kwargs['face_index'][1], torch.LongTensor([0, 0, 0])) 
    assert torch.equal(t_params.kwargs['face_index'][0], torch.LongTensor([3, 4, 5])) 

    assert torch.equal(house_complex.y, house.y)


def test_gudhi_clique_complex_dataset_conversion(house_edge_index):
    house1 = Data(edge_index=house_edge_index, x=torch.range(0, 4).view(5, 1), y=torch.tensor([1]))
    house2 = Data(edge_index=house_edge_index, x=torch.range(0, 4).view(5, 1), y=torch.tensor([1]))
    house3 = Data(edge_index=house_edge_index, x=torch.range(0, 4).view(5, 1), y=torch.tensor([1]))
    dataset = [house1, house2, house3]

    complexes, dim, num_features = convert_graph_dataset_with_gudhi(dataset, expansion_dim=3)
    assert dim == 2
    assert len(num_features) == 3
    for i in range(len(num_features)):
        assert num_features[i] == 1
    assert len(complexes) == 3
    for i in range(len(complexes)):
        # Do some basic checks for each complex.
        assert complexes[i].dimension == 2
        assert complexes[i].nodes.face_index is None
        assert list(complexes[i].edges.face_index.size()) == [2, 2*6]
        assert list(complexes[i].triangles.face_index.size()) == [2, 3*1]
        assert complexes[i].edges.lower_index.size(1) == 18
        assert torch.equal(complexes[i].nodes.x, house1.x)
        assert torch.equal(complexes[i].y, house1.y)


def test_gudhi_clique_complex_dataset_conversion_with_down_adj_excluded(house_edge_index):
    house1 = Data(edge_index=house_edge_index, x=torch.range(0, 4).view(5, 1), y=torch.tensor([1]))
    house2 = Data(edge_index=house_edge_index, x=torch.range(0, 4).view(5, 1), y=torch.tensor([1]))
    house3 = Data(edge_index=house_edge_index, x=torch.range(0, 4).view(5, 1), y=torch.tensor([1]))
    dataset = [house1, house2, house3]

    complexes, dim, num_features = convert_graph_dataset_with_gudhi(dataset, expansion_dim=3,
                                                                    include_down_adj=False)
    assert dim == 2
    assert len(num_features) == 3
    for i in range(len(num_features)):
        assert num_features[i] == 1
    assert len(complexes) == 3
    for i in range(len(complexes)):
        # Do some basic checks for each complex.
        assert complexes[i].dimension == 2
        assert complexes[i].nodes.face_index is None
        assert list(complexes[i].edges.face_index.size()) == [2, 2*6]
        assert list(complexes[i].triangles.face_index.size()) == [2, 3*1]
        assert complexes[i].edges.lower_index is None
        assert torch.equal(complexes[i].nodes.x, house1.x)
        assert torch.equal(complexes[i].y, house1.y)


def test_gudhi_integration_with_batching_without_adj(house_edge_index):
    house1 = Data(edge_index=house_edge_index, x=torch.range(0, 4).view(5, 1),
                  y=torch.tensor([1]))
    house2 = Data(edge_index=house_edge_index, x=torch.range(0, 4).view(5, 1),
                  y=torch.tensor([1]))
    house3 = Data(edge_index=house_edge_index, x=torch.range(0, 4).view(5, 1),
                  y=torch.tensor([1]))
    dataset = [house1, house2, house3]

    # Without down-adj
    complexes, dim, num_features = convert_graph_dataset_with_gudhi(dataset, expansion_dim=3,
                                                                    include_down_adj=False)

    batch = ComplexBatch.from_complex_list(complexes)
    assert batch.dimension == 2
    assert batch.edges.lower_index is None
    assert batch.nodes.face_index is None
    assert list(batch.edges.face_index.size()) == [2, 3*2*6]
    assert list(batch.triangles.face_index.size()) == [2, 1*3*3]


def test_gudhi_integration_with_batching_with_adj(house_edge_index):
    house1 = Data(edge_index=house_edge_index, x=torch.range(0, 4).view(5, 1),
                  y=torch.tensor([1]))
    house2 = Data(edge_index=house_edge_index, x=torch.range(0, 4).view(5, 1),
                  y=torch.tensor([1]))
    house3 = Data(edge_index=house_edge_index, x=torch.range(0, 4).view(5, 1),
                  y=torch.tensor([1]))
    dataset = [house1, house2, house3]

    # Without down-adj
    complexes, dim, num_features = convert_graph_dataset_with_gudhi(dataset, expansion_dim=3,
                                                                    include_down_adj=True)

    batch = ComplexBatch.from_complex_list(complexes)
    assert batch.dimension == 2
    assert batch.edges.lower_index.size(1) == 18*3
    assert list(batch.edges.face_index.size()) == [2, 3*2*6]
    assert list(batch.triangles.face_index.size()) == [2, 1*3*3]


def test_graphtool_and_gudhi_cell_complex(house_edge_index):
    house = Data(edge_index=house_edge_index, x=torch.range(0, 4).view(5, 1), y=torch.tensor([1]))
    house.num_nodes = house_edge_index.max().item() + 1

    house_complex = compute_ring_2complex_with_graphtool_and_gudhi(house.x, house.edge_index, house.num_nodes,
                                                                   max_k=4, y=house.y)

    # Check the number of simplices
    assert house_complex.nodes.num_simplices_down is None
    assert house_complex.nodes.num_simplices_up == 6
    assert house_complex.nodes.face_index is None
    assert house_complex.edges.num_simplices_down == 5
    assert house_complex.edges.num_simplices_up == 2
    assert list(house_complex.edges.face_index.size()) == [2, 2*6]
    assert house_complex.chains[2].num_simplices == 2
    assert house_complex.chains[2].num_simplices_down == 6
    assert house_complex.chains[2].num_simplices_up == 0
    assert list(house_complex.chains[2].face_index.size()) == [2, 3+4]

    # Check the returned parameters
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
    assert v_params.kwargs['face_attr'] is None

    e_params = house_complex.get_chain_params(dim=1)
    expected_e_x = torch.tensor([[1], [3], [3], [5], [6], [7]], dtype=torch.float)
    assert torch.equal(e_params.x, expected_e_x)

    expected_e_up_index = torch.tensor([[0, 1, 0, 2, 0, 3, 1, 2, 1, 3, 2, 3, 3, 4, 3, 5, 4, 5],
                                        [1, 0, 2, 0, 3, 0, 2, 1, 3, 1, 3, 2, 4, 3, 5, 3, 5, 4]], dtype=torch.long)
    assert torch.equal(e_params.up_index, expected_e_up_index)

    expected_e_up_attr = torch.tensor([[6], [6], [6], [6], [6], [6], [6], [6], [6], [6], [6], [6], [9], [9], [9], [9], [9], [9]], dtype=torch.float)
    assert torch.equal(e_params.kwargs['up_attr'], expected_e_up_attr)

    expected_e_down_index = torch.tensor([[0, 1, 0, 2, 2, 3, 2, 4, 3, 4, 1, 3, 1, 5, 3, 5, 4, 5],
                                          [1, 0, 2, 0, 3, 2, 4, 2, 4, 3, 3, 1, 5, 1, 5, 3, 5, 4]],
                                         dtype=torch.long)
    assert torch.equal(e_params.down_index, expected_e_down_index)
    expected_e_down_attr = torch.tensor([[0], [0], [1], [1], [2], [2], [2], [2], [2], [2],
                                         [3], [3], [3], [3], [3], [3], [4], [4]],
                                        dtype=torch.float)
    assert torch.equal(e_params.kwargs['down_attr'], expected_e_down_attr)

    assert torch.equal(e_params.kwargs['face_attr'], house.x)
    assert list(e_params.kwargs['face_index'].size()) == [2, 2*house_complex.edges.num_simplices]
    assert torch.equal(e_params.kwargs['face_index'][1], torch.LongTensor([0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5]))
    assert torch.equal(e_params.kwargs['face_index'][0], torch.LongTensor([0, 1, 0, 3, 1, 2, 2, 3, 2, 4, 3, 4]))

    t_params = house_complex.get_chain_params(dim=2)
    expected_t_x = torch.tensor([[6], [9]], dtype=torch.float)
    assert torch.equal(t_params.x, expected_t_x)
    expected_t_down_index = torch.tensor([[0, 1],
                                          [1, 0]],
                                         dtype=torch.long)
    assert torch.equal(t_params.down_index, expected_t_down_index)
    assert t_params.up_index is None
    assert torch.equal(t_params.kwargs['face_attr'], expected_e_x)
    expected_t_face_index = torch.tensor([[0, 1, 2, 3, 3, 4, 5],
                                          [0, 0, 0, 0, 1, 1, 1]], dtype=torch.long)
    assert torch.equal(t_params.kwargs['face_index'], expected_t_face_index)
    assert torch.equal(house_complex.y, house.y)