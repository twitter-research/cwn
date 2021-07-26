import torch

from torch_geometric.data import Data
from data.utils import compute_clique_complex_with_gudhi, compute_ring_2complex
from data.utils import convert_graph_dataset_with_gudhi, convert_graph_dataset_with_rings
from data.complex import ComplexBatch
from data.dummy_complexes import convert_to_graph, get_testing_complex_list
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


@pytest.fixture
def house_edge_index():
    return torch.tensor([[0, 0, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4],
                         [1, 3, 0, 2, 1, 3, 4, 0, 2, 4, 2, 3]], dtype=torch.long)


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
    house = Data(edge_index=house_edge_index, x=torch.arange(0, 5, dtype=torch.float).view(5, 1), y=torch.tensor([1]))
    house.num_nodes = house_edge_index.max().item() + 1

    house_complex = compute_clique_complex_with_gudhi(house.x, house.edge_index, house.num_nodes,
                                                      y=house.y)

    # Check the number of simplices
    assert house_complex.nodes.num_cells_down is None
    assert house_complex.nodes.num_cells_up == 6
    assert house_complex.edges.num_cells_down == 5
    assert house_complex.edges.num_cells_up == 1
    assert house_complex.two_cells.num_cells_down == 6
    assert house_complex.two_cells.num_cells_up == 0

    # Check the returned parameters
    v_params = house_complex.get_cochain_params(dim=0)
    assert torch.equal(v_params.x, house.x)
    assert v_params.down_index is None

    expected_v_up_index = torch.tensor([[0, 1, 0, 3, 1, 2, 2, 3, 2, 4, 3, 4],
                                        [1, 0, 3, 0, 2, 1, 3, 2, 4, 2, 4, 3]], dtype=torch.long)
    assert torch.equal(v_params.up_index, expected_v_up_index)

    expected_v_up_attr = torch.tensor([[1], [1], [3], [3], [3], [3],
                                       [5], [5], [6], [6], [7], [7]], dtype=torch.float)
    assert torch.equal(v_params.kwargs['up_attr'], expected_v_up_attr)
    assert v_params.kwargs['down_attr'] is None
    assert v_params.kwargs['boundary_attr'] is None

    e_params = house_complex.get_cochain_params(dim=1)
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

    assert torch.equal(e_params.kwargs['boundary_attr'], house.x)
    assert list(e_params.kwargs['boundary_index'].size()) == [2, 2*house_complex.edges.num_cells]
    assert torch.equal(e_params.kwargs['boundary_index'][1], torch.LongTensor([0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5]))
    assert torch.equal(e_params.kwargs['boundary_index'][0], torch.LongTensor([0, 1, 0, 3, 1, 2, 2, 3, 2, 4, 3, 4]))

    t_params = house_complex.get_cochain_params(dim=2)
    expected_t_x = torch.tensor([[9]], dtype=torch.float)
    assert torch.equal(t_params.x, expected_t_x)
    assert t_params.down_index is None
    assert t_params.up_index is None
    assert torch.equal(t_params.kwargs['boundary_attr'], expected_e_x)
    assert list(t_params.kwargs['boundary_index'].size()) == [2, 3*house_complex.two_cells.num_cells]
    assert torch.equal(t_params.kwargs['boundary_index'][1], torch.LongTensor([0, 0, 0]))
    assert torch.equal(t_params.kwargs['boundary_index'][0], torch.LongTensor([3, 4, 5]))

    assert torch.equal(house_complex.y, house.y)


def test_gudhi_clique_complex_dataset_conversion(house_edge_index):
    house1 = Data(edge_index=house_edge_index, x=torch.arange(0, 5, dtype=torch.float).view(5, 1), y=torch.tensor([1]))
    house2 = Data(edge_index=house_edge_index, x=torch.arange(0, 5, dtype=torch.float).view(5, 1), y=torch.tensor([1]))
    house3 = Data(edge_index=house_edge_index, x=torch.arange(0, 5, dtype=torch.float).view(5, 1), y=torch.tensor([1]))
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
        assert complexes[i].nodes.boundary_index is None
        assert list(complexes[i].edges.boundary_index.size()) == [2, 2*6]
        assert list(complexes[i].two_cells.boundary_index.size()) == [2, 3*1]
        assert complexes[i].edges.lower_index.size(1) == 18
        assert torch.equal(complexes[i].nodes.x, house1.x)
        assert torch.equal(complexes[i].y, house1.y)


def test_gudhi_clique_complex_dataset_conversion_with_down_adj_excluded(house_edge_index):
    house1 = Data(edge_index=house_edge_index, x=torch.arange(0, 5, dtype=torch.float).view(5, 1), y=torch.tensor([1]))
    house2 = Data(edge_index=house_edge_index, x=torch.arange(0, 5, dtype=torch.float).view(5, 1), y=torch.tensor([1]))
    house3 = Data(edge_index=house_edge_index, x=torch.arange(0, 5, dtype=torch.float).view(5, 1), y=torch.tensor([1]))
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
        assert complexes[i].nodes.boundary_index is None
        assert list(complexes[i].edges.boundary_index.size()) == [2, 2*6]
        assert list(complexes[i].two_cells.boundary_index.size()) == [2, 3*1]
        assert complexes[i].edges.lower_index is None
        assert torch.equal(complexes[i].nodes.x, house1.x)
        assert torch.equal(complexes[i].y, house1.y)


def test_gudhi_integration_with_batching_without_adj(house_edge_index):
    house1 = Data(edge_index=house_edge_index, x=torch.arange(0, 5, dtype=torch.float).view(5, 1),
                  y=torch.tensor([1]))
    house2 = Data(edge_index=house_edge_index, x=torch.arange(0, 5, dtype=torch.float).view(5, 1),
                  y=torch.tensor([1]))
    house3 = Data(edge_index=house_edge_index, x=torch.arange(0, 5, dtype=torch.float).view(5, 1),
                  y=torch.tensor([1]))
    dataset = [house1, house2, house3]

    # Without down-adj
    complexes, dim, num_features = convert_graph_dataset_with_gudhi(dataset, expansion_dim=3,
                                                                    include_down_adj=False)

    batch = ComplexBatch.from_complex_list(complexes)
    assert batch.dimension == 2
    assert batch.edges.lower_index is None
    assert batch.nodes.boundary_index is None
    assert list(batch.edges.boundary_index.size()) == [2, 3*2*6]
    assert list(batch.two_cells.boundary_index.size()) == [2, 1*3*3]


def test_gudhi_integration_with_batching_with_adj(house_edge_index):
    house1 = Data(edge_index=house_edge_index, x=torch.arange(0, 5, dtype=torch.float).view(5, 1),
                  y=torch.tensor([1]))
    house2 = Data(edge_index=house_edge_index, x=torch.arange(0, 5, dtype=torch.float).view(5, 1),
                  y=torch.tensor([1]))
    house3 = Data(edge_index=house_edge_index, x=torch.arange(0, 5, dtype=torch.float).view(5, 1),
                  y=torch.tensor([1]))
    dataset = [house1, house2, house3]

    # Without down-adj
    complexes, dim, num_features = convert_graph_dataset_with_gudhi(dataset, expansion_dim=3,
                                                                    include_down_adj=True)

    batch = ComplexBatch.from_complex_list(complexes)
    assert batch.dimension == 2
    assert batch.edges.lower_index.size(1) == 18*3
    assert list(batch.edges.boundary_index.size()) == [2, 3*2*6]
    assert list(batch.two_cells.boundary_index.size()) == [2, 1*3*3]
    

def test_construction_of_ring_2complex(house_edge_index):
    house = Data(edge_index=house_edge_index, x=torch.arange(0, 5, dtype=torch.float).view(5, 1), y=torch.tensor([1]))
    house.num_nodes = house_edge_index.max().item() + 1

    house_complex = compute_ring_2complex(house.x, house.edge_index, None, house.num_nodes,
                                          max_k=4, y=house.y, init_rings=True)

    # Check the number of cells
    assert house_complex.nodes.num_cells_down is None
    assert house_complex.nodes.num_cells_up == 6
    assert house_complex.nodes.boundary_index is None
    assert house_complex.edges.num_cells_down == 5
    assert house_complex.edges.num_cells_up == 2
    assert list(house_complex.edges.boundary_index.size()) == [2, 2*6]
    assert house_complex.cochains[2].num_cells == 2
    assert house_complex.cochains[2].num_cells_down == 6
    assert house_complex.cochains[2].num_cells_up == 0
    assert list(house_complex.cochains[2].boundary_index.size()) == [2, 3+4]

    # Check the returned parameters
    v_params = house_complex.get_cochain_params(dim=0)
    assert torch.equal(v_params.x, house.x)
    assert v_params.down_index is None

    expected_v_up_index = torch.tensor([[0, 1, 0, 3, 1, 2, 2, 3, 2, 4, 3, 4],
                                        [1, 0, 3, 0, 2, 1, 3, 2, 4, 2, 4, 3]], dtype=torch.long)
    assert torch.equal(v_params.up_index, expected_v_up_index)

    expected_v_up_attr = torch.tensor([[1], [1], [3], [3], [3], [3],
                                       [5], [5], [6], [6], [7], [7]], dtype=torch.float)
    assert torch.equal(v_params.kwargs['up_attr'], expected_v_up_attr)
    assert v_params.kwargs['down_attr'] is None
    assert v_params.kwargs['boundary_attr'] is None

    e_params = house_complex.get_cochain_params(dim=1)
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

    assert torch.equal(e_params.kwargs['boundary_attr'], house.x)
    assert list(e_params.kwargs['boundary_index'].size()) == [2, 2*house_complex.edges.num_cells]
    assert torch.equal(e_params.kwargs['boundary_index'][1], torch.LongTensor([0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5]))
    assert torch.equal(e_params.kwargs['boundary_index'][0], torch.LongTensor([0, 1, 0, 3, 1, 2, 2, 3, 2, 4, 3, 4]))

    t_params = house_complex.get_cochain_params(dim=2)
    expected_t_x = torch.tensor([[6], [9]], dtype=torch.float)
    assert torch.equal(t_params.x, expected_t_x)
    expected_t_down_index = torch.tensor([[0, 1],
                                          [1, 0]],
                                         dtype=torch.long)
    assert torch.equal(t_params.down_index, expected_t_down_index)
    expected_t_down_attr = torch.tensor([[5], [5]], dtype=torch.float)
    assert torch.equal(t_params.kwargs['down_attr'], expected_t_down_attr)
    
    assert t_params.up_index is None
    assert torch.equal(t_params.kwargs['boundary_attr'], expected_e_x)
    expected_t_boundary_index = torch.tensor([[0, 1, 2, 3, 3, 4, 5],
                                          [0, 0, 0, 0, 1, 1, 1]], dtype=torch.long)
    assert torch.equal(t_params.kwargs['boundary_index'], expected_t_boundary_index)
    assert torch.equal(house_complex.y, house.y)
    

def test_construction_of_ring_2complex_with_edge_feats(house_edge_index):
    
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
      | 1 |
      .---. 
    '''
    
    edge_attr = torch.FloatTensor(
                            [[0.0, 1.0],
                             [0.0, 3.0],
                             [0.0, 1.0],
                             [1.0, 2.0],
                             [1.0, 2.0],
                             [2.0, 3.0],
                             [2.0, 4.0],
                             [0.0, 3.0],
                             [2.0, 3.0],
                             [3.0, 4.0],
                             [2.0, 4.0],
                             [3.0, 4.0]])
    
    house = Data(edge_index=house_edge_index, x=torch.arange(0, 5, dtype=torch.float).view(5, 1), y=torch.tensor([1]),
                 edge_attr=edge_attr)
    house.num_nodes = house_edge_index.max().item() + 1

    house_complex = compute_ring_2complex(house.x, house.edge_index, house.edge_attr, house.num_nodes,
                                          max_k=4, y=house.y, init_rings=False)

    # Check the number of cells
    assert house_complex.nodes.num_cells_down is None
    assert house_complex.nodes.num_cells_up == 6
    assert house_complex.nodes.boundary_index is None
    assert house_complex.edges.num_cells_down == 5
    assert house_complex.edges.num_cells_up == 2
    assert list(house_complex.edges.boundary_index.size()) == [2, 2*6]
    assert house_complex.cochains[2].num_cells == 2
    assert house_complex.cochains[2].num_cells_down == 6
    assert house_complex.cochains[2].num_cells_up == 0
    assert list(house_complex.cochains[2].boundary_index.size()) == [2, 3+4]

    # Check the returned parameters
    v_params = house_complex.get_cochain_params(dim=0)
    assert torch.equal(v_params.x, house.x)
    assert v_params.down_index is None

    expected_v_up_index = torch.tensor([[0, 1, 0, 3, 1, 2, 2, 3, 2, 4, 3, 4],
                                        [1, 0, 3, 0, 2, 1, 3, 2, 4, 2, 4, 3]], dtype=torch.long)
    assert torch.equal(v_params.up_index, expected_v_up_index)

    expected_v_up_attr = torch.tensor([[0.0, 1.0], [0.0, 1.0], [0.0, 3.0], [0.0, 3.0],
                                       [1.0, 2.0], [1.0, 2.0], [2.0, 3.0], [2.0, 3.0],
                                       [2.0, 4.0], [2.0, 4.0], [3.0, 4.0], [3.0, 4.0]], dtype=torch.float)
    assert torch.equal(v_params.kwargs['up_attr'], expected_v_up_attr)
    assert v_params.kwargs['down_attr'] is None
    assert v_params.kwargs['boundary_attr'] is None

    e_params = house_complex.get_cochain_params(dim=1)
    expected_e_x = torch.FloatTensor(
                                [[0.0, 1.0],
                                 [0.0, 3.0],
                                 [1.0, 2.0],
                                 [2.0, 3.0],
                                 [2.0, 4.0],
                                 [3.0, 4.0]])
    assert torch.equal(e_params.x, expected_e_x)

    expected_e_up_index = torch.tensor([[0, 1, 0, 2, 0, 3, 1, 2, 1, 3, 2, 3, 3, 4, 3, 5, 4, 5],
                                        [1, 0, 2, 0, 3, 0, 2, 1, 3, 1, 3, 2, 4, 3, 5, 3, 5, 4]], dtype=torch.long)
    assert torch.equal(e_params.up_index, expected_e_up_index)
    assert e_params.kwargs['up_attr'] is None

    expected_e_down_index = torch.tensor([[0, 1, 0, 2, 2, 3, 2, 4, 3, 4, 1, 3, 1, 5, 3, 5, 4, 5],
                                          [1, 0, 2, 0, 3, 2, 4, 2, 4, 3, 3, 1, 5, 1, 5, 3, 5, 4]],
                                         dtype=torch.long)
    assert torch.equal(e_params.down_index, expected_e_down_index)
    expected_e_down_attr = torch.tensor([[0], [0], [1], [1], [2], [2], [2], [2], [2], [2],
                                         [3], [3], [3], [3], [3], [3], [4], [4]],
                                        dtype=torch.float)
    assert torch.equal(e_params.kwargs['down_attr'], expected_e_down_attr)

    assert torch.equal(e_params.kwargs['boundary_attr'], house.x)
    assert list(e_params.kwargs['boundary_index'].size()) == [2, 2*house_complex.edges.num_cells]
    assert torch.equal(e_params.kwargs['boundary_index'][1], torch.LongTensor([0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5]))
    assert torch.equal(e_params.kwargs['boundary_index'][0], torch.LongTensor([0, 1, 0, 3, 1, 2, 2, 3, 2, 4, 3, 4]))

    t_params = house_complex.get_cochain_params(dim=2)
    assert t_params.x is None
    expected_t_down_index = torch.tensor([[0, 1],
                                          [1, 0]],
                                         dtype=torch.long)
    assert torch.equal(t_params.down_index, expected_t_down_index)
    expected_t_down_attr = torch.tensor([[2.0, 3.0], [2.0, 3.0]], dtype=torch.float)
    assert torch.equal(t_params.kwargs['down_attr'], expected_t_down_attr)
    
    assert t_params.up_index is None
    assert torch.equal(t_params.kwargs['boundary_attr'], expected_e_x)
    expected_t_boundary_index = torch.tensor([[0, 1, 2, 3, 3, 4, 5],
                                          [0, 0, 0, 0, 1, 1, 1]], dtype=torch.long)
    assert torch.equal(t_params.kwargs['boundary_index'], expected_t_boundary_index)
    assert torch.equal(house_complex.y, house.y)


def test_construction_of_ring_2complex_with_larger_k_size(house_edge_index):
    
    # Here we check that the max ring size does not have any effect when it is larger
    # then the largest ring present in the original graph

    house = Data(edge_index=house_edge_index, x=torch.arange(0, 5, dtype=torch.float).view(5, 1), y=torch.tensor([1]))
    house.num_nodes = house_edge_index.max().item() + 1

    house_cell_a = compute_ring_2complex(house.x, house.edge_index, None, house.num_nodes,
                                         max_k=4, y=house.y, init_rings=True)
    house_cell_b = compute_ring_2complex(house.x, house.edge_index, None, house.num_nodes,
                                         max_k=10, y=house.y, init_rings=True)

    # Check the number of cells
    assert house_cell_a.nodes.num_cells_down is None
    assert house_cell_b.nodes.num_cells_down is None
    assert house_cell_a.nodes.num_cells_up == house_cell_b.nodes.num_cells_up
    assert house_cell_a.nodes.boundary_index is None
    assert house_cell_b.nodes.boundary_index is None
    assert house_cell_a.edges.num_cells_down == house_cell_b.edges.num_cells_down
    assert house_cell_a.edges.num_cells_up == house_cell_b.edges.num_cells_up
    assert list(house_cell_a.edges.boundary_index.size()) == list(house_cell_b.edges.boundary_index.size())
    assert house_cell_a.two_cells.num_cells == 2  # We have 2 rings in the house complex
    assert house_cell_a.two_cells.num_cells == house_cell_b.two_cells.num_cells
    assert house_cell_a.two_cells.num_cells_down == house_cell_b.two_cells.num_cells_down
    assert house_cell_a.two_cells.num_cells_up == house_cell_b.two_cells.num_cells_up
    assert list(house_cell_a.two_cells.boundary_index.size()) == list(house_cell_b.two_cells.boundary_index.size())

    # Check the returned node parameters
    v_params_a = house_cell_a.get_cochain_params(dim=0)
    v_params_b = house_cell_b.get_cochain_params(dim=0)
    assert torch.equal(v_params_a.x, v_params_b.x)
    assert v_params_a.down_index is None
    assert v_params_b.down_index is None
    assert torch.equal(v_params_a.up_index, v_params_b.up_index)
    assert torch.equal(v_params_a.kwargs['up_attr'], v_params_b.kwargs['up_attr'])
    assert v_params_a.kwargs['down_attr'] is None
    assert v_params_b.kwargs['down_attr'] is None
    assert v_params_a.kwargs['boundary_attr'] is None
    assert v_params_b.kwargs['boundary_attr'] is None

    # Check the returned edge parameters
    e_params_a = house_cell_a.get_cochain_params(dim=1)
    e_params_b = house_cell_b.get_cochain_params(dim=1)
    assert torch.equal(e_params_a.x, e_params_b.x)
    assert torch.equal(e_params_a.up_index, e_params_b.up_index)
    assert torch.equal(e_params_a.kwargs['up_attr'], e_params_b.kwargs['up_attr'])
    assert torch.equal(e_params_a.down_index, e_params_b.down_index)
    assert torch.equal(e_params_a.kwargs['down_attr'], e_params_b.kwargs['down_attr'])
    assert torch.equal(e_params_a.kwargs['boundary_attr'], e_params_b.kwargs['boundary_attr'])
    assert list(e_params_a.kwargs['boundary_index'].size()) == list(e_params_b.kwargs['boundary_index'].size())
    assert torch.equal(e_params_a.kwargs['boundary_index'][1], e_params_b.kwargs['boundary_index'][1])
    assert torch.equal(e_params_a.kwargs['boundary_index'][0], e_params_b.kwargs['boundary_index'][0])

    # Check the returned ring parameters
    t_params_a = house_cell_a.get_cochain_params(dim=2)
    t_params_b = house_cell_b.get_cochain_params(dim=2)
    assert t_params_a.x.size(0) == 2
    assert torch.equal(t_params_a.x, t_params_b.x)
    assert torch.equal(t_params_a.down_index, t_params_b.down_index)
    assert torch.equal(t_params_a.kwargs['down_attr'], t_params_b.kwargs['down_attr'])
    assert t_params_a.up_index is None
    assert t_params_b.up_index is None
    assert t_params_a.kwargs['up_attr'] is None
    assert t_params_b.kwargs['up_attr'] is None
    assert torch.equal(t_params_a.kwargs['boundary_attr'], t_params_b.kwargs['boundary_attr'])
    assert torch.equal(t_params_a.kwargs['boundary_index'], t_params_b.kwargs['boundary_index'])
    
    # Check label
    assert torch.equal(house_cell_a.y, house_cell_b.y)


def test_construction_of_ring_2complex_with_smaller_k_size(house_edge_index):

    # Here we check that when we consider rings up to length 3, then the output cell complex
    # exactly corresponds to a 2-simplicial complex extracted with the alternative routine

    house = Data(edge_index=house_edge_index, x=torch.arange(0, 5, dtype=torch.float).view(5, 1), y=torch.tensor([1]))
    house.num_nodes = house_edge_index.max().item() + 1

    house_cell = compute_ring_2complex(house.x, house.edge_index, None, house.num_nodes,
                                       max_k=3, y=house.y, init_rings=True)
    house_simp = compute_clique_complex_with_gudhi(house.x, house.edge_index, house.num_nodes,
                                                   y=house.y)

    # Check the number of cells
    assert house_cell.nodes.num_cells_down is None
    assert house_simp.nodes.num_cells_down is None
    assert house_cell.nodes.num_cells_up == house_simp.nodes.num_cells_up
    assert house_cell.nodes.boundary_index is None
    assert house_simp.nodes.boundary_index is None
    assert house_cell.edges.num_cells_down == house_simp.edges.num_cells_down
    assert house_cell.edges.num_cells_up == house_simp.edges.num_cells_up
    assert list(house_cell.edges.boundary_index.size()) == list(house_simp.edges.boundary_index.size())
    assert house_cell.two_cells.num_cells == 1
    assert house_cell.two_cells.num_cells == house_simp.two_cells.num_cells
    assert house_cell.two_cells.num_cells_down == house_simp.two_cells.num_cells_down
    assert house_cell.two_cells.num_cells_up == house_simp.two_cells.num_cells_up
    assert list(house_cell.two_cells.boundary_index.size()) == list(house_simp.two_cells.boundary_index.size())

    # Check the returned node parameters
    v_params_a = house_cell.get_cochain_params(dim=0)
    v_params_b = house_simp.get_cochain_params(dim=0)
    assert torch.equal(v_params_a.x, v_params_b.x)
    assert v_params_a.down_index is None
    assert v_params_b.down_index is None
    assert torch.equal(v_params_a.up_index, v_params_b.up_index)
    assert torch.equal(v_params_a.kwargs['up_attr'], v_params_b.kwargs['up_attr'])
    assert v_params_a.kwargs['down_attr'] is None
    assert v_params_b.kwargs['down_attr'] is None
    assert v_params_a.kwargs['boundary_attr'] is None
    assert v_params_b.kwargs['boundary_attr'] is None

    # Check the returned edge parameters
    e_params_a = house_cell.get_cochain_params(dim=1)
    e_params_b = house_simp.get_cochain_params(dim=1)
    assert torch.equal(e_params_a.x, e_params_b.x)
    assert torch.equal(e_params_a.up_index, e_params_b.up_index)
    assert torch.equal(e_params_a.kwargs['up_attr'], e_params_b.kwargs['up_attr'])
    assert torch.equal(e_params_a.down_index, e_params_b.down_index)
    assert torch.equal(e_params_a.kwargs['down_attr'], e_params_b.kwargs['down_attr'])
    assert torch.equal(e_params_a.kwargs['boundary_attr'], e_params_b.kwargs['boundary_attr'])
    assert list(e_params_a.kwargs['boundary_index'].size()) == list(e_params_b.kwargs['boundary_index'].size())
    assert torch.equal(e_params_a.kwargs['boundary_index'][1], e_params_b.kwargs['boundary_index'][1])
    assert torch.equal(e_params_a.kwargs['boundary_index'][0], e_params_b.kwargs['boundary_index'][0])

    # Check the returned ring parameters
    t_params_a = house_cell.get_cochain_params(dim=2)
    t_params_b = house_simp.get_cochain_params(dim=2)
    assert t_params_a.x.size(0) == 1
    assert torch.equal(t_params_a.x, t_params_b.x)
    assert t_params_a.down_index is None
    assert t_params_b.down_index is None
    assert t_params_a.kwargs['down_attr'] is None
    assert t_params_b.kwargs['down_attr'] is None
    assert t_params_a.up_index is None
    assert t_params_b.up_index is None
    assert t_params_a.kwargs['up_attr'] is None
    assert t_params_b.kwargs['up_attr'] is None
    assert torch.equal(t_params_a.kwargs['boundary_attr'], t_params_b.kwargs['boundary_attr'])
    assert torch.equal(t_params_a.kwargs['boundary_index'], t_params_b.kwargs['boundary_index'])
    
    # Check label
    assert torch.equal(house_cell.y, house_simp.y)

    
def test_ring_2complex_dataset_conversion(house_edge_index):
    house1 = Data(edge_index=house_edge_index, x=torch.arange(0, 5, dtype=torch.float).view(5, 1), y=torch.tensor([1]))
    house2 = Data(edge_index=house_edge_index, x=torch.arange(0, 5, dtype=torch.float).view(5, 1), y=torch.tensor([1]))
    house3 = Data(edge_index=house_edge_index, x=torch.arange(0, 5, dtype=torch.float).view(5, 1), y=torch.tensor([1]))
    dataset = [house1, house2, house3]
    complexes, dim, num_features = convert_graph_dataset_with_rings(dataset, include_down_adj=True,
        init_rings=True)
    assert dim == 2
    assert len(num_features) == 3
    for i in range(len(num_features)):
        assert num_features[i] == 1
    assert len(complexes) == 3
    for i in range(len(complexes)):
        # Do some basic checks for each complex.
        # Checks the number of rings in `boundary_index`
        assert complexes[i].cochains[2].boundary_index[:, 1].max().item() == 1
        assert complexes[i].dimension == 2
        assert complexes[i].nodes.boundary_index is None
        assert list(complexes[i].edges.boundary_index.size()) == [2, 2*6]
        assert list(complexes[i].two_cells.boundary_index.size()) == [2, 3+4]
        assert complexes[i].edges.lower_index.size(1) == 18
        assert torch.equal(complexes[i].nodes.x, house1.x)
        assert torch.equal(complexes[i].y, house1.y)
        
        
def test_ring_2complex_dataset_conversion_with_edge_feats(house_edge_index):    
    edge_attr = torch.FloatTensor(
                        [[0.0, 1.0],
                         [0.0, 3.0],
                         [0.0, 1.0],
                         [1.0, 2.0],
                         [1.0, 2.0],
                         [2.0, 3.0],
                         [2.0, 4.0],
                         [0.0, 3.0],
                         [2.0, 3.0],
                         [3.0, 4.0],
                         [2.0, 4.0],
                         [3.0, 4.0]])
    e_x = torch.FloatTensor(
                        [[0.0, 1.0],
                         [0.0, 3.0],
                         [1.0, 2.0],
                         [2.0, 3.0],
                         [2.0, 4.0],
                         [3.0, 4.0]])
    house1 = Data(edge_index=house_edge_index, x=torch.arange(0, 5, dtype=torch.float).view(5, 1), edge_attr=edge_attr, y=torch.tensor([1]))
    house2 = Data(edge_index=house_edge_index, x=torch.arange(0, 5, dtype=torch.float).view(5, 1), edge_attr=edge_attr, y=torch.tensor([1]))
    house3 = Data(edge_index=house_edge_index, x=torch.arange(0, 5, dtype=torch.float).view(5, 1), edge_attr=edge_attr, y=torch.tensor([1]))
    dataset = [house1, house2, house3]
    complexes, dim, num_features = convert_graph_dataset_with_rings(dataset,
        init_edges=True, include_down_adj=True, init_rings=False)
    assert dim == 2
    assert len(num_features) == 3
    assert num_features[0] == 1
    assert num_features[1] == 2
    assert num_features[2] == 0
    assert len(complexes) == 3
    for i in range(len(complexes)):
        # Do some basic checks for each complex.
        assert complexes[i].dimension == 2
        assert complexes[i].nodes.boundary_index is None
        assert list(complexes[i].edges.boundary_index.size()) == [2, 2*6]
        assert list(complexes[i].two_cells.boundary_index.size()) == [2, 3+4]
        assert complexes[i].edges.lower_index.size(1) == 18
        assert torch.equal(complexes[i].nodes.x, house1.x)
        assert torch.equal(complexes[i].edges.x, e_x)
        assert complexes[i].two_cells.x is None
        assert torch.equal(complexes[i].y, house1.y)

        
def test_simp_complex_conversion_completes():
    graphs = list(map(convert_to_graph, get_testing_complex_list()))
    _ = convert_graph_dataset_with_gudhi(graphs, expansion_dim=3)


def test_cell_complex_conversion_completes():
    graphs = list(map(convert_to_graph, get_testing_complex_list()))
    _ = convert_graph_dataset_with_rings(graphs, init_rings=True)
