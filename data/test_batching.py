import torch
import pytest
import itertools

from data.dummy_complexes import (get_house_complex, get_square_complex, get_pyramid_complex,
    get_square_dot_complex, get_kite_complex)

from data.complex import ComplexBatch
from data.dummy_complexes import get_testing_complex_list
from data.data_loading import DataLoader, load_dataset


def validate_double_house(batch):
    
    expected_node_upper = torch.tensor([[0, 1, 0, 3, 1, 2, 2, 3, 2, 4, 3, 4, 5, 6, 5, 8, 6, 7, 7, 8, 7, 9, 8, 9],
                                        [1, 0, 3, 0, 2, 1, 3, 2, 4, 2, 4, 3, 6, 5, 8, 5, 7, 6, 8, 7, 9, 7, 9, 8]], dtype=torch.long)
    expected_node_shared_coboundaries = torch.tensor([0, 0, 3, 3, 1, 1, 2, 2, 5, 5, 4, 4, 6, 6, 9, 9, 7, 7, 8, 8, 11, 11, 10, 10], dtype=torch.long)
    expected_node_x = torch.tensor([[1], [2], [3], [4], [5], [1], [2], [3], [4], [5]], dtype=torch.float)
    expected_node_y = torch.tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=torch.long)
    expected_node_batch = torch.tensor([0, 0, 0, 0, 0, 1, 1, 1, 1, 1], dtype=torch.long)
    
    expected_edge_upper = torch.tensor([[2, 4, 2, 5, 4, 5, 8, 10, 8, 11, 10, 11],
                                        [4, 2, 5, 2, 5, 4, 10, 8, 11, 8, 11, 10]], dtype=torch.long)
    expected_edge_shared_coboundaries = torch.tensor([0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1], dtype=torch.long)
    expected_edge_lower = torch.tensor([[0, 1, 0, 3, 1, 2, 1, 5, 2, 3, 2, 4, 2, 5, 3, 4, 4, 5, 6, 7, 6, 9, 7, 8, 7, 11, 8, 9, 8, 10, 8, 11, 9, 10, 10, 11],
                                 [1, 0, 3, 0, 2, 1, 5, 1, 3, 2, 4, 2, 5, 2, 4, 3, 5, 4, 7, 6, 9, 6, 8, 7, 11, 7, 9, 8, 10, 8, 11, 8, 10, 9, 11, 10]],
                                dtype=torch.long)
    expected_edge_shared_boundaries = torch.tensor([1, 1, 0, 0, 2, 2, 2, 2, 3, 3, 3, 3, 2, 2, 3, 3, 4, 4, 6, 6, 5, 5, 7, 7, 7, 7, 8, 8, 8, 8, 7, 7, 8, 8, 9, 9],
                                  dtype=torch.long)
    expected_edge_x = torch.tensor([[1], [2], [3], [4], [5], [6], [1], [2], [3], [4], [5], [6]], dtype=torch.float)
    expected_edge_y = torch.tensor([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], dtype=torch.long)
    expected_edge_batch = torch.tensor([0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1], dtype=torch.long)
    
    expected_two_cell_x = torch.tensor([[1], [1]], dtype=torch.float)
    expected_two_cell_y = torch.tensor([2, 2], dtype=torch.long)
    expected_two_cell_batch = torch.tensor([0, 1], dtype=torch.long)
    
    assert torch.equal(expected_node_upper, batch.nodes.upper_index)
    assert torch.equal(expected_node_shared_coboundaries, batch.nodes.shared_coboundaries)
    assert batch.nodes.lower_index is None
    assert batch.nodes.shared_boundaries is None
    assert torch.equal(expected_node_x, batch.nodes.x)
    assert torch.equal(expected_node_y, batch.nodes.y)
    assert torch.equal(expected_node_batch, batch.nodes.batch)
    
    assert torch.equal(expected_edge_upper, batch.edges.upper_index)
    assert torch.equal(expected_edge_shared_coboundaries, batch.edges.shared_coboundaries)
    assert torch.equal(expected_edge_lower, batch.edges.lower_index)
    assert torch.equal(expected_edge_shared_boundaries, batch.edges.shared_boundaries)
    assert torch.equal(expected_edge_x, batch.edges.x)
    assert torch.equal(expected_edge_y, batch.edges.y)
    assert torch.equal(expected_edge_batch, batch.edges.batch)
    
    assert batch.two_cells.upper_index is None
    assert batch.two_cells.lower_index is None
    assert batch.two_cells.shared_coboundaries is None
    assert batch.two_cells.shared_boundaries is None
    assert torch.equal(expected_two_cell_x, batch.two_cells.x)
    assert torch.equal(expected_two_cell_y, batch.two_cells.y)
    assert torch.equal(expected_two_cell_batch, batch.two_cells.batch)
    

def validate_square_dot_and_square(batch):
    
    expected_node_upper = torch.tensor([        [0, 1, 0, 3, 1, 2, 2, 3, 5, 6, 5, 8, 6, 7, 7, 8],
                                                [1, 0, 3, 0, 2, 1, 3, 2, 6, 5, 8, 5, 7, 6, 8, 7]], dtype=torch.long)
    expected_node_shared_coboundaries = torch.tensor([0, 0, 3, 3, 1, 1, 2, 2, 4, 4, 7, 7, 5, 5, 6, 6], dtype=torch.long)
    expected_node_x = torch.tensor([[1], [2], [3], [4], [5], [1], [2], [3], [4]], dtype=torch.float)
    expected_node_y = torch.tensor([0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=torch.long)
    expected_node_batch = torch.tensor([0, 0, 0, 0, 0, 1, 1, 1, 1], dtype=torch.long)
    
    expected_edge_lower = torch.tensor([      [0, 1, 0, 3, 1, 2, 2, 3, 4, 5, 4, 7, 5, 6, 6, 7],
                                              [1, 0, 3, 0, 2, 1, 3, 2, 5, 4, 7, 4, 6, 5, 7, 6]], dtype=torch.long)
    expected_edge_shared_boundaries = torch.tensor([1, 1, 0, 0, 2, 2, 3, 3, 6, 6, 5, 5, 7, 7, 8, 8],
                                  dtype=torch.long)
    expected_edge_x = torch.tensor([[1], [2], [3], [4], [1], [2], [3], [4]], dtype=torch.float)
    expected_edge_y = torch.tensor([1, 1, 1, 1, 1, 1, 1, 1,], dtype=torch.long)
    expected_edge_batch = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1], dtype=torch.long)
    
    assert torch.equal(expected_node_upper, batch.nodes.upper_index)
    assert torch.equal(expected_node_shared_coboundaries, batch.nodes.shared_coboundaries)
    assert batch.nodes.lower_index is None
    assert batch.nodes.shared_boundaries is None
    assert torch.equal(expected_node_x, batch.nodes.x)
    assert torch.equal(expected_node_y, batch.nodes.y)
    assert torch.equal(expected_node_batch, batch.nodes.batch)
    
    assert batch.edges.upper_index is None
    assert batch.edges.shared_coboundaries is None
    assert torch.equal(expected_edge_lower, batch.edges.lower_index)
    assert torch.equal(expected_edge_shared_boundaries, batch.edges.shared_boundaries)
    assert torch.equal(expected_edge_x, batch.edges.x)
    assert torch.equal(expected_edge_y, batch.edges.y)
    assert torch.equal(expected_edge_batch, batch.edges.batch)
    

def validate_kite_and_house(batch):

    kite_node_upper = torch.tensor([[0, 1, 0, 2, 1, 2, 1, 3, 2, 3, 3, 4],
                                    [1, 0, 2, 0, 2, 1, 3, 1, 3, 2, 4, 3]], dtype=torch.long)
    shifted_house_node_upper = 5 + torch.tensor([[0, 1, 0, 3, 1, 2, 2, 3, 2, 4, 3, 4],
                                        [1, 0, 3, 0, 2, 1, 3, 2, 4, 2, 4, 3]], dtype=torch.long)
    expected_node_upper = torch.cat([kite_node_upper, shifted_house_node_upper], 1)
    
    kite_node_shared_coboundaries = torch.tensor([0, 0, 2, 2, 1, 1, 3, 3, 4, 4, 5, 5], dtype=torch.long)
    shifted_house_node_shared_coboundaries = 6 + torch.tensor([0, 0, 3, 3, 1, 1, 2, 2, 5, 5, 4, 4], dtype=torch.long)
    expected_node_shared_coboundaries = torch.cat([kite_node_shared_coboundaries, shifted_house_node_shared_coboundaries], 0)
    
    expected_node_x = torch.tensor([[1], [2], [3], [4], [5], [1], [2], [3], [4], [5]], dtype=torch.float)
    expected_node_y = torch.tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=torch.long)
    expected_node_batch = torch.tensor([0, 0, 0, 0, 0, 1, 1, 1, 1, 1], dtype=torch.long)
    
    kite_edge_upper = torch.tensor([[0, 1, 0, 2, 1, 2, 1, 3, 1, 4, 3, 4],
                                    [1, 0, 2, 0, 2, 1, 3, 1, 4, 1, 4, 3]], dtype=torch.long)
    shifted_house_edge_upper = 6 + torch.tensor([[2, 4, 2, 5, 4, 5],
                                                 [4, 2, 5, 2, 5, 4]], dtype=torch.long)
    expected_edge_upper = torch.cat([kite_edge_upper, shifted_house_edge_upper], 1)
    
    kite_edge_shared_coboundaries = torch.tensor([0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1], dtype=torch.long)
    shifted_house_edge_shared_coboundaries = 2 + torch.tensor([0, 0, 0, 0, 0, 0], dtype=torch.long)
    expected_edge_shared_coboundaries = torch.cat([kite_edge_shared_coboundaries, shifted_house_edge_shared_coboundaries], 0)
    
    kite_edge_lower = torch.tensor([ [0, 1, 0, 3, 1, 3, 0, 2, 1, 2, 2, 4, 1, 4, 3, 4, 3, 5, 4, 5],
                                     [1, 0, 3, 0, 3, 1, 2, 0, 2, 1, 4, 2, 4, 1, 4, 3, 5, 3, 5, 4]], dtype=torch.long)
    shifted_house_lower = 6 + torch.tensor([[0, 1, 0, 3, 1, 2, 1, 5, 2, 3, 2, 4, 2, 5, 3, 4, 4, 5],
                                            [1, 0, 3, 0, 2, 1, 5, 1, 3, 2, 4, 2, 5, 2, 4, 3, 5, 4]], dtype=torch.long)
    expected_edge_lower = torch.cat([kite_edge_lower, shifted_house_lower], 1)
    
    kite_edge_shared_boundaries = torch.tensor([1, 1, 1, 1, 1, 1, 0, 0, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3], dtype=torch.long)
    shifted_house_edge_shared_boundaries = 5 + torch.tensor([1, 1, 0, 0, 2, 2, 2, 2, 3, 3, 3, 3, 2, 2, 3, 3, 4, 4], dtype=torch.long)
    expected_edge_shared_boundaries = torch.cat([kite_edge_shared_boundaries, shifted_house_edge_shared_boundaries], 0)
    
    expected_edge_x = torch.tensor([[1], [2], [3], [4], [5], [6], [1], [2], [3], [4], [5], [6]], dtype=torch.float)
    expected_edge_y = torch.tensor([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], dtype=torch.long)
    expected_edge_batch = torch.tensor([0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1], dtype=torch.long)
    
    expected_two_cell_lower = torch.tensor([[0, 1],
                                            [1, 0]], dtype=torch.long)
    expected_two_cell_shared_boundaries = torch.tensor([1, 1], dtype=torch.long)
    expected_two_cell_x = torch.tensor([[1], [2], [1]], dtype=torch.float)
    expected_two_cell_y = torch.tensor([2, 2, 2], dtype=torch.long)
    expected_two_cell_batch = torch.tensor([0, 0, 1], dtype=torch.long)
    
    assert torch.equal(expected_node_upper, batch.nodes.upper_index)
    assert torch.equal(expected_node_shared_coboundaries, batch.nodes.shared_coboundaries)
    assert batch.nodes.lower_index is None
    assert batch.nodes.shared_boundaries is None
    assert torch.equal(expected_node_x, batch.nodes.x)
    assert torch.equal(expected_node_y, batch.nodes.y)
    assert torch.equal(expected_node_batch, batch.nodes.batch)
    
    assert torch.equal(expected_edge_upper, batch.edges.upper_index)
    assert torch.equal(expected_edge_shared_coboundaries, batch.edges.shared_coboundaries)
    assert torch.equal(expected_edge_lower, batch.edges.lower_index)
    assert torch.equal(expected_edge_shared_boundaries, batch.edges.shared_boundaries)
    assert torch.equal(expected_edge_x, batch.edges.x)
    assert torch.equal(expected_edge_y, batch.edges.y)
    assert torch.equal(expected_edge_batch, batch.edges.batch)
    
    assert batch.two_cells.upper_index is None
    assert batch.two_cells.shared_coboundaries is None
    assert torch.equal(expected_two_cell_lower, batch.two_cells.lower_index)
    assert torch.equal(expected_two_cell_shared_boundaries, batch.two_cells.shared_boundaries)
    assert torch.equal(expected_two_cell_x, batch.two_cells.x)
    assert torch.equal(expected_two_cell_y, batch.two_cells.y)
    assert torch.equal(expected_two_cell_batch, batch.two_cells.batch)
    
    
def validate_house_and_square(batch):
    
    expected_node_upper = torch.tensor([[0, 1, 0, 3, 1, 2, 2, 3, 2, 4, 3, 4, 5, 6, 5, 8, 6, 7, 7, 8],
                                        [1, 0, 3, 0, 2, 1, 3, 2, 4, 2, 4, 3, 6, 5, 8, 5, 7, 6, 8, 7]], dtype=torch.long)
    expected_node_shared_coboundaries = torch.tensor([0, 0, 3, 3, 1, 1, 2, 2, 5, 5, 4, 4, 6, 6, 9, 9, 7, 7, 8, 8], dtype=torch.long)
    expected_node_x = torch.tensor([[1], [2], [3], [4], [5], [1], [2], [3], [4]], dtype=torch.float)
    expected_node_y = torch.tensor([0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=torch.long)
    expected_node_batch = torch.tensor([0, 0, 0, 0, 0, 1, 1, 1, 1], dtype=torch.long)
    
    expected_edge_upper = torch.tensor([[2, 4, 2, 5, 4, 5],
                                        [4, 2, 5, 2, 5, 4]], dtype=torch.long)
    expected_edge_shared_coboundaries = torch.tensor([0, 0, 0, 0, 0, 0], dtype=torch.long)
    expected_edge_lower = torch.tensor([      [0, 1, 0, 3, 1, 2, 1, 5, 2, 3, 2, 4, 2, 5, 3, 4, 4, 5, 6, 7, 6, 9, 7, 8, 8, 9],
                                              [1, 0, 3, 0, 2, 1, 5, 1, 3, 2, 4, 2, 5, 2, 4, 3, 5, 4, 7, 6, 9, 6, 8, 7, 9, 8]],
                                dtype=torch.long)
    expected_edge_shared_boundaries = torch.tensor([1, 1, 0, 0, 2, 2, 2, 2, 3, 3, 3, 3, 2, 2, 3, 3, 4, 4, 6, 6, 5, 5, 7, 7, 8, 8],
                                  dtype=torch.long)
    expected_edge_x = torch.tensor([[1], [2], [3], [4], [5], [6], [1], [2], [3], [4]], dtype=torch.float)
    expected_edge_y = torch.tensor([1, 1, 1, 1, 1, 1, 1, 1, 1, 1,], dtype=torch.long)
    expected_edge_batch = torch.tensor([0, 0, 0, 0, 0, 0, 1, 1, 1, 1], dtype=torch.long)
    
    expected_two_cell_x = torch.tensor([[1]], dtype=torch.float)
    expected_two_cell_y = torch.tensor([2], dtype=torch.long)
    expected_two_cell_batch = torch.tensor([0], dtype=torch.long)
    
    assert torch.equal(expected_node_upper, batch.nodes.upper_index)
    assert torch.equal(expected_node_shared_coboundaries, batch.nodes.shared_coboundaries)
    assert batch.nodes.lower_index is None
    assert batch.nodes.shared_boundaries is None
    assert torch.equal(expected_node_x, batch.nodes.x)
    assert torch.equal(expected_node_y, batch.nodes.y)
    assert torch.equal(expected_node_batch, batch.nodes.batch)
    
    assert torch.equal(expected_edge_upper, batch.edges.upper_index)
    assert torch.equal(expected_edge_shared_coboundaries, batch.edges.shared_coboundaries)
    assert torch.equal(expected_edge_lower, batch.edges.lower_index)
    assert torch.equal(expected_edge_shared_boundaries, batch.edges.shared_boundaries)
    assert torch.equal(expected_edge_x, batch.edges.x)
    assert torch.equal(expected_edge_y, batch.edges.y)
    assert torch.equal(expected_edge_batch, batch.edges.batch)
    
    assert batch.two_cells.upper_index is None
    assert batch.two_cells.lower_index is None
    assert batch.two_cells.shared_coboundaries is None
    assert batch.two_cells.shared_boundaries is None
    assert torch.equal(expected_two_cell_x, batch.two_cells.x)
    assert torch.equal(expected_two_cell_y, batch.two_cells.y)
    assert torch.equal(expected_two_cell_batch, batch.two_cells.batch)
    
    
def validate_house_square_house(batch):
    
    expected_node_upper = torch.tensor([[0, 1, 0, 3, 1, 2, 2, 3, 2, 4, 3, 4, 5, 6, 5, 8, 6, 7, 7, 8, 9, 10, 9, 12, 10, 11, 11, 12, 11, 13, 12, 13],
                                        [1, 0, 3, 0, 2, 1, 3, 2, 4, 2, 4, 3, 6, 5, 8, 5, 7, 6, 8, 7, 10, 9, 12, 9, 11, 10, 12, 11, 13, 11, 13, 12]],
                                       dtype=torch.long)
    expected_node_shared_coboundaries = torch.tensor([0, 0, 3, 3, 1, 1, 2, 2, 5, 5, 4, 4, 6, 6, 9, 9, 7, 7, 8, 8, 10, 10, 13, 13, 11, 11, 12, 12, 15, 15, 14, 14],
                                                dtype=torch.long)
    expected_node_x = torch.tensor([[1], [2], [3], [4], [5], [1], [2], [3], [4], [1], [2], [3], [4], [5]], dtype=torch.float)
    expected_node_y = torch.tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=torch.long)
    expected_node_batch = torch.tensor([0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 2], dtype=torch.long)
    
    expected_edge_upper = torch.tensor([[2, 4, 2, 5, 4, 5, 12, 14, 12, 15, 14, 15],
                                        [4, 2, 5, 2, 5, 4, 14, 12, 15, 12, 15, 14]], dtype=torch.long)
    expected_edge_shared_coboundaries = torch.tensor([0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1], dtype=torch.long)
    expected_edge_lower = torch.tensor([      [0, 1, 0, 3, 1, 2, 1, 5, 2, 3, 2, 4, 2, 5, 3, 4, 4, 5, 6, 7, 6, 9, 7, 8, 8, 9, 10, 11, 10, 13, 11, 12, 11, 15, 12, 13, 12, 14, 12, 15, 13, 14, 14, 15],
                                              [1, 0, 3, 0, 2, 1, 5, 1, 3, 2, 4, 2, 5, 2, 4, 3, 5, 4, 7, 6, 9, 6, 8, 7, 9, 8, 11, 10, 13, 10, 12, 11, 15, 11, 13, 12, 14, 12, 15, 12, 14, 13, 15, 14]],
                                dtype=torch.long)
    expected_edge_shared_boundaries = torch.tensor([1, 1, 0, 0, 2, 2, 2, 2, 3, 3, 3, 3, 2, 2, 3, 3, 4, 4, 6, 6, 5, 5, 7, 7, 8, 8, 10, 10, 9,  9, 11, 11, 11, 11, 12, 12, 12, 12, 11, 11, 12, 12, 13, 13],
                                  dtype=torch.long)
    expected_edge_x = torch.tensor([[1], [2], [3], [4], [5], [6], [1], [2], [3], [4], [1], [2], [3], [4], [5], [6]], dtype=torch.float)
    expected_edge_y = torch.tensor([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], dtype=torch.long)
    expected_edge_batch = torch.tensor([0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2], dtype=torch.long)
    
    expected_two_cell_x = torch.tensor([[1], [1]], dtype=torch.float)
    expected_two_cell_y = torch.tensor([2, 2], dtype=torch.long)
    expected_two_cell_batch = torch.tensor([0, 2], dtype=torch.long)
    
    assert torch.equal(expected_node_upper, batch.nodes.upper_index)
    assert torch.equal(expected_node_shared_coboundaries, batch.nodes.shared_coboundaries)
    assert batch.nodes.lower_index is None
    assert batch.nodes.shared_boundaries is None
    assert torch.equal(expected_node_x, batch.nodes.x)
    assert torch.equal(expected_node_y, batch.nodes.y)
    assert torch.equal(expected_node_batch, batch.nodes.batch)
    
    assert torch.equal(expected_edge_upper, batch.edges.upper_index)
    assert torch.equal(expected_edge_shared_coboundaries, batch.edges.shared_coboundaries)
    assert torch.equal(expected_edge_lower, batch.edges.lower_index)
    assert torch.equal(expected_edge_shared_boundaries, batch.edges.shared_boundaries)
    assert torch.equal(expected_edge_x, batch.edges.x)
    assert torch.equal(expected_edge_y, batch.edges.y)
    assert torch.equal(expected_edge_batch, batch.edges.batch)
    
    assert batch.two_cells.upper_index is None
    assert batch.two_cells.lower_index is None
    assert batch.two_cells.shared_coboundaries is None
    assert batch.two_cells.shared_boundaries is None
    assert torch.equal(expected_two_cell_x, batch.two_cells.x)
    assert torch.equal(expected_two_cell_y, batch.two_cells.y)
    assert torch.equal(expected_two_cell_batch, batch.two_cells.batch)
    
    
def validate_house_no_batching(batch):
        
    expected_node_upper = torch.tensor([[0, 1, 0, 3, 1, 2, 2, 3, 2, 4, 3, 4],
                                        [1, 0, 3, 0, 2, 1, 3, 2, 4, 2, 4, 3]], dtype=torch.long)
    expected_node_shared_coboundaries = torch.tensor([0, 0, 3, 3, 1, 1, 2, 2, 5, 5, 4, 4], dtype=torch.long)
    expected_node_x = torch.tensor([[1], [2], [3], [4], [5]], dtype=torch.float)
    expected_node_y = torch.tensor([0, 0, 0, 0, 0], dtype=torch.long)
    expected_node_batch = torch.tensor([0, 0, 0, 0, 0], dtype=torch.long)
    
    expected_edge_upper = torch.tensor([[2, 4, 2, 5, 4, 5],
                                        [4, 2, 5, 2, 5, 4]], dtype=torch.long)
    expected_edge_shared_coboundaries = torch.tensor([0, 0, 0, 0, 0, 0], dtype=torch.long)
    expected_edge_lower = torch.tensor([[0, 1, 0, 3, 1, 2, 1, 5, 2, 3, 2, 4, 2, 5, 3, 4, 4, 5],
                                        [1, 0, 3, 0, 2, 1, 5, 1, 3, 2, 4, 2, 5, 2, 4, 3, 5, 4]],
                                dtype=torch.long)
    expected_edge_shared_boundaries = torch.tensor([1, 1, 0, 0, 2, 2, 2, 2, 3, 3, 3, 3, 2, 2, 3, 3, 4, 4],
                                  dtype=torch.long)
    expected_edge_x = torch.tensor([[1], [2], [3], [4], [5], [6]], dtype=torch.float)
    expected_edge_y = torch.tensor([1, 1, 1, 1, 1, 1], dtype=torch.long)
    expected_edge_batch = torch.tensor([0, 0, 0, 0, 0, 0], dtype=torch.long)
    
    expected_two_cell_x = torch.tensor([[1]], dtype=torch.float)
    expected_two_cell_y = torch.tensor([2], dtype=torch.long)
    expected_two_cell_batch = torch.tensor([0], dtype=torch.long)
    
    assert torch.equal(expected_node_upper, batch.nodes.upper_index)
    assert torch.equal(expected_node_shared_coboundaries, batch.nodes.shared_coboundaries)
    assert batch.nodes.lower_index is None
    assert batch.nodes.shared_boundaries is None
    assert torch.equal(expected_node_x, batch.nodes.x)
    assert torch.equal(expected_node_y, batch.nodes.y)
    assert torch.equal(expected_node_batch, batch.nodes.batch)
    
    assert torch.equal(expected_edge_upper, batch.edges.upper_index)
    assert torch.equal(expected_edge_shared_coboundaries, batch.edges.shared_coboundaries)
    assert torch.equal(expected_edge_lower, batch.edges.lower_index)
    assert torch.equal(expected_edge_shared_boundaries, batch.edges.shared_boundaries)
    assert torch.equal(expected_edge_x, batch.edges.x)
    assert torch.equal(expected_edge_y, batch.edges.y)
    assert torch.equal(expected_edge_batch, batch.edges.batch)
    
    assert batch.two_cells.upper_index is None
    assert batch.two_cells.lower_index is None
    assert batch.two_cells.shared_coboundaries is None
    assert batch.two_cells.shared_boundaries is None
    assert torch.equal(expected_two_cell_x, batch.two_cells.x)
    assert torch.equal(expected_two_cell_y, batch.two_cells.y)
    assert torch.equal(expected_two_cell_batch, batch.two_cells.batch)
    
    
def test_double_house_batching():
    """
       4         9
      / \       / \
     3---2     8---7
     |   |     |   |
     0---1     5---6

       .         .
      4 5      10 11
     . 2 .     . 8 .
     3   1     9   7
     . 0 .     . 6 .

       .         .
      /0\       /1\
     .---.     .---.
     |   |     |   |
     .---.     .---.
    """
    
    house_1 = get_house_complex()
    house_2 = get_house_complex()
    complex_list = [house_1, house_2]
    batch = ComplexBatch.from_complex_list(complex_list)
    validate_double_house(batch)
    
    
def test_house_and_square_batching():
    """
       4
      / \
     3---2     8---7
     |   |     |   |
     0---1     5---6

       .
      4 5
     . 2 .     . 8 .
     3   1     9   7
     . 0 .     . 6 .

       .
      /0\
     .---.     .---.
     |   |     |   |
     .---.     .---.
    """
    
    house_1 = get_house_complex()
    square = get_square_complex()
    complex_list = [house_1, square]
    batch = ComplexBatch.from_complex_list(complex_list)
    
    validate_house_and_square(batch)
    
    
def test_house_square_house_batching():
    """
       4                   13
      / \                 / \
     3---2     8---7     12--11
     |   |     |   |     |   |
     0---1     5---6     9---10

       .                   .
      4 5                14 15
     . 2 .     . 8 .     . 12.
     3   1     9   7     13  11
     . 0 .     . 6 .     . 10 .

       .                   .
      /0\                 /1\
     .---.     .---.     .---.
     |   |     |   |     |   |
     .---.     .---.     .---.
    """
    
    house_1 = get_house_complex()
    house_2 = get_house_complex()
    square = get_square_complex()
    complex_list = [house_1, square, house_2]
    batch = ComplexBatch.from_complex_list(complex_list)
    
    validate_house_square_house(batch)
    

def test_square_dot_square_batching():
    
    '''
     3---2           8---7
     |   |           |   |
     0---1  4        5---6

     . 2 .           . 6 .
     3   1           7   5
     . 0 .  .        . 4 .

     .---.           .---.
     |   |           |   |
     .---.  .        .---. 
    '''
    square_dot = get_square_dot_complex()
    square = get_square_complex()
    complex_list = [square_dot, square]
    batch = ComplexBatch.from_complex_list(complex_list)
    
    validate_square_dot_and_square(batch)
    
    
def test_kite_house_batching():
    
    '''
    
      2---3---4          9
     / \ /              / \
    0---1              8---7
                       |   |
                       5---6
                       
      . 4 . 5 .          .
     2 1 3             10 11
    . 0 .              . 8 .
                       9   7
                       . 6 .
                       
      .---.---.          .
     /0\1/              /2\
    .---.              .---.
                       |   |
                       .---.
    
    '''
    kite = get_kite_complex()
    house = get_house_complex()
    complex_list = [kite, house]
    batch = ComplexBatch.from_complex_list(complex_list)
    
    validate_kite_and_house(batch)
    

def test_data_loader():
    
    data_list_1 = [
        get_house_complex(),
        get_house_complex(),
        get_house_complex(),
        get_square_complex()]
    
    data_list_2 = [
        get_house_complex(),
        get_square_complex(),
        get_house_complex(),
        get_house_complex()]
    
    data_list_3 = [
        get_house_complex(),
        get_square_complex(),
        get_pyramid_complex(),
        get_pyramid_complex()]
    
    data_list_4 = [
        get_square_dot_complex(),
        get_square_complex(),
        get_kite_complex(),
        get_house_complex(),
        get_house_complex()]
    
    data_loader_1 = DataLoader(data_list_1, batch_size=2)
    data_loader_2 = DataLoader(data_list_2, batch_size=3)
    data_loader_3 = DataLoader(data_list_3, batch_size=3, max_dim=3)
    data_loader_4 = DataLoader(data_list_4, batch_size=2)
    
    
    count = 0
    for batch in data_loader_1:
        count += 1
        if count == 1:
            validate_double_house(batch)
        elif count == 2:
            validate_house_and_square(batch)
    assert count == 2
            
    count = 0
    for batch in data_loader_2:
        count += 1
        if count == 1:
            validate_house_square_house(batch)
        elif count == 2:
            validate_house_no_batching(batch)
    assert count == 2
    
    count = 0
    for batch in data_loader_3:
        count += 1
    assert count == 2
    
    count = 0
    for batch in data_loader_4:
        count += 1
        if count == 1:
            validate_square_dot_and_square(batch)
        elif count == 2:
            validate_kite_and_house(batch)
        else:
            validate_house_no_batching(batch)
    assert count == 3


def test_set_for_features_in_batch():
    house_1 = get_house_complex()
    house_2 = get_house_complex()
    square = get_square_complex()
    complex_list = [house_1, square, house_2]

    vx = torch.arange(21, 35, dtype=torch.float).view(14, 1)
    ex = torch.arange(21, 37, dtype=torch.float).view(16, 1)
    tx = torch.arange(21, 23, dtype=torch.float).view(2, 1)
    xs = [vx, ex, tx]

    batch = ComplexBatch.from_complex_list(complex_list)
    batch.set_xs(xs)

    assert torch.equal(batch.cochains[0].x, vx)
    assert torch.equal(batch.cochains[1].x, ex)
    assert torch.equal(batch.cochains[2].x, tx)


def test_set_xs_does_not_mutate_dataset():
    """Batches should be copied, so these mutations should not change the dataset"""

    data_list = get_testing_complex_list()
    data_loader = DataLoader(data_list, batch_size=5, max_dim=2)

    # Save batch contents
    xs = [[] for _ in range(4)]  # we consider up to dim 3 due to the presence of pyramids
    for batch in data_loader:
        for i in range(batch.dimension + 1):
            xs[i].append(batch.cochains[i].x)
    txs = []
    for i in range(4):
        txs.append(torch.cat(xs[i], dim=0) if len(xs[i]) > 0 else None)

    # Set batch features
    for batch in data_loader:
        new_xs = []
        for i in range(batch.dimension + 1):
            new_xs.append(torch.zeros_like(batch.cochains[i].x))
        batch.set_xs(new_xs)

    # Save batch contents after set_xs
    xs_after = [[] for _ in range(4)]
    for batch in data_loader:
        for i in range(batch.dimension + 1):
            xs_after[i].append(batch.cochains[i].x)
    txs_after = []
    for i in range(4):
        txs_after.append(torch.cat(xs_after[i], dim=0) if len(xs_after[i]) > 0 else None)

    # Check that the batch features are the same
    for i in range(4):
        if txs_after[i] is None:
            assert txs[i] is None
        else:
            assert torch.equal(txs_after[i], txs[i])


def test_batching_returns_the_same_features():
    data_list = get_testing_complex_list()

    # Try multiple parameters
    dims = [1, 2, 3]
    bs = list(range(2, 11))
    params = itertools.product(bs, dims)
    for batch_size, batch_max_dim, in params:
        data_loader = DataLoader(data_list, batch_size=batch_size, max_dim=batch_max_dim)

        batched_x = [[] for _ in range(batch_max_dim+1)]
        for batch in data_loader:
            params = batch.get_all_cochain_params()
            assert len(params) <= batch_max_dim+1
            for dim in range(len(params)):
                batched_x[dim].append(params[dim].x)
        batched_xs = [None for _ in range(batch_max_dim+1)]
        for i in range(batch_max_dim+1):
            if len(batched_x[i]) > 0:
                batched_xs[i] = torch.cat(batched_x[i], dim=0)

        x = [[] for _ in range(batch_max_dim+1)]
        for complex in data_list:
            params = complex.get_all_cochain_params()
            for dim in range(min(len(params), batch_max_dim+1)):
                x[dim].append(params[dim].x)
        xs = [None for _ in range(batch_max_dim+1)]
        for i in range(batch_max_dim+1):
            if len(x[i]) > 0:
                xs[i] = torch.cat(x[i], dim=0)

        for i in range(batch_max_dim+1):
            if xs[i] is None or batched_xs[i] is None:
                assert xs[i] == batched_xs[i]
            else:
                assert torch.equal(batched_xs[i], xs[i])


@pytest.mark.data
def test_batching_returns_the_same_features_on_proteins():
    dataset = load_dataset('PROTEINS', max_dim=3, fold=0, init_method='mean')
    assert len(dataset) == 1113
    split_idx = dataset.get_idx_split()
    dataset = dataset[split_idx['valid']]
    assert len(dataset) == 111

    batch_max_dim = 3
    data_loader = DataLoader(dataset, batch_size=32, max_dim=batch_max_dim)

    batched_x = [[] for _ in range(batch_max_dim+1)]
    for batch in data_loader:
        params = batch.get_all_cochain_params()
        assert len(params) <= batch_max_dim+1
        for dim in range(len(params)):
            batched_x[dim].append(params[dim].x)
    batched_xs = [None for _ in range(batch_max_dim+1)]
    for i in range(batch_max_dim+1):
        if len(batched_x[i]) > 0:
            batched_xs[i] = torch.cat(batched_x[i], dim=0)

    x = [[] for _ in range(batch_max_dim+1)]
    for complex in dataset:
        params = complex.get_all_cochain_params()
        for dim in range(min(len(params), batch_max_dim+1)):
            x[dim].append(params[dim].x)
    xs = [None for _ in range(batch_max_dim+1)]
    for i in range(batch_max_dim+1):
        if len(x[i]) > 0:
            xs[i] = torch.cat(x[i], dim=0)

    for i in range(batch_max_dim+1):
        if xs[i] is None or batched_xs[i] is None:
            assert xs[i] == batched_xs[i]
        else:
            assert torch.equal(batched_xs[i], xs[i])


@pytest.mark.data
def test_batching_returns_the_same_features_on_ring_proteins():
    dataset = load_dataset('PROTEINS', max_dim=2, fold=0, init_method='mean',
                           max_ring_size=7)
    assert len(dataset) == 1113
    assert dataset.max_dim == 2
    split_idx = dataset.get_idx_split()
    dataset = dataset[split_idx['valid']]
    assert len(dataset) == 111

    batch_max_dim = 3
    data_loader = DataLoader(dataset, batch_size=32, max_dim=batch_max_dim)

    batched_x = [[] for _ in range(batch_max_dim+1)]
    for batch in data_loader:
        params = batch.get_all_cochain_params()
        assert len(params) <= batch_max_dim+1
        for dim in range(len(params)):
            batched_x[dim].append(params[dim].x)
    batched_xs = [None for _ in range(batch_max_dim+1)]
    for i in range(batch_max_dim+1):
        if len(batched_x[i]) > 0:
            batched_xs[i] = torch.cat(batched_x[i], dim=0)

    x = [[] for _ in range(batch_max_dim+1)]
    for complex in dataset:
        params = complex.get_all_cochain_params()
        for dim in range(min(len(params), batch_max_dim+1)):
            x[dim].append(params[dim].x)
    xs = [None for _ in range(batch_max_dim+1)]
    for i in range(batch_max_dim+1):
        if len(x[i]) > 0:
            xs[i] = torch.cat(x[i], dim=0)

    for i in range(batch_max_dim+1):
        if xs[i] is None or batched_xs[i] is None:
            assert xs[i] == batched_xs[i]
        else:
            assert torch.equal(batched_xs[i], xs[i])


@pytest.mark.data
def test_batching_returns_the_same_up_attr_on_proteins():
    dataset = load_dataset('PROTEINS', max_dim=3, fold=0, init_method='mean')
    assert len(dataset) == 1113
    split_idx = dataset.get_idx_split()
    dataset = dataset[split_idx['valid']]
    assert len(dataset) == 111

    batch_max_dim = 3
    data_loader = DataLoader(dataset, batch_size=32, max_dim=batch_max_dim)

    # Batched
    batched_x = [[] for _ in range(batch_max_dim+1)]
    for batch in data_loader:
        params = batch.get_all_cochain_params()
        assert len(params) <= batch_max_dim+1
        for dim in range(len(params)):
            if params[dim].kwargs['up_attr'] is not None:
                batched_x[dim].append(params[dim].kwargs['up_attr'])

    batched_xs = [None for _ in range(batch_max_dim+1)]
    for i in range(batch_max_dim+1):
        if len(batched_x[i]) > 0:
            batched_xs[i] = torch.cat(batched_x[i], dim=0)

    # Un-batched
    x = [[] for _ in range(batch_max_dim+1)]
    for complex in dataset:
        params = complex.get_all_cochain_params()
        for dim in range(min(len(params), batch_max_dim+1)):
            # TODO: Modify test after merging the top_feature branch
            # Right now, the last level cannot have top features
            if params[dim].kwargs['up_attr'] is not None and dim < batch_max_dim:
                x[dim].append(params[dim].kwargs['up_attr'])

    xs = [None for _ in range(batch_max_dim+1)]
    for i in range(batch_max_dim+1):
        if len(x[i]) > 0:
            xs[i] = torch.cat(x[i], dim=0)

    for i in range(batch_max_dim+1):
        if xs[i] is None or batched_xs[i] is None:
            assert xs[i] == batched_xs[i]
        else:
            assert torch.equal(xs[i], batched_xs[i])


def test_batching_returns_the_same_up_attr():
    data_list = get_testing_complex_list()

    # Try multiple parameters
    dims = [1, 2, 3]
    bs = list(range(2, 11))
    params = itertools.product(bs, dims)
    for batch_size, batch_max_dim, in params:
        data_loader = DataLoader(data_list, batch_size=batch_size, max_dim=batch_max_dim)

        # Batched
        batched_x = [[] for _ in range(batch_max_dim+1)]
        for batch in data_loader:
            params = batch.get_all_cochain_params()
            assert len(params) <= batch_max_dim+1
            for dim in range(len(params)):
                if params[dim].kwargs['up_attr'] is not None:
                    batched_x[dim].append(params[dim].kwargs['up_attr'])

        batched_xs = [None for _ in range(batch_max_dim+1)]
        for i in range(batch_max_dim+1):
            if len(batched_x[i]) > 0:
                batched_xs[i] = torch.cat(batched_x[i], dim=0)

        # Un-batched
        x = [[] for _ in range(batch_max_dim+1)]
        for complex in data_list:
            params = complex.get_all_cochain_params()
            for dim in range(min(len(params), batch_max_dim+1)):
                # TODO: Modify test after merging the top_feature branch
                # Right now, the last level cannot have top features
                if params[dim].kwargs['up_attr'] is not None and dim < batch_max_dim:
                    x[dim].append(params[dim].kwargs['up_attr'])

        xs = [None for _ in range(batch_max_dim+1)]
        for i in range(batch_max_dim+1):
            if len(x[i]) > 0:
                xs[i] = torch.cat(x[i], dim=0)

        for i in range(batch_max_dim+1):
            if xs[i] is None or batched_xs[i] is None:
                assert xs[i] == batched_xs[i]
            else:
                assert torch.equal(xs[i], batched_xs[i])


@pytest.mark.data
def test_batching_returns_the_same_down_attr_on_proteins():
    dataset = load_dataset('PROTEINS', max_dim=3, fold=0, init_method='mean')
    assert len(dataset) == 1113
    split_idx = dataset.get_idx_split()
    dataset = dataset[split_idx['valid']]
    assert len(dataset) == 111

    batch_max_dim = 3
    data_loader = DataLoader(dataset, batch_size=32, max_dim=batch_max_dim)

    batched_x = [[] for _ in range(batch_max_dim+1)]
    for batch in data_loader:
        params = batch.get_all_cochain_params()
        assert len(params) <= batch_max_dim+1
        for dim in range(len(params)):
            if params[dim].kwargs['down_attr'] is not None:
                batched_x[dim].append(params[dim].kwargs['down_attr'])

    batched_xs = [None for _ in range(batch_max_dim+1)]
    for i in range(batch_max_dim+1):
        if len(batched_x[i]) > 0:
            batched_xs[i] = torch.cat(batched_x[i], dim=0)

    # Un-batched
    x = [[] for _ in range(batch_max_dim+1)]
    for complex in dataset:
        params = complex.get_all_cochain_params()
        for dim in range(min(len(params), batch_max_dim+1)):
            if params[dim].kwargs['down_attr'] is not None:
                x[dim].append(params[dim].kwargs['down_attr'])

    xs = [None for _ in range(batch_max_dim+1)]
    for i in range(batch_max_dim+1):
        if len(x[i]) > 0:
            xs[i] = torch.cat(x[i], dim=0)

    for i in range(batch_max_dim+1):
        if xs[i] is None or batched_xs[i] is None:
            assert xs[i] == batched_xs[i]
        else:
            assert len(xs[i]) == len(batched_xs[i])
            assert torch.equal(xs[i], batched_xs[i])


def test_batching_returns_the_same_down_attr():
    data_list = get_testing_complex_list()

    # Try multiple parameters
    dims = [1, 2, 3]
    bs = list(range(2, 11))
    params = itertools.product(bs, dims)
    for batch_size, batch_max_dim, in params:
        data_loader = DataLoader(data_list, batch_size=batch_size, max_dim=batch_max_dim)

        batched_x = [[] for _ in range(batch_max_dim+1)]
        for batch in data_loader:
            params = batch.get_all_cochain_params()
            assert len(params) <= batch_max_dim+1
            for dim in range(len(params)):
                if params[dim].kwargs['down_attr'] is not None:
                    batched_x[dim].append(params[dim].kwargs['down_attr'])

        batched_xs = [None for _ in range(batch_max_dim+1)]
        for i in range(batch_max_dim+1):
            if len(batched_x[i]) > 0:
                batched_xs[i] = torch.cat(batched_x[i], dim=0)

        # Un-batched
        x = [[] for _ in range(batch_max_dim+1)]
        for complex in data_list:
            params = complex.get_all_cochain_params()
            for dim in range(min(len(params), batch_max_dim+1)):
                if params[dim].kwargs['down_attr'] is not None:
                    x[dim].append(params[dim].kwargs['down_attr'])

        xs = [None for _ in range(batch_max_dim+1)]
        for i in range(batch_max_dim+1):
            if len(x[i]) > 0:
                xs[i] = torch.cat(x[i], dim=0)

        for i in range(batch_max_dim+1):
            if xs[i] is None or batched_xs[i] is None:
                assert xs[i] == batched_xs[i]
            else:
                assert len(xs[i]) == len(batched_xs[i])
                assert torch.equal(xs[i], batched_xs[i])
                

@pytest.mark.data
def test_batching_of_boundary_index_on_proteins():
    dataset = load_dataset('PROTEINS', max_dim=3, fold=0, init_method='mean')
    assert len(dataset) == 1113
    split_idx = dataset.get_idx_split()
    dataset = dataset[split_idx['valid']]
    assert len(dataset) == 111

    batch_max_dim = 3
    data_loader = DataLoader(dataset, batch_size=32, max_dim=batch_max_dim)

    batched_x_boundaries = [[] for _ in range(batch_max_dim+1)]
    batched_x_cells = [[] for _ in range(batch_max_dim+1)]
    for batch in data_loader:
        params = batch.get_all_cochain_params()
        assert len(params) <= batch_max_dim+1
        for dim in range(len(params)):
            if params[dim].kwargs['boundary_attr'] is not None:
                assert params[dim].boundary_index is not None
                boundary_attrs = params[dim].kwargs['boundary_attr']
                batched_x_boundaries[dim].append(
                    torch.index_select(boundary_attrs, 0, params[dim].boundary_index[0]))
                batched_x_cells[dim].append(
                    torch.index_select(params[dim].x, 0, params[dim].boundary_index[1]))

    batched_xs_boundaries = [None for _ in range(batch_max_dim+1)]
    batched_xs_cells = [None for _ in range(batch_max_dim+1)]
    for i in range(batch_max_dim+1):
        if len(batched_x_boundaries[i]) > 0:
            batched_xs_boundaries[i] = torch.cat(batched_x_boundaries[i], dim=0)
        if len(batched_x_cells[i]) > 0:
            batched_xs_cells[i] = torch.cat(batched_x_cells[i], dim=0)

    # Un-batched
    x_boundaries = [[] for _ in range(batch_max_dim+1)]
    x_cells = [[] for _ in range(batch_max_dim+1)]
    for complex in dataset:
        params = complex.get_all_cochain_params()
        for dim in range(min(len(params), batch_max_dim+1)):
            if params[dim].kwargs['boundary_attr'] is not None:
                assert params[dim].boundary_index is not None
                boundary_attrs = params[dim].kwargs['boundary_attr']
                x_boundaries[dim].append(
                    torch.index_select(boundary_attrs, 0, params[dim].boundary_index[0]))
                x_cells[dim].append(
                    torch.index_select(params[dim].x, 0, params[dim].boundary_index[1]))

    xs_boundaries = [None for _ in range(batch_max_dim+1)]
    xs_cells = [None for _ in range(batch_max_dim+1)]
    for i in range(batch_max_dim+1):
        if len(x_boundaries[i]) > 0:
            xs_boundaries[i] = torch.cat(x_boundaries[i], dim=0)
            xs_cells[i] = torch.cat(x_cells[i], dim=0)

    for i in range(batch_max_dim+1):
        if xs_boundaries[i] is None or batched_xs_boundaries[i] is None:
            assert xs_boundaries[i] == batched_xs_boundaries[i]
        else:
            assert len(xs_boundaries[i]) == len(batched_xs_boundaries[i])
            assert torch.equal(xs_boundaries[i], batched_xs_boundaries[i])
        if xs_cells[i] is None or batched_xs_cells[i] is None:
            assert xs_cells[i] == batched_xs_cells[i]
        else:
            assert len(xs_cells[i]) == len(batched_xs_cells[i])
            assert torch.equal(xs_cells[i], batched_xs_cells[i])
                


def test_batching_of_boundary_index():
    data_list = get_testing_complex_list()

    # Try multiple parameters
    dims = [1, 2, 3]
    bs = list(range(2, 11))
    params = itertools.product(bs, dims)
    for batch_size, batch_max_dim, in params:
        data_loader = DataLoader(data_list, batch_size=batch_size, max_dim=batch_max_dim)

        batched_x_boundaries = [[] for _ in range(batch_max_dim+1)]
        batched_x_cells = [[] for _ in range(batch_max_dim+1)]
        for batch in data_loader:
            params = batch.get_all_cochain_params()
            assert len(params) <= batch_max_dim+1
            for dim in range(len(params)):
                if params[dim].kwargs['boundary_attr'] is not None:
                    assert params[dim].boundary_index is not None
                    boundary_attrs = params[dim].kwargs['boundary_attr']
                    batched_x_boundaries[dim].append(
                        torch.index_select(boundary_attrs, 0, params[dim].boundary_index[0]))
                    batched_x_cells[dim].append(
                        torch.index_select(params[dim].x, 0, params[dim].boundary_index[1]))

        batched_xs_boundaries = [None for _ in range(batch_max_dim+1)]
        batched_xs_cells = [None for _ in range(batch_max_dim+1)]
        for i in range(batch_max_dim+1):
            if len(batched_x_boundaries[i]) > 0:
                batched_xs_boundaries[i] = torch.cat(batched_x_boundaries[i], dim=0)
            if len(batched_x_cells[i]) > 0:
                batched_xs_cells[i] = torch.cat(batched_x_cells[i], dim=0)

        # Un-batched
        x_boundaries = [[] for _ in range(batch_max_dim+1)]
        x_cells = [[] for _ in range(batch_max_dim+1)]
        for complex in data_list:
            params = complex.get_all_cochain_params()
            for dim in range(min(len(params), batch_max_dim+1)):
                if params[dim].kwargs['boundary_attr'] is not None:
                    assert params[dim].boundary_index is not None
                    boundary_attrs = params[dim].kwargs['boundary_attr']
                    x_boundaries[dim].append(
                        torch.index_select(boundary_attrs, 0, params[dim].boundary_index[0]))
                    x_cells[dim].append(
                        torch.index_select(params[dim].x, 0, params[dim].boundary_index[1]))

        xs_boundaries = [None for _ in range(batch_max_dim+1)]
        xs_cells = [None for _ in range(batch_max_dim+1)]
        for i in range(batch_max_dim+1):
            if len(x_boundaries[i]) > 0:
                xs_boundaries[i] = torch.cat(x_boundaries[i], dim=0)
                xs_cells[i] = torch.cat(x_cells[i], dim=0)

        for i in range(batch_max_dim+1):
            if xs_boundaries[i] is None or batched_xs_boundaries[i] is None:
                assert xs_boundaries[i] == batched_xs_boundaries[i]
            else:
                assert len(xs_boundaries[i]) == len(batched_xs_boundaries[i])
                assert torch.equal(xs_boundaries[i], batched_xs_boundaries[i])
            if xs_cells[i] is None or batched_xs_cells[i] is None:
                assert xs_cells[i] == batched_xs_cells[i]
            else:
                assert len(xs_cells[i]) == len(batched_xs_cells[i])
                assert torch.equal(xs_cells[i], batched_xs_cells[i])
                

@pytest.mark.data
def test_data_loader_shuffling():
    dataset = load_dataset('PROTEINS', max_dim=3, fold=0, init_method='mean')
    data_loader = DataLoader(dataset, batch_size=32)

    unshuffled_ys = []
    for data in data_loader:
        unshuffled_ys.append(data.y)

    data_loader = DataLoader(dataset, batch_size=32, shuffle=True)
    shuffled_ys = []
    for data in data_loader:
        shuffled_ys.append(data.y)

    unshuffled_ys = torch.cat(unshuffled_ys, dim=0)
    shuffled_ys = torch.cat(shuffled_ys, dim=0)

    assert list(unshuffled_ys.size()) == list(shuffled_ys.size())
    assert not torch.equal(unshuffled_ys, shuffled_ys)


@pytest.mark.data
def test_idx_splitting_works():
    dataset = load_dataset('PROTEINS', max_dim=3, fold=0, init_method='mean')
    splits = dataset.get_idx_split()

    val_dataset = dataset[splits["valid"]]
    ys1 = []
    for data in val_dataset:
        ys1.append(data.y)

    ys2 = []
    for i in splits['valid']:
        data = dataset.get(i)
        ys2.append(data.y)

    ys1 = torch.cat(ys1, dim=0)
    ys2 = torch.cat(ys2, dim=0)

    assert torch.equal(ys1, ys2)
