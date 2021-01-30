import pytest
import torch

from torch import Tensor
from data.complex import ComplexBatch
from data.dummy_complexes import get_house_complex, get_square_complex, get_pyramid_complex, get_square_dot_complex, get_kite_complex
from data.data_loading import DataLoader

from data.complex import ComplexBatch
from data.dummy_complexes import get_testing_complex_list
from data.data_loading import DataLoader
from mp.models import SIN0


def validate_double_house(batch):
    
    expected_node_upper = torch.tensor([[0, 1, 0, 3, 1, 2, 2, 3, 2, 4, 3, 4, 5, 6, 5, 8, 6, 7, 7, 8, 7, 9, 8, 9],
                                        [1, 0, 3, 0, 2, 1, 3, 2, 4, 2, 4, 3, 6, 5, 8, 5, 7, 6, 8, 7, 9, 7, 9, 8]], dtype=torch.long)
    expected_node_shared_cofaces = torch.tensor([0, 0, 3, 3, 1, 1, 2, 2, 5, 5, 4, 4, 6, 6, 9, 9, 7, 7, 8, 8, 11, 11, 10, 10], dtype=torch.long)
    expected_node_x = torch.tensor([[1], [2], [3], [4], [5], [1], [2], [3], [4], [5]], dtype=torch.float)
    expected_node_y = torch.tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=torch.long)
    expected_node_batch = torch.tensor([0, 0, 0, 0, 0, 1, 1, 1, 1, 1], dtype=torch.long)
    
    expected_edge_upper = torch.tensor([[2, 4, 2, 5, 4, 5, 8, 10, 8, 11, 10, 11],
                                        [4, 2, 5, 2, 5, 4, 10, 8, 11, 8, 11, 10]], dtype=torch.long)
    expected_edge_shared_cofaces = torch.tensor([0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1], dtype=torch.long)
    expected_edge_lower = torch.tensor([[0, 1, 0, 3, 1, 2, 1, 5, 2, 3, 2, 4, 2, 5, 3, 4, 4, 5, 6, 7, 6, 9, 7, 8, 7, 11, 8, 9, 8, 10, 8, 11, 9, 10, 10, 11],
                                 [1, 0, 3, 0, 2, 1, 5, 1, 3, 2, 4, 2, 5, 2, 4, 3, 5, 4, 7, 6, 9, 6, 8, 7, 11, 7, 9, 8, 10, 8, 11, 8, 10, 9, 11, 10]],
                                dtype=torch.long)
    expected_edge_shared_faces = torch.tensor([1, 1, 0, 0, 2, 2, 2, 2, 3, 3, 3, 3, 2, 2, 3, 3, 4, 4, 6, 6, 5, 5, 7, 7, 7, 7, 8, 8, 8, 8, 7, 7, 8, 8, 9, 9],
                                  dtype=torch.long)
    expected_edge_x = torch.tensor([[1], [2], [3], [4], [5], [6], [1], [2], [3], [4], [5], [6]], dtype=torch.float)
    expected_edge_y = torch.tensor([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], dtype=torch.long)
    expected_edge_batch = torch.tensor([0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1], dtype=torch.long)
    
    expected_triangle_x = torch.tensor([[1], [1]], dtype=torch.float)
    expected_triangle_y = torch.tensor([2, 2], dtype=torch.long)
    expected_triangle_batch = torch.tensor([0, 1], dtype=torch.long)
    
    assert torch.equal(expected_node_upper, batch.nodes.upper_index)
    assert torch.equal(expected_node_shared_cofaces, batch.nodes.shared_cofaces)
    assert batch.nodes.lower_index is None
    assert batch.nodes.shared_faces is None
    assert torch.equal(expected_node_x, batch.nodes.x)
    assert torch.equal(expected_node_y, batch.nodes.y)
    assert torch.equal(expected_node_batch, batch.nodes.batch)
    
    assert torch.equal(expected_edge_upper, batch.edges.upper_index)
    assert torch.equal(expected_edge_shared_cofaces, batch.edges.shared_cofaces)
    assert torch.equal(expected_edge_lower, batch.edges.lower_index)
    assert torch.equal(expected_edge_shared_faces, batch.edges.shared_faces)
    assert torch.equal(expected_edge_x, batch.edges.x)
    assert torch.equal(expected_edge_y, batch.edges.y)
    assert torch.equal(expected_edge_batch, batch.edges.batch)
    
    assert batch.triangles.upper_index is None
    assert batch.triangles.lower_index is None
    assert batch.triangles.shared_cofaces is None
    assert batch.triangles.shared_faces is None
    assert torch.equal(expected_triangle_x, batch.triangles.x)
    assert torch.equal(expected_triangle_y, batch.triangles.y)
    assert torch.equal(expected_triangle_batch, batch.triangles.batch)
    

def validate_square_dot_and_square(batch):
    
    expected_node_upper = torch.tensor([        [0, 1, 0, 3, 1, 2, 2, 3, 5, 6, 5, 8, 6, 7, 7, 8],
                                                [1, 0, 3, 0, 2, 1, 3, 2, 6, 5, 8, 5, 7, 6, 8, 7]], dtype=torch.long)
    expected_node_shared_cofaces = torch.tensor([0, 0, 3, 3, 1, 1, 2, 2, 4, 4, 7, 7, 5, 5, 6, 6], dtype=torch.long)
    expected_node_x = torch.tensor([[1], [2], [3], [4], [5], [1], [2], [3], [4]], dtype=torch.float)
    expected_node_y = torch.tensor([0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=torch.long)
    expected_node_batch = torch.tensor([0, 0, 0, 0, 0, 1, 1, 1, 1], dtype=torch.long)
    
    expected_edge_lower = torch.tensor([      [0, 1, 0, 3, 1, 2, 2, 3, 4, 5, 4, 7, 5, 6, 6, 7],
                                              [1, 0, 3, 0, 2, 1, 3, 2, 5, 4, 7, 4, 6, 5, 7, 6]], dtype=torch.long)
    expected_edge_shared_faces = torch.tensor([1, 1, 0, 0, 2, 2, 3, 3, 6, 6, 5, 5, 7, 7, 8, 8],
                                  dtype=torch.long)
    expected_edge_x = torch.tensor([[1], [2], [3], [4], [1], [2], [3], [4]], dtype=torch.float)
    expected_edge_y = torch.tensor([1, 1, 1, 1, 1, 1, 1, 1,], dtype=torch.long)
    expected_edge_batch = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1], dtype=torch.long)
    
    assert torch.equal(expected_node_upper, batch.nodes.upper_index)
    assert torch.equal(expected_node_shared_cofaces, batch.nodes.shared_cofaces)
    assert batch.nodes.lower_index is None
    assert batch.nodes.shared_faces is None
    assert torch.equal(expected_node_x, batch.nodes.x)
    assert torch.equal(expected_node_y, batch.nodes.y)
    assert torch.equal(expected_node_batch, batch.nodes.batch)
    
    assert batch.edges.upper_index is None
    assert batch.edges.shared_cofaces is None
    assert torch.equal(expected_edge_lower, batch.edges.lower_index)
    assert torch.equal(expected_edge_shared_faces, batch.edges.shared_faces)
    assert torch.equal(expected_edge_x, batch.edges.x)
    assert torch.equal(expected_edge_y, batch.edges.y)
    assert torch.equal(expected_edge_batch, batch.edges.batch)
    

def validate_kite_and_house(batch):

    kite_node_upper = torch.tensor([[0, 1, 0, 2, 1, 2, 1, 3, 2, 3, 3, 4],
                                    [1, 0, 2, 0, 2, 1, 3, 1, 3, 2, 4, 3]], dtype=torch.long)
    shifted_house_node_upper = 5 + torch.tensor([[0, 1, 0, 3, 1, 2, 2, 3, 2, 4, 3, 4],
                                        [1, 0, 3, 0, 2, 1, 3, 2, 4, 2, 4, 3]], dtype=torch.long)
    expected_node_upper = torch.cat([kite_node_upper, shifted_house_node_upper], 1)
    
    kite_node_shared_cofaces = torch.tensor([0, 0, 2, 2, 1, 1, 3, 3, 4, 4, 5, 5], dtype=torch.long)
    shifted_house_node_shared_cofaces = 6 + torch.tensor([0, 0, 3, 3, 1, 1, 2, 2, 5, 5, 4, 4], dtype=torch.long)
    expected_node_shared_cofaces = torch.cat([kite_node_shared_cofaces, shifted_house_node_shared_cofaces], 0)
    
    expected_node_x = torch.tensor([[1], [2], [3], [4], [5], [1], [2], [3], [4], [5]], dtype=torch.float)
    expected_node_y = torch.tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=torch.long)
    expected_node_batch = torch.tensor([0, 0, 0, 0, 0, 1, 1, 1, 1, 1], dtype=torch.long)
    
    kite_edge_upper = torch.tensor([[0, 1, 0, 2, 1, 2, 1, 3, 1, 4, 3, 4],
                                    [1, 0, 2, 0, 2, 1, 3, 1, 4, 1, 4, 3]], dtype=torch.long)
    shifted_house_edge_upper = 6 + torch.tensor([[2, 4, 2, 5, 4, 5],
                                                 [4, 2, 5, 2, 5, 4]], dtype=torch.long)
    expected_edge_upper = torch.cat([kite_edge_upper, shifted_house_edge_upper], 1)
    
    kite_edge_shared_cofaces = torch.tensor([0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1], dtype=torch.long)
    shifted_house_edge_shared_cofaces = 2 + torch.tensor([0, 0, 0, 0, 0, 0], dtype=torch.long)
    expected_edge_shared_cofaces = torch.cat([kite_edge_shared_cofaces, shifted_house_edge_shared_cofaces], 0)
    
    kite_edge_lower = torch.tensor([ [0, 1, 0, 3, 1, 3, 0, 2, 1, 2, 2, 4, 1, 4, 3, 4, 3, 5, 4, 5],
                                     [1, 0, 3, 0, 3, 1, 2, 0, 2, 1, 4, 2, 4, 1, 4, 3, 5, 3, 5, 4]], dtype=torch.long)
    shifted_house_lower = 6 + torch.tensor([[0, 1, 0, 3, 1, 2, 1, 5, 2, 3, 2, 4, 2, 5, 3, 4, 4, 5],
                                            [1, 0, 3, 0, 2, 1, 5, 1, 3, 2, 4, 2, 5, 2, 4, 3, 5, 4]], dtype=torch.long)
    expected_edge_lower = torch.cat([kite_edge_lower, shifted_house_lower], 1)
    
    kite_edge_shared_faces = torch.tensor([1, 1, 1, 1, 1, 1, 0, 0, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3], dtype=torch.long)
    shifted_house_edge_shared_faces = 5 + torch.tensor([1, 1, 0, 0, 2, 2, 2, 2, 3, 3, 3, 3, 2, 2, 3, 3, 4, 4], dtype=torch.long)
    expected_edge_shared_faces = torch.cat([kite_edge_shared_faces, shifted_house_edge_shared_faces], 0)
    
    expected_edge_x = torch.tensor([[1], [2], [3], [4], [5], [6], [1], [2], [3], [4], [5], [6]], dtype=torch.float)
    expected_edge_y = torch.tensor([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], dtype=torch.long)
    expected_edge_batch = torch.tensor([0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1], dtype=torch.long)
    
    expected_triangle_lower = torch.tensor([[0, 1],
                                            [1, 0]], dtype=torch.long)
    expected_triangle_shared_faces = torch.tensor([1, 1], dtype=torch.long)
    expected_triangle_x = torch.tensor([[1], [2], [1]], dtype=torch.float)
    expected_triangle_y = torch.tensor([2, 2, 2], dtype=torch.long)
    expected_triangle_batch = torch.tensor([0, 0, 1], dtype=torch.long)
    
    assert torch.equal(expected_node_upper, batch.nodes.upper_index)
    assert torch.equal(expected_node_shared_cofaces, batch.nodes.shared_cofaces)
    assert batch.nodes.lower_index is None
    assert batch.nodes.shared_faces is None
    assert torch.equal(expected_node_x, batch.nodes.x)
    assert torch.equal(expected_node_y, batch.nodes.y)
    assert torch.equal(expected_node_batch, batch.nodes.batch)
    
    assert torch.equal(expected_edge_upper, batch.edges.upper_index)
    assert torch.equal(expected_edge_shared_cofaces, batch.edges.shared_cofaces)
    assert torch.equal(expected_edge_lower, batch.edges.lower_index)
    assert torch.equal(expected_edge_shared_faces, batch.edges.shared_faces)
    assert torch.equal(expected_edge_x, batch.edges.x)
    assert torch.equal(expected_edge_y, batch.edges.y)
    assert torch.equal(expected_edge_batch, batch.edges.batch)
    
    assert batch.triangles.upper_index is None
    assert batch.triangles.shared_cofaces is None
    assert torch.equal(expected_triangle_lower, batch.triangles.lower_index)
    assert torch.equal(expected_triangle_shared_faces, batch.triangles.shared_faces)
    assert torch.equal(expected_triangle_x, batch.triangles.x)
    assert torch.equal(expected_triangle_y, batch.triangles.y)
    assert torch.equal(expected_triangle_batch, batch.triangles.batch)
    
    
def validate_house_and_square(batch):
    
    expected_node_upper = torch.tensor([[0, 1, 0, 3, 1, 2, 2, 3, 2, 4, 3, 4, 5, 6, 5, 8, 6, 7, 7, 8],
                                        [1, 0, 3, 0, 2, 1, 3, 2, 4, 2, 4, 3, 6, 5, 8, 5, 7, 6, 8, 7]], dtype=torch.long)
    expected_node_shared_cofaces = torch.tensor([0, 0, 3, 3, 1, 1, 2, 2, 5, 5, 4, 4, 6, 6, 9, 9, 7, 7, 8, 8], dtype=torch.long)
    expected_node_x = torch.tensor([[1], [2], [3], [4], [5], [1], [2], [3], [4]], dtype=torch.float)
    expected_node_y = torch.tensor([0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=torch.long)
    expected_node_batch = torch.tensor([0, 0, 0, 0, 0, 1, 1, 1, 1], dtype=torch.long)
    
    expected_edge_upper = torch.tensor([[2, 4, 2, 5, 4, 5],
                                        [4, 2, 5, 2, 5, 4]], dtype=torch.long)
    expected_edge_shared_cofaces = torch.tensor([0, 0, 0, 0, 0, 0], dtype=torch.long)
    expected_edge_lower = torch.tensor([      [0, 1, 0, 3, 1, 2, 1, 5, 2, 3, 2, 4, 2, 5, 3, 4, 4, 5, 6, 7, 6, 9, 7, 8, 8, 9],
                                              [1, 0, 3, 0, 2, 1, 5, 1, 3, 2, 4, 2, 5, 2, 4, 3, 5, 4, 7, 6, 9, 6, 8, 7, 9, 8]],
                                dtype=torch.long)
    expected_edge_shared_faces = torch.tensor([1, 1, 0, 0, 2, 2, 2, 2, 3, 3, 3, 3, 2, 2, 3, 3, 4, 4, 6, 6, 5, 5, 7, 7, 8, 8],
                                  dtype=torch.long)
    expected_edge_x = torch.tensor([[1], [2], [3], [4], [5], [6], [1], [2], [3], [4]], dtype=torch.float)
    expected_edge_y = torch.tensor([1, 1, 1, 1, 1, 1, 1, 1, 1, 1,], dtype=torch.long)
    expected_edge_batch = torch.tensor([0, 0, 0, 0, 0, 0, 1, 1, 1, 1], dtype=torch.long)
    
    expected_triangle_x = torch.tensor([[1]], dtype=torch.float)
    expected_triangle_y = torch.tensor([2], dtype=torch.long)
    expected_triangle_batch = torch.tensor([0], dtype=torch.long)
    
    assert torch.equal(expected_node_upper, batch.nodes.upper_index)
    assert torch.equal(expected_node_shared_cofaces, batch.nodes.shared_cofaces)
    assert batch.nodes.lower_index is None
    assert batch.nodes.shared_faces is None
    assert torch.equal(expected_node_x, batch.nodes.x)
    assert torch.equal(expected_node_y, batch.nodes.y)
    assert torch.equal(expected_node_batch, batch.nodes.batch)
    
    assert torch.equal(expected_edge_upper, batch.edges.upper_index)
    assert torch.equal(expected_edge_shared_cofaces, batch.edges.shared_cofaces)
    assert torch.equal(expected_edge_lower, batch.edges.lower_index)
    assert torch.equal(expected_edge_shared_faces, batch.edges.shared_faces)
    assert torch.equal(expected_edge_x, batch.edges.x)
    assert torch.equal(expected_edge_y, batch.edges.y)
    assert torch.equal(expected_edge_batch, batch.edges.batch)
    
    assert batch.triangles.upper_index is None
    assert batch.triangles.lower_index is None
    assert batch.triangles.shared_cofaces is None
    assert batch.triangles.shared_faces is None
    assert torch.equal(expected_triangle_x, batch.triangles.x)
    assert torch.equal(expected_triangle_y, batch.triangles.y)
    assert torch.equal(expected_triangle_batch, batch.triangles.batch)
    
    
def validate_house_square_house(batch):
    
    expected_node_upper = torch.tensor([[0, 1, 0, 3, 1, 2, 2, 3, 2, 4, 3, 4, 5, 6, 5, 8, 6, 7, 7, 8, 9, 10, 9, 12, 10, 11, 11, 12, 11, 13, 12, 13],
                                        [1, 0, 3, 0, 2, 1, 3, 2, 4, 2, 4, 3, 6, 5, 8, 5, 7, 6, 8, 7, 10, 9, 12, 9, 11, 10, 12, 11, 13, 11, 13, 12]],
                                       dtype=torch.long)
    expected_node_shared_cofaces = torch.tensor([0, 0, 3, 3, 1, 1, 2, 2, 5, 5, 4, 4, 6, 6, 9, 9, 7, 7, 8, 8, 10, 10, 13, 13, 11, 11, 12, 12, 15, 15, 14, 14],
                                                dtype=torch.long)
    expected_node_x = torch.tensor([[1], [2], [3], [4], [5], [1], [2], [3], [4], [1], [2], [3], [4], [5]], dtype=torch.float)
    expected_node_y = torch.tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=torch.long)
    expected_node_batch = torch.tensor([0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 2], dtype=torch.long)
    
    expected_edge_upper = torch.tensor([[2, 4, 2, 5, 4, 5, 12, 14, 12, 15, 14, 15],
                                        [4, 2, 5, 2, 5, 4, 14, 12, 15, 12, 15, 14]], dtype=torch.long)
    expected_edge_shared_cofaces = torch.tensor([0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1], dtype=torch.long)
    expected_edge_lower = torch.tensor([      [0, 1, 0, 3, 1, 2, 1, 5, 2, 3, 2, 4, 2, 5, 3, 4, 4, 5, 6, 7, 6, 9, 7, 8, 8, 9, 10, 11, 10, 13, 11, 12, 11, 15, 12, 13, 12, 14, 12, 15, 13, 14, 14, 15],
                                              [1, 0, 3, 0, 2, 1, 5, 1, 3, 2, 4, 2, 5, 2, 4, 3, 5, 4, 7, 6, 9, 6, 8, 7, 9, 8, 11, 10, 13, 10, 12, 11, 15, 11, 13, 12, 14, 12, 15, 12, 14, 13, 15, 14]],
                                dtype=torch.long)
    expected_edge_shared_faces = torch.tensor([1, 1, 0, 0, 2, 2, 2, 2, 3, 3, 3, 3, 2, 2, 3, 3, 4, 4, 6, 6, 5, 5, 7, 7, 8, 8, 10, 10, 9,  9, 11, 11, 11, 11, 12, 12, 12, 12, 11, 11, 12, 12, 13, 13],
                                  dtype=torch.long)
    expected_edge_x = torch.tensor([[1], [2], [3], [4], [5], [6], [1], [2], [3], [4], [1], [2], [3], [4], [5], [6]], dtype=torch.float)
    expected_edge_y = torch.tensor([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], dtype=torch.long)
    expected_edge_batch = torch.tensor([0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2], dtype=torch.long)
    
    expected_triangle_x = torch.tensor([[1], [1]], dtype=torch.float)
    expected_triangle_y = torch.tensor([2, 2], dtype=torch.long)
    expected_triangle_batch = torch.tensor([0, 2], dtype=torch.long)
    
    assert torch.equal(expected_node_upper, batch.nodes.upper_index)
    assert torch.equal(expected_node_shared_cofaces, batch.nodes.shared_cofaces)
    assert batch.nodes.lower_index is None
    assert batch.nodes.shared_faces is None
    assert torch.equal(expected_node_x, batch.nodes.x)
    assert torch.equal(expected_node_y, batch.nodes.y)
    assert torch.equal(expected_node_batch, batch.nodes.batch)
    
    assert torch.equal(expected_edge_upper, batch.edges.upper_index)
    assert torch.equal(expected_edge_shared_cofaces, batch.edges.shared_cofaces)
    assert torch.equal(expected_edge_lower, batch.edges.lower_index)
    assert torch.equal(expected_edge_shared_faces, batch.edges.shared_faces)
    assert torch.equal(expected_edge_x, batch.edges.x)
    assert torch.equal(expected_edge_y, batch.edges.y)
    assert torch.equal(expected_edge_batch, batch.edges.batch)
    
    assert batch.triangles.upper_index is None
    assert batch.triangles.lower_index is None
    assert batch.triangles.shared_cofaces is None
    assert batch.triangles.shared_faces is None
    assert torch.equal(expected_triangle_x, batch.triangles.x)
    assert torch.equal(expected_triangle_y, batch.triangles.y)
    assert torch.equal(expected_triangle_batch, batch.triangles.batch)
    
    
def validate_house_no_batching(batch):
        
    expected_node_upper = torch.tensor([[0, 1, 0, 3, 1, 2, 2, 3, 2, 4, 3, 4],
                                        [1, 0, 3, 0, 2, 1, 3, 2, 4, 2, 4, 3]], dtype=torch.long)
    expected_node_shared_cofaces = torch.tensor([0, 0, 3, 3, 1, 1, 2, 2, 5, 5, 4, 4], dtype=torch.long)
    expected_node_x = torch.tensor([[1], [2], [3], [4], [5]], dtype=torch.float)
    expected_node_y = torch.tensor([0, 0, 0, 0, 0], dtype=torch.long)
    expected_node_batch = torch.tensor([0, 0, 0, 0, 0], dtype=torch.long)
    
    expected_edge_upper = torch.tensor([[2, 4, 2, 5, 4, 5],
                                        [4, 2, 5, 2, 5, 4]], dtype=torch.long)
    expected_edge_shared_cofaces = torch.tensor([0, 0, 0, 0, 0, 0], dtype=torch.long)
    expected_edge_lower = torch.tensor([[0, 1, 0, 3, 1, 2, 1, 5, 2, 3, 2, 4, 2, 5, 3, 4, 4, 5],
                                        [1, 0, 3, 0, 2, 1, 5, 1, 3, 2, 4, 2, 5, 2, 4, 3, 5, 4]],
                                dtype=torch.long)
    expected_edge_shared_faces = torch.tensor([1, 1, 0, 0, 2, 2, 2, 2, 3, 3, 3, 3, 2, 2, 3, 3, 4, 4],
                                  dtype=torch.long)
    expected_edge_x = torch.tensor([[1], [2], [3], [4], [5], [6]], dtype=torch.float)
    expected_edge_y = torch.tensor([1, 1, 1, 1, 1, 1], dtype=torch.long)
    expected_edge_batch = torch.tensor([0, 0, 0, 0, 0, 0], dtype=torch.long)
    
    expected_triangle_x = torch.tensor([[1]], dtype=torch.float)
    expected_triangle_y = torch.tensor([2], dtype=torch.long)
    expected_triangle_batch = torch.tensor([0], dtype=torch.long)
    
    assert torch.equal(expected_node_upper, batch.nodes.upper_index)
    assert torch.equal(expected_node_shared_cofaces, batch.nodes.shared_cofaces)
    assert batch.nodes.lower_index is None
    assert batch.nodes.shared_faces is None
    assert torch.equal(expected_node_x, batch.nodes.x)
    assert torch.equal(expected_node_y, batch.nodes.y)
    assert torch.equal(expected_node_batch, batch.nodes.batch)
    
    assert torch.equal(expected_edge_upper, batch.edges.upper_index)
    assert torch.equal(expected_edge_shared_cofaces, batch.edges.shared_cofaces)
    assert torch.equal(expected_edge_lower, batch.edges.lower_index)
    assert torch.equal(expected_edge_shared_faces, batch.edges.shared_faces)
    assert torch.equal(expected_edge_x, batch.edges.x)
    assert torch.equal(expected_edge_y, batch.edges.y)
    assert torch.equal(expected_edge_batch, batch.edges.batch)
    
    assert batch.triangles.upper_index is None
    assert batch.triangles.lower_index is None
    assert batch.triangles.shared_cofaces is None
    assert batch.triangles.shared_faces is None
    assert torch.equal(expected_triangle_x, batch.triangles.x)
    assert torch.equal(expected_triangle_y, batch.triangles.y)
    assert torch.equal(expected_triangle_batch, batch.triangles.batch)
    
    
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
    
    data_loader_1 = DataLoader(data_list_1, batch_size=2)
    data_loader_2 = DataLoader(data_list_2, batch_size=3)
    data_loader_3 = DataLoader(data_list_3, batch_size=3, max_dim=3)
    
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


def test_set_for_features_in_batch():
    house_1 = get_house_complex()
    house_2 = get_house_complex()
    square = get_square_complex()
    complex_list = [house_1, square, house_2]

    vx = torch.range(21, 34).view(14, 1)
    ex = torch.range(21, 36).view(16, 1)
    tx = torch.range(21, 22).view(2, 1)
    xs = [vx, ex, tx]

    batch = ComplexBatch.from_complex_list(complex_list)
    batch.set_xs(xs)

    assert torch.equal(batch.chains[0].x, vx)
    assert torch.equal(batch.chains[1].x, ex)
    assert torch.equal(batch.chains[2].x, tx)

import itertools


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
            params = batch.get_all_chain_params()
            assert len(params) <= batch_max_dim+1
            for dim in range(len(params)):
                batched_x[dim].append(params[dim].x)
        batched_xs = [None for _ in range(batch_max_dim+1)]
        for i in range(batch_max_dim+1):
            if len(batched_x[i]) > 0:
                batched_xs[i] = torch.cat(batched_x[i], dim=0)

        x = [[] for _ in range(batch_max_dim+1)]
        for complex in data_list:
            params = complex.get_all_chain_params()
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
            params = batch.get_all_chain_params()
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
            params = complex.get_all_chain_params()
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
            params = batch.get_all_chain_params()
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
            params = complex.get_all_chain_params()
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
