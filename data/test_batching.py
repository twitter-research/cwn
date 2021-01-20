import pytest
import torch

from data.complex import ComplexBatch
from data.dummy_complexes import get_house_complex

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