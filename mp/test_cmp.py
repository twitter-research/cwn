import pytest
import torch

from mp.smp import ChainMessagePassing


@pytest.fixture
def build_cmp():
    return ChainMessagePassing()


def test_propagate_in_cmp(build_cmp):
    """We build a graph in the shape of a house (a triangle on top of a square)
    and test propagation at the edge level."""

    # [0, 1, 2] are the edges that form the triangle. They are all upper adjacent.
    up_index = torch.tensor([[0, 1, 0, 2, 1, 2],
                             [1, 0, 2, 0, 2, 1]], dtype=torch.long)
    # [2, 3, 4, 5] for the edges of the square. They are lower adjacent (share a common vertex).
    # We also need to add the edges of the triangle again because they are also lower adjacent.
    down_index = torch.tensor([[0, 1, 0, 2, 1, 2, 2, 3, 3, 4, 4, 5, 2, 5, 0, 3, 1, 5],
                               [1, 0, 2, 0, 2, 1, 3, 2, 4, 3, 5, 4, 5, 2, 3, 0, 5, 1]],
                              dtype=torch.long)
    # We initialise the edges with dummy scalar features
    x = torch.tensor([[1], [2], [3], [4], [5], [6]], dtype=torch.float)

    # Extract the message passing object and propagate
    cmp = build_cmp
    updated_x = cmp.propagate(up_index, down_index, x=x)
    expected_updated_x = torch.tensor([[14], [14], [16], [9], [10], [10]], dtype=torch.float)

    assert torch.equal(updated_x, expected_updated_x)


def test_propagate_at_vertex_level_in_cmp(build_cmp):
    """We build a graph in the shape of a house (a triangle on top of a square)
    and test propagation at the vertex level. This makes sure propagate works when
    down_index is None.
    """

    # [0, 1, 2] are the edges that form the triangle. They are all upper adjacent.
    up_index = torch.tensor([[0, 1, 0, 4, 1, 2, 1, 4, 2, 3, 3, 4],
                             [1, 0, 4, 0, 2, 1, 4, 1, 3, 2, 4, 3]], dtype=torch.long)
    # [2, 3, 4, 5] for the edges of the square. They are lower adjacent (share a common vertex).
    # We also need to add the edges of the triangle again because they are also lower adjacent.
    down_index = None

    # We initialise the vertices with dummy scalar features
    x = torch.tensor([[1], [2], [3], [4], [5]], dtype=torch.float)

    # Extract the message passing object and propagate
    cmp = build_cmp
    updated_x = cmp.propagate(up_index, down_index, x=x)
    expected_updated_x = torch.tensor([[7], [9], [6], [8], [7]], dtype=torch.float)

    assert torch.equal(updated_x, expected_updated_x)


def test_propagate_at_triangle_level_in_cmp_when_there_is_a_single_one(build_cmp):
    """We build a graph in the shape of a house (a triangle on top of a square)
    and test propagation at the triangle level. This makes sure that propagate works when
    up_index is None."""

    # When there is a single triangle, there is no upper or lower adjacency
    up_index = None
    down_index = None

    # We initialise the vertices with dummy scalar features
    x = torch.tensor([[1]], dtype=torch.float)

    # Extract the message passing object and propagate
    cmp = build_cmp
    updated_x = cmp.propagate(up_index, down_index, x=x)
    expected_updated_x = torch.tensor([[1]], dtype=torch.float)

    assert torch.equal(updated_x, expected_updated_x)


def test_propagate_at_triangle_level_in_cmp(build_cmp):
    """We build a graph formed of two triangles sharing an edge.
    This makes sure that propagate works when up_index is None."""

    # When there is a single triangle, there is no upper or lower adjacency
    up_index = None
    down_index = torch.tensor([[0, 1],
                               [1, 0]], dtype=torch.long)

    # We initialise the vertices with dummy scalar features
    x = torch.tensor([[32], [17]], dtype=torch.float)

    # Extract the message passing object and propagate
    cmp = build_cmp
    updated_x = cmp.propagate(up_index, down_index, x=x)
    expected_updated_x = torch.tensor([[17], [32]], dtype=torch.float)

    assert torch.equal(updated_x, expected_updated_x)