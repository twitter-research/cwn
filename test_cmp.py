import pytest
import torch

from cmp import ChainMessagePassing


@pytest.fixture
def build_cmp():
    return ChainMessagePassing()


def test_propagate(build_cmp):
    """We build a graph in the shape of a house (a triangle on top of a square)
    and test propagation at the edge level."""

    # [0, 1, 2] are the edges that form the triangle. They are all upper adjacent.
    up_index = torch.tensor([[0, 1, 0, 2, 1, 2],
                             [1, 0, 2, 0, 2, 1]], dtype=torch.long)
    # [2, 3, 4, 5] for the edges of the square. They are lower adjacent (share a common edge).
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