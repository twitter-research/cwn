import pytest
import torch

from mp.smp import ChainMessagePassing
from torch_geometric.nn.conv import MessagePassing
from data.dummy_complexes import get_square_dot_complex, get_house_complex


def test_edge_propagate_in_cmp():
    """We build a graph in the shape of a house (a triangle on top of a square)
    and test propagation at the edge level."""

    house_complex = get_house_complex()
    e = house_complex.get_chain_params(dim=1)
    assert e.kwargs['face_index'] is not None, e.kwargs['face_index']

    # Extract the message passing object and propagate
    cmp = ChainMessagePassing(up_msg_size=1, down_msg_size=1)
    up_msg, down_msg, face_msg = cmp.propagate(e.up_index, e.down_index, 
                                               e.face_index, x=e.x,
                                               up_attr=e.kwargs['up_attr'],
                                               down_attr=e.kwargs['down_attr'],
                                               face_attr=e.kwargs['face_attr'])
    expected_down_msg = torch.tensor([[6], [10], [17], [9], [13], [10]], dtype=torch.float)
    assert torch.equal(down_msg, expected_down_msg)

    expected_up_msg = torch.tensor([[0], [0], [11], [0], [9], [8]], dtype=torch.float)
    assert torch.equal(up_msg, expected_up_msg)

    expected_face_msg = torch.tensor([[3], [5], [7], [5], [9], [8]], dtype=torch.float)
    assert torch.equal(face_msg, expected_face_msg)


def test_propagate_at_vertex_level_in_cmp():
    """We build a graph in the shape of a house (a triangle on top of a square)
    and test propagation at the vertex level. This makes sure propagate works when
    down_index is None.
    """

    house_complex = get_house_complex()
    v = house_complex.get_chain_params(dim=0)

    # Extract the message passing object and propagate
    cmp = ChainMessagePassing(up_msg_size=1, down_msg_size=1)
    up_msg, down_msg, face_msg = cmp.propagate(v.up_index, v.down_index,
                                               v.face_index, x=v.x,
                                               up_attr=v.kwargs['up_attr'],
                                               down_attr=v.kwargs['down_attr'],
                                               face_attr=v.kwargs['face_attr'])

    expected_up_msg = torch.tensor([[6], [4], [11], [9], [7]], dtype=torch.float)
    assert torch.equal(up_msg, expected_up_msg)

    expected_down_msg = torch.zeros(5, 1)
    assert torch.equal(down_msg, expected_down_msg)

    expected_face_msg = torch.zeros(5, 1)
    assert torch.equal(face_msg, expected_face_msg)


def test_propagate_at_triangle_level_in_cmp_when_there_is_a_single_one():
    """We build a graph in the shape of a house (a triangle on top of a square)
    and test propagation at the triangle level. This makes sure that propagate works when
    up_index is None."""

    house_complex = get_house_complex()
    t = house_complex.get_chain_params(dim=2)

    # Extract the message passing object and propagate
    cmp = ChainMessagePassing(up_msg_size=1, down_msg_size=1)
    up_msg, down_msg, face_msg = cmp.propagate(t.up_index, t.down_index,
                                               t.face_index, x=t.x,
                                               up_attr=t.kwargs['up_attr'],
                                               down_attr=t.kwargs['down_attr'],
                                               face_attr=t.kwargs['face_attr'])

    expected_up_msg = torch.zeros(1, 1)
    assert torch.equal(up_msg, expected_up_msg)

    expected_down_msg = torch.zeros(1, 1)
    assert torch.equal(down_msg, expected_down_msg)

    expected_face_msg = torch.tensor([[14]], dtype=torch.float)
    assert torch.equal(face_msg, expected_face_msg)


def test_propagate_at_triangle_level_in_cmp():
    """We build a graph formed of two triangles sharing an edge.
    This makes sure that propagate works when up_index is None."""
    # TODO: Refactor this test to use the kite complex

    # When there is a single triangle, there is no upper or lower adjacency
    up_index = None
    down_index = torch.tensor([[0, 1],
                               [1, 0]], dtype=torch.long)
    # Add features for the edges shared by the triangles
    down_attr = torch.tensor([[1], [1]])

    # We initialise the vertices with dummy scalar features
    x = torch.tensor([[32], [17]], dtype=torch.float)

    # Extract the message passing object and propagate
    cmp = ChainMessagePassing(up_msg_size=1, down_msg_size=1)
    up_msg, down_msg, _ = cmp.propagate(up_index, down_index, None, x=x, down_attr=down_attr)
    expected_updated_x = torch.tensor([[17], [32]], dtype=torch.float)

    assert torch.equal(up_msg + down_msg, expected_updated_x)


def test_smp_messaging_with_isolated_nodes():
    """
    This checks how pyG handles messages for isolated nodes. This shows that it sends a zero vector.
    """
    square_dot_complex = get_square_dot_complex()
    params = square_dot_complex.get_chain_params(dim=0)

    mp = MessagePassing()
    out = mp.propagate(edge_index=params.up_index, x=params.x)
    isolated_out = out[4]

    # This confirms pyG returns a zero message to isolated vertices
    assert torch.equal(isolated_out, torch.zeros_like(isolated_out))
    for i in range(4):
        assert not torch.equal(out[i], torch.zeros_like(out[i]))

    cmp = ChainMessagePassing(up_msg_size=1, down_msg_size=1)
    up_msg, down_msg, _ = cmp.propagate(up_index=params.up_index, down_index=None, face_index=None, 
                                        x=params.x, up_attr=None)
    assert torch.equal(out, up_msg)
    assert torch.equal(down_msg, torch.zeros_like(down_msg))
