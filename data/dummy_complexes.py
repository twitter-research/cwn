import torch

from data.complex import Cochain, Complex
from torch_geometric.data import Data


# TODO: make the features for these dummy complexes disjoint to stress tests even more
def convert_to_graph(complex):
    """Extracts the underlying graph of a cochain complex."""
    assert 0 in complex.cochains
    assert complex.cochains[0].num_cells > 0
    cochain = complex.cochains[0]
    x = cochain.x
    y = complex.y
    edge_attr = None
    if cochain.upper_index is None:
        edge_index = torch.LongTensor([[], []])
    else:
        edge_index = cochain.upper_index
        if 1 in complex.cochains and complex.cochains[1].x is not None and cochain.shared_coboundaries is not None:
            edge_attr = torch.index_select(complex.cochains[1].x, 0, cochain.shared_coboundaries)
    if edge_attr is None:
        edge_attr = torch.FloatTensor([[]])
    graph = Data(x=x, edge_index=edge_index, y=y, edge_attr=edge_attr)
    return graph


def get_testing_complex_list():
    """Returns a list of cell complexes used for testing. The list contains many edge cases."""
    return [get_fullstop_complex(), get_pyramid_complex(), get_house_complex(), get_kite_complex(), get_square_complex(),
            get_square_dot_complex(), get_square_complex(), get_fullstop_complex(), get_house_complex(),
            get_kite_complex(), get_pyramid_complex(), get_bridged_complex(), get_square_dot_complex(), get_colon_complex(),
            get_filled_square_complex(), get_molecular_complex(), get_fullstop_complex(), get_colon_complex(),
            get_bridged_complex(), get_colon_complex(), get_fullstop_complex(), get_fullstop_complex(), get_colon_complex()]


def get_mol_testing_complex_list():
    """Returns a list of cell complexes used for testing. The list contains many edge cases."""
    return [get_house_complex(), get_kite_complex(), get_square_complex(), get_fullstop_complex(), get_bridged_complex(),
            get_square_dot_complex(), get_square_complex(), get_filled_square_complex(), get_colon_complex(), get_bridged_complex(),
            get_kite_complex(), get_square_dot_complex(), get_colon_complex(), get_molecular_complex(), get_bridged_complex(),
            get_filled_square_complex(), get_molecular_complex(), get_fullstop_complex(), get_colon_complex()]


def get_house_complex():
    """
    Returns the `house graph` below with dummy features.
    The `house graph` (3-2-4 is a filled triangle):
       4
      / \
     3---2
     |   |
     0---1

       .
      4 5
     . 2 .
     3   1
     . 0 .

       .
      /0\
     .---.
     |   |
     .---.
    """
    v_up_index = torch.tensor([[0, 1, 0, 3, 1, 2, 2, 3, 2, 4, 3, 4],
                               [1, 0, 3, 0, 2, 1, 3, 2, 4, 2, 4, 3]], dtype=torch.long)
    v_shared_coboundaries = torch.tensor([0, 0, 3, 3, 1, 1, 2, 2, 5, 5, 4, 4], dtype=torch.long)
    v_x = torch.tensor([[1], [2], [3], [4], [5]], dtype=torch.float)
    yv = torch.tensor([0, 0, 0, 0, 0], dtype=torch.long)
    v_cochain = Cochain(dim=0, x=v_x, upper_index=v_up_index, shared_coboundaries=v_shared_coboundaries, y=yv)

    e_boundaries = [[0, 1], [1, 2], [2, 3], [0, 3], [3, 4], [2, 4]]
    e_boundary_index = torch.stack([
        torch.LongTensor(e_boundaries).view(-1),
        torch.LongTensor([0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5]).view(-1)], 0)

    e_up_index = torch.tensor([[2, 4, 2, 5, 4, 5],
                               [4, 2, 5, 2, 5, 4]], dtype=torch.long)
    e_shared_coboundaries = torch.tensor([0, 0, 0, 0, 0, 0], dtype=torch.long)
    e_down_index = torch.tensor([[0, 1, 0, 3, 1, 2, 1, 5, 2, 3, 2, 4, 2, 5, 3, 4, 4, 5],
                                 [1, 0, 3, 0, 2, 1, 5, 1, 3, 2, 4, 2, 5, 2, 4, 3, 5, 4]],
        dtype=torch.long)
    e_shared_boundaries = torch.tensor([1, 1, 0, 0, 2, 2, 2, 2, 3, 3, 3, 3, 2, 2, 3, 3, 4, 4],
        dtype=torch.long)
    e_x = torch.tensor([[1], [2], [3], [4], [5], [6]], dtype=torch.float)
    ye = torch.tensor([1, 1, 1, 1, 1, 1], dtype=torch.long)
    e_cochain = Cochain(dim=1, x=e_x, upper_index=e_up_index, lower_index=e_down_index,
        shared_coboundaries=e_shared_coboundaries, shared_boundaries=e_shared_boundaries,
        boundary_index=e_boundary_index, y=ye)

    t_boundaries = [[2, 4, 5]]
    t_boundary_index = torch.stack([
        torch.LongTensor(t_boundaries).view(-1),
        torch.LongTensor([0, 0, 0]).view(-1)], 0)
    t_x = torch.tensor([[1]], dtype=torch.float)
    yt = torch.tensor([2], dtype=torch.long)
    t_cochain = Cochain(dim=2, x=t_x, y=yt, boundary_index=t_boundary_index)
    
    y = torch.LongTensor([v_x.shape[0]])
    return Complex(v_cochain, e_cochain, t_cochain, y=y)


def get_bridged_complex():
    """
    Returns the `bridged graph` below with dummy features.
    The `bridged graph` (0-1-4-3, 1-2-3-4, 0-1-2-3 are filled rings): 
      
     3---2
     |\  |  
     | 4 |
     |  \|
     0---1

     .-2-.
     |4  |  
     3 . 1
     |  5|
     .-0-.

     .---.
     |\1 |  
     | . |
     | 0\|
     .---.
     
     .---.
     |   |  
     | 2 |
     |   |
     .---.
    """
    v_up_index = torch.tensor(     [[0, 1, 0, 3, 1, 2, 1, 4, 2, 3, 3, 4],
                                    [1, 0, 3, 0, 2, 1, 4, 1, 3, 2, 4, 3]], dtype=torch.long)
    v_shared_coboundaries = torch.tensor([0, 0, 3, 3, 1, 1, 5, 5, 2, 2, 4, 4], dtype=torch.long)
    v_x = torch.tensor([[1], [2], [3], [4], [5]], dtype=torch.float)
    yv = torch.tensor([0, 0, 0, 0, 0], dtype=torch.long)
    v_cochain = Cochain(dim=0, x=v_x, upper_index=v_up_index, shared_coboundaries=v_shared_coboundaries, y=yv)

    e_boundaries = [[0, 1], [1, 2], [2, 3], [0, 3], [3, 4], [1, 4]]
    e_boundary_index = torch.stack([
        torch.LongTensor(e_boundaries).view(-1),
        torch.LongTensor([0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5]).view(-1)], 0)

    e_up_index = torch.tensor(     [[0, 1, 0, 2, 0, 3, 0, 3, 0, 4, 0, 5, 1, 2, 1, 2, 1, 3, 1, 4, 1, 5, 2, 3, 2, 4, 2, 5, 3, 4, 3, 5, 4, 5, 4, 5],
                                    [1, 0, 2, 0, 3, 0, 3, 0, 4, 0, 5, 0, 2, 1, 2, 1, 3, 1, 4, 1, 5, 1, 3, 2, 4, 2, 5, 2, 4, 3, 5, 3, 5, 4, 5, 4]], dtype=torch.long)
    e_shared_coboundaries = torch.tensor([2, 2, 2, 2, 0, 0, 2, 2, 0, 0, 0, 0, 1, 1, 2, 2, 2, 2, 1, 1, 1, 1, 2, 2, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1], dtype=torch.long)
    
    e_down_index = torch.tensor( [[0, 1, 0, 3, 0, 5, 1, 2, 1, 5, 2, 3, 2, 4, 3, 4, 4, 5],
                                  [1, 0, 3, 0, 5, 0, 2, 1, 5, 1, 3, 2, 4, 2, 4, 3, 5, 4]], dtype=torch.long)
    e_shared_boundaries = torch.tensor([1, 1, 0, 0, 1, 1, 2, 2, 1, 1, 3, 3, 3, 3, 3, 3, 4, 4], dtype=torch.long)
    
    e_x = torch.tensor([[1], [2], [3], [4], [5], [6]], dtype=torch.float)
    ye = torch.tensor([1, 1, 1, 1, 1, 1], dtype=torch.long)
    e_cochain = Cochain(dim=1, x=e_x, upper_index=e_up_index, lower_index=e_down_index,
        shared_coboundaries=e_shared_coboundaries, shared_boundaries=e_shared_boundaries,
        boundary_index=e_boundary_index, y=ye)
    
    t_boundaries = [[0, 3, 4, 5], [1, 2, 4, 5], [0, 1, 2, 3]]
    t_boundary_index = torch.stack([
        torch.LongTensor(t_boundaries).view(-1),
        torch.LongTensor([0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2]).view(-1)], 0)
    t_down_index = torch.tensor( [[0, 1, 0, 1, 0, 2, 0, 2, 1, 2, 1, 2],
                                  [1, 0, 1, 0, 2, 0, 2, 0, 2, 1, 2, 1]], dtype=torch.long)
    t_shared_boundaries = torch.tensor([4, 4, 5, 5, 0, 0, 3, 3, 1, 1, 2, 2], dtype=torch.long)
    t_x = torch.tensor([[1], [2], [3]], dtype=torch.float)
    yt = torch.tensor([2, 2, 2], dtype=torch.long)
    t_cochain = Cochain(dim=2, x=t_x, y=yt, boundary_index=t_boundary_index, lower_index=t_down_index, shared_boundaries=t_shared_boundaries)

    y = torch.LongTensor([v_x.shape[0]])

    return Complex(v_cochain, e_cochain, t_cochain, y=y)


def get_fullstop_complex():
    """
    Returns the `fullstop graph` below with dummy features.
    The `fullstop graph` is a single isolated node:

    0

    """
    v_x = torch.tensor([[1]], dtype=torch.float)
    yv = torch.tensor([0], dtype=torch.long)
    v_cochain = Cochain(dim=0, x=v_x, y=yv)
    y = torch.LongTensor([v_x.shape[0]])
    return Complex(v_cochain, y=y)


def get_colon_complex():
    """
    Returns the `colon graph` below with dummy features.
    The `colon graph` is made up of two isolated nodes:

    1

    0

    """
    v_x = torch.tensor([[1], [2]], dtype=torch.float)
    yv = torch.tensor([0, 0], dtype=torch.long)
    v_cochain = Cochain(dim=0, x=v_x, y=yv)
    y = torch.LongTensor([v_x.shape[0]])
    return Complex(v_cochain, y=y)


def get_square_complex():
    """
    Returns the `square graph` below with dummy features.
    The `square graph`:

     3---2
     |   |
     0---1

     . 2 .
     3   1
     . 0 .

     .---.
     |   |
     .---.
    """
    v_up_index = torch.tensor([[0, 1, 0, 3, 1, 2, 2, 3],
                               [1, 0, 3, 0, 2, 1, 3, 2]], dtype=torch.long)
    v_shared_coboundaries = torch.tensor([0, 0, 3, 3, 1, 1, 2, 2], dtype=torch.long)
    v_x = torch.tensor([[1], [2], [3], [4]], dtype=torch.float)
    yv = torch.tensor([0, 0, 0, 0], dtype=torch.long)
    v_cochain = Cochain(dim=0, x=v_x, upper_index=v_up_index, shared_coboundaries=v_shared_coboundaries, y=yv)

    e_boundaries = [[0, 1], [1, 2], [2, 3], [0, 3]]
    e_boundary_index = torch.stack([
        torch.LongTensor(e_boundaries).view(-1),
        torch.LongTensor([0, 0, 1, 1, 2, 2, 3, 3]).view(-1)], 0)
    e_down_index = torch.tensor([[0, 1, 0, 3, 1, 2, 2, 3],
                                 [1, 0, 3, 0, 2, 1, 3, 2]], dtype=torch.long)
    e_shared_boundaries = torch.tensor([1, 1, 0, 0, 2, 2, 3, 3], dtype=torch.long)
    e_x = torch.tensor([[1], [2], [3], [4]], dtype=torch.float)
    ye = torch.tensor([1, 1, 1, 1], dtype=torch.long)
    e_cochain = Cochain(dim=1, x=e_x, lower_index=e_down_index, shared_boundaries=e_shared_boundaries, y=ye,
        boundary_index=e_boundary_index)
    
    y = torch.LongTensor([v_x.shape[0]])
    
    return Complex(v_cochain, e_cochain, y=y)


def get_square_dot_complex():
    """
    Returns the `square-dot graph` below with dummy features.
    The `square-dot graph`:

     3---2
     |   |
     0---1  4

     . 2 .
     3   1
     . 0 .  .

     .---.
     |   |
     .---.  .
    """
    v_up_index = torch.tensor([[0, 1, 0, 3, 1, 2, 2, 3],
                               [1, 0, 3, 0, 2, 1, 3, 2]], dtype=torch.long)
    v_shared_coboundaries = torch.tensor([0, 0, 3, 3, 1, 1, 2, 2], dtype=torch.long)
    v_x = torch.tensor([[1], [2], [3], [4], [5]], dtype=torch.float)
    yv = torch.tensor([0, 0, 0, 0, 0], dtype=torch.long)
    v_cochain = Cochain(dim=0, x=v_x, upper_index=v_up_index, shared_coboundaries=v_shared_coboundaries, y=yv)

    e_boundaries = [[0, 1], [1, 2], [2, 3], [0, 3]]
    e_boundary_index = torch.stack([
        torch.LongTensor(e_boundaries).view(-1),
        torch.LongTensor([0, 0, 1, 1, 2, 2, 3, 3]).view(-1)], 0)
    e_down_index = torch.tensor([[0, 1, 0, 3, 1, 2, 2, 3],
                                 [1, 0, 3, 0, 2, 1, 3, 2]], dtype=torch.long)
    e_shared_boundaries = torch.tensor([1, 1, 0, 0, 2, 2, 3, 3], dtype=torch.long)
    e_x = torch.tensor([[1], [2], [3], [4]], dtype=torch.float)
    ye = torch.tensor([1, 1, 1, 1], dtype=torch.long)
    e_cochain = Cochain(dim=1, x=e_x, lower_index=e_down_index, shared_boundaries=e_shared_boundaries, y=ye,
        boundary_index=e_boundary_index)
    
    y = torch.LongTensor([v_x.shape[0]])

    return Complex(v_cochain, e_cochain, y=y)


def get_kite_complex():
    """
    Returns the `kite graph` below with dummy features.
    The `kite graph`:

      2---3---4
     / \ /
    0---1

      . 4 . 5 .
     2 1 3
    . 0 .

      .---.---.
     /0\1/
    .---.
    
    """
    v_up_index = torch.tensor([[0, 1, 0, 2, 1, 2, 1, 3, 2, 3, 3, 4],
                               [1, 0, 2, 0, 2, 1, 3, 1, 3, 2, 4, 3]], dtype=torch.long)
    v_shared_coboundaries = torch.tensor([0, 0, 2, 2, 1, 1, 3, 3, 4, 4, 5, 5], dtype=torch.long)
    v_x = torch.tensor([[1], [2], [3], [4], [5]], dtype=torch.float)
    yv = torch.tensor([0, 0, 0, 0, 0], dtype=torch.long)
    v_cochain = Cochain(dim=0, x=v_x, upper_index=v_up_index, shared_coboundaries=v_shared_coboundaries, y=yv)

    e_boundaries = [[0, 1], [1, 2], [0, 2], [1, 3], [2, 3], [3, 4]]
    e_boundary_index = torch.stack([
        torch.LongTensor(e_boundaries).view(-1),
        torch.LongTensor([0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5]).view(-1)], 0)

    e_down_index = torch.tensor([[0, 1, 0, 3, 1, 3, 0, 2, 1, 2, 2, 4, 1, 4, 3, 4, 3, 5, 4, 5],
                                 [1, 0, 3, 0, 3, 1, 2, 0, 2, 1, 4, 2, 4, 1, 4, 3, 5, 3, 5, 4]],
        dtype=torch.long)
    e_shared_boundaries = torch.tensor([1, 1, 1, 1, 1, 1, 0, 0, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3],
        dtype=torch.long)
    e_up_index = torch.tensor([[0, 1, 0, 2, 1, 2, 1, 3, 1, 4, 3, 4],
                               [1, 0, 2, 0, 2, 1, 3, 1, 4, 1, 4, 3]], dtype=torch.long)
    e_shared_coboundaries = torch.tensor([0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1], dtype=torch.long)

    e_x = torch.tensor([[1], [2], [3], [4], [5], [6]], dtype=torch.float)
    ye = torch.tensor([1, 1, 1, 1, 1, 1], dtype=torch.long)
    e_cochain = Cochain(dim=1, x=e_x, lower_index=e_down_index, shared_boundaries=e_shared_boundaries,
        upper_index=e_up_index, shared_coboundaries=e_shared_coboundaries, y=ye,
        boundary_index=e_boundary_index)

    t_boundaries = [[0, 1, 2], [1, 3, 4]]
    t_boundary_index = torch.stack([
        torch.LongTensor(t_boundaries).view(-1),
        torch.LongTensor([0, 0, 0, 1, 1, 1]).view(-1)], 0)

    t_down_index = torch.tensor([[0, 1],
                                 [1, 0]], dtype=torch.long)
    t_shared_boundaries = torch.tensor([1, 1], dtype=torch.long)
    t_x = torch.tensor([[1], [2]], dtype=torch.float)
    yt = torch.tensor([2, 2], dtype=torch.long)
    t_cochain = Cochain(dim=2, x=t_x, lower_index=t_down_index, shared_boundaries=t_shared_boundaries, y=yt,
        boundary_index=t_boundary_index)

    y = torch.LongTensor([v_x.shape[0]])

    return Complex(v_cochain, e_cochain, t_cochain, y=y)


def get_pyramid_complex():
    """
    Returns the `pyramid` below with dummy features.
    The `pyramid` (corresponds to a 4-clique):
    
       3
      /|\
     /_2_\
    0-----1

       .
     5 4 3
      2.1
    .  0  .
    
       3
      / \
     /   \
    2-----1
   / \   / \
  /   \ /   \
 3-----0-----3
    
       .
      / \
     4   3
    .--1--.
   / 2   0 \
  4   \ /   3
 .--5--.--5--.
    
       3
      / \
     / 2 \
    2-----1
   / \ 0 / \
  / 3 \ / 1 \
 3-----0-----3
 
       .
      /|\
     /_0_\
    .-----.
  
  """
    v_up_index = torch.tensor([[0, 1, 0, 2, 0, 3, 1, 2, 1, 3, 2, 3],
                               [1, 0, 2, 0, 3, 0, 2, 1, 3, 1, 3, 2]], dtype=torch.long)
    v_shared_coboundaries = torch.tensor([0, 0, 2, 2, 5, 5, 1, 1, 3, 3, 4, 4], dtype=torch.long)
    v_x = torch.tensor([[1], [2], [3], [4]], dtype=torch.float)
    yv = torch.tensor([3, 3, 3, 3], dtype=torch.long)
    v_cochain = Cochain(dim=0, x=v_x, upper_index=v_up_index, shared_coboundaries=v_shared_coboundaries, y=yv)

    e_boundaries = [[0, 1], [1, 2], [0, 2], [1, 3], [2, 3], [0, 3]]
    e_boundary_index = torch.stack([
        torch.LongTensor(e_boundaries).view(-1),
        torch.LongTensor([0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5]).view(-1)], 0)

    e_up_index = torch.tensor(
        [[0, 1, 0, 2, 1, 2, 0, 5, 0, 3, 3, 5, 1, 3, 1, 4, 3, 4, 2, 4, 2, 5, 4, 5],
         [1, 0, 2, 0, 2, 1, 5, 0, 3, 0, 5, 3, 3, 1, 4, 1, 4, 3, 4, 2, 5, 2, 5, 4]],
        dtype=torch.long)
    e_shared_coboundaries = torch.tensor(
        [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3], dtype=torch.long)
    e_down_index = torch.tensor(
        [[0, 1, 0, 2, 0, 3, 0, 5, 1, 2, 1, 3, 1, 4, 2, 4, 2, 5, 3, 4, 3, 5, 4, 5],
         [1, 0, 2, 0, 3, 0, 5, 0, 2, 1, 3, 1, 4, 1, 4, 2, 5, 2, 4, 3, 5, 3, 5, 4]],
        dtype=torch.long)
    e_shared_boundaries = torch.tensor(
        [1, 1, 0, 0, 1, 1, 0, 0, 2, 2, 1, 1, 2, 2, 2, 2, 0, 0, 3, 3, 3, 3, 3, 3], dtype=torch.long)
    e_x = torch.tensor([[1], [2], [3], [4], [5], [6]], dtype=torch.float)
    ye = torch.tensor([1, 1, 1, 1, 1, 1], dtype=torch.long)
    e_cochain = Cochain(dim=1, x=e_x, lower_index=e_down_index, upper_index=e_up_index,
        shared_boundaries=e_shared_boundaries, shared_coboundaries=e_shared_coboundaries, y=ye,
        boundary_index=e_boundary_index)

    t_boundaries = [[0, 1, 2], [0, 3, 5], [1, 3, 4], [2, 4, 5]]
    t_boundary_index = torch.stack([
        torch.LongTensor(t_boundaries).view(-1),
        torch.LongTensor([0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3]).view(-1)], 0)

    t_up_index = torch.tensor([[0, 1, 0, 2, 1, 2, 0, 3, 1, 3, 2, 3],
                               [1, 0, 2, 0, 2, 1, 3, 0, 3, 1, 3, 2]], dtype=torch.long)
    t_shared_coboundaries = torch.tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=torch.long)
    t_down_index = torch.tensor([[0, 1, 0, 2, 0, 3, 1, 2, 1, 3, 2, 3],
                                 [1, 0, 2, 0, 3, 0, 2, 1, 3, 1, 3, 2]], dtype=torch.long)
    t_shared_boundaries = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3, 5, 5, 4, 4], dtype=torch.long)
    t_x = torch.tensor([[1], [2], [3], [4]], dtype=torch.float)
    yt = torch.tensor([2, 2, 2, 2], dtype=torch.long)
    t_cochain = Cochain(dim=2, x=t_x, lower_index=t_down_index, upper_index=t_up_index,
        shared_boundaries=t_shared_boundaries, shared_coboundaries=t_shared_coboundaries, y=yt,
        boundary_index=t_boundary_index)

    p_boundaries = [[0, 1, 2, 3]]
    p_boundary_index = torch.stack([
        torch.LongTensor(p_boundaries).view(-1),
        torch.LongTensor([0, 0, 0, 0]).view(-1)], 0)
    p_x = torch.tensor([[1]], dtype=torch.float)
    yp = torch.tensor([3], dtype=torch.long)
    p_cochain = Cochain(dim=3, x=p_x, y=yp, boundary_index=p_boundary_index)

    y = torch.LongTensor([v_x.shape[0]])
        
    return Complex(v_cochain, e_cochain, t_cochain, p_cochain, y=y)


def get_filled_square_complex():
    """This is a cell / cubical complex formed of a single filled square.

     3---2
     |   |
     0---1

     . 2 .
     3   1
     . 0 .

     .---.
     | 0 |
     .---.
    """

    v_up_index = torch.tensor([[0, 1, 0, 3, 1, 2, 2, 3],
                               [1, 0, 3, 0, 2, 1, 3, 2]], dtype=torch.long)
    v_shared_coboundaries = torch.tensor([0, 0, 3, 3, 1, 1, 2, 2], dtype=torch.long)
    v_x = torch.tensor([[1], [2], [3], [4]], dtype=torch.float)
    yv = torch.tensor([0, 0, 0, 0], dtype=torch.long)
    v_cochain = Cochain(dim=0, x=v_x, upper_index=v_up_index, shared_coboundaries=v_shared_coboundaries, y=yv)

    e_boundaries = [[0, 1], [1, 2], [2, 3], [0, 3]]
    e_boundary_index = torch.stack([
        torch.LongTensor(e_boundaries).view(-1),
        torch.LongTensor([0, 0, 1, 1, 2, 2, 3, 3]).view(-1)], 0)
    e_down_index = torch.tensor([[0, 1, 0, 3, 1, 2, 2, 3],
                                 [1, 0, 3, 0, 2, 1, 3, 2]], dtype=torch.long)
    e_shared_boundaries = torch.tensor([1, 1, 0, 0, 2, 2, 3, 3], dtype=torch.long)
    e_x = torch.tensor([[1], [2], [3], [4]], dtype=torch.float)
    ye = torch.tensor([1, 1, 1, 1], dtype=torch.long)

    e_upper_index = torch.tensor([[0, 1, 0, 2, 0, 3, 1, 2, 1, 3, 2, 3],
                                  [1, 0, 2, 0, 3, 0, 2, 1, 3, 1, 3, 2]], dtype=torch.long)
    e_shared_coboundaries = torch.tensor([0]*12, dtype=torch.long)

    e_cochain = Cochain(dim=1, x=e_x, lower_index=e_down_index, shared_boundaries=e_shared_boundaries,
        upper_index=e_upper_index, y=ye, shared_coboundaries=e_shared_coboundaries, boundary_index=e_boundary_index)

    c_boundary_index = torch.LongTensor(
        [[0, 1, 2, 3],
         [0, 0, 0, 0]]
    )
    c_x = torch.tensor([[1]], dtype=torch.float)
    yc = torch.tensor([2], dtype=torch.long)
    c_cochain = Cochain(dim=2, x=c_x, y=yc, boundary_index=c_boundary_index)
    
    y = torch.LongTensor([v_x.shape[0]])

    return Complex(v_cochain, e_cochain, c_cochain, y=y)


def get_molecular_complex():
    """This is a molecule with filled rings.

     3---2---4---5
     |   |       |
     0---1-------6---7

     . 2 . 4 . 5 .
     3   1       6
     . 0 .   7   . 8 .

     .---. --- . --- .
     | 0 |    1      |
     .---. --------- . ---- .
    """

    v_up_index = torch.tensor([[0, 1, 0, 3, 1, 2, 1, 6, 2, 3, 2, 4, 4, 5, 5, 6, 6, 7],
                               [1, 0, 3, 0, 2, 1, 6, 1, 3, 2, 4, 2, 5, 4, 6, 5, 7, 6]],
        dtype=torch.long)
    v_shared_coboundaries = torch.tensor([0, 0, 3, 3, 1, 1, 7, 7, 2, 2, 4, 4, 5, 5, 6, 6, 8, 8],
        dtype=torch.long)
    v_x = torch.tensor([[1], [2], [3], [4], [5], [6], [7], [8]], dtype=torch.float)
    yv = torch.tensor([0, 0, 0, 0, 0, 0, 0, 0], dtype=torch.long)
    v_cochain = Cochain(dim=0, x=v_x, upper_index=v_up_index, shared_coboundaries=v_shared_coboundaries, y=yv)

    e_boundaries = [[0, 1], [1, 2], [2, 3], [0, 3], [1, 6], [2, 4], [4, 5], [5, 6], [6, 7]]
    e_boundary_index = torch.stack([
        torch.LongTensor(e_boundaries).view(-1),
        torch.LongTensor([0, 0, 1, 1, 2, 2, 3, 3, 7, 7, 4, 4, 5, 5, 6, 6, 8, 8]).view(-1)], 0)
    e_down_index = torch.tensor(
        [[0, 1, 0, 3, 1, 2, 2, 3, 1, 4, 2, 4, 4, 5, 5, 6, 6, 7, 6, 8, 7, 8, 0, 7, 1, 7],
         [1, 0, 3, 0, 2, 1, 3, 2, 4, 1, 4, 2, 5, 4, 6, 5, 7, 6, 8, 6, 8, 7, 7, 0, 7, 1]],
        dtype=torch.long)
    e_shared_boundaries = torch.tensor(
        [1, 1, 0, 0, 2, 2, 3, 3, 2, 2, 2, 2, 4, 4, 5, 5, 6, 6, 6, 6, 6, 6, 1, 1, 1, 1],
        dtype=torch.long)
    e_x = torch.tensor([[1], [2], [3], [4], [5], [6], [7], [8], [9]], dtype=torch.float)
    ye = torch.tensor([1, 1, 1, 1, 1, 1, 1, 1, 1], dtype=torch.long)

    e_upper_index_c1 = torch.tensor([[0, 1, 0, 2, 0, 3, 1, 2, 1, 3, 2, 3],
                                     [1, 0, 2, 0, 3, 0, 2, 1, 3, 1, 3, 2]], dtype=torch.long)
    e_upper_index_c2 = torch.tensor([[1, 4, 1, 5, 1, 6, 1, 7, 4, 5, 4, 6, 4, 7, 5, 6, 5, 7, 6, 7],
                                     [4, 1, 5, 1, 6, 1, 7, 1, 5, 4, 6, 4, 7, 4, 6, 5, 7, 5, 7, 6]],
        dtype=torch.long)
    e_upper_index = torch.cat((e_upper_index_c1, e_upper_index_c2), dim=-1)
    e_shared_coboundaries = torch.tensor([0]*12 + [1]*20, dtype=torch.long)

    e_cochain = Cochain(dim=1, x=e_x, lower_index=e_down_index, shared_boundaries=e_shared_boundaries,
        upper_index=e_upper_index, y=ye, shared_coboundaries=e_shared_coboundaries, boundary_index=e_boundary_index)

    c_boundary_index = torch.LongTensor(
        [[0, 1, 2, 3, 1, 4, 5, 6, 7],
         [0, 0, 0, 0, 1, 1, 1, 1, 1]]
    )
    c_x = torch.tensor([[1], [2]], dtype=torch.float)
    c_down_index = torch.tensor([[0, 1],
                                 [1, 0]], dtype=torch.long)
    c_shared_boundaries = torch.tensor([1, 1],  dtype=torch.long)

    yc = torch.tensor([2, 2], dtype=torch.long)
    c_cochain = Cochain(dim=2, x=c_x, y=yc, boundary_index=c_boundary_index, lower_index=c_down_index,
        shared_boundaries=c_shared_boundaries)
    
    y = torch.LongTensor([v_x.shape[0]])

    return Complex(v_cochain, e_cochain, c_cochain, y=y)
