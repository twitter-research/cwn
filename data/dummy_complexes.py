import torch

from data.complex import Chain, Complex


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
    v_shared_cofaces = torch.tensor([0, 0, 3, 3, 1, 1, 2, 2, 5, 5, 4, 4], dtype=torch.long)
    v_x = torch.tensor([[1], [2], [3], [4], [5]], dtype=torch.float)
    yv = torch.tensor([0, 0, 0, 0, 0], dtype=torch.long)
    v_chain = Chain(dim=0, x=v_x, upper_index=v_up_index, shared_cofaces=v_shared_cofaces, y=yv)

    e_up_index = torch.tensor([[2, 4, 2, 5, 4, 5],
                               [4, 2, 5, 2, 5, 4]], dtype=torch.long)
    e_shared_cofaces = torch.tensor([0, 0, 0, 0, 0, 0], dtype=torch.long)
    e_down_index = torch.tensor([[0, 1, 0, 3, 1, 2, 1, 5, 2, 3, 2, 4, 2, 5, 3, 4, 4, 5],
                                 [1, 0, 3, 0, 2, 1, 5, 1, 3, 2, 4, 2, 5, 2, 4, 3, 5, 4]],
                                dtype=torch.long)
    e_shared_faces = torch.tensor([1, 1, 0, 0, 2, 2, 2, 2, 3, 3, 3, 3, 2, 2, 3, 3, 4, 4],
                                  dtype=torch.long)
    e_x = torch.tensor([[1], [2], [3], [4], [5], [6]], dtype=torch.float)
    ye = torch.tensor([1, 1, 1, 1, 1, 1], dtype=torch.long)
    e_chain = Chain(dim=1, x=e_x, upper_index=e_up_index, lower_index=e_down_index,
                    shared_cofaces=e_shared_cofaces, shared_faces=e_shared_faces, y=ye)

    t_x = torch.tensor([[1]], dtype=torch.float)
    yt = torch.tensor([2], dtype=torch.long)
    t_chain = Chain(dim=2, x=t_x, y=yt)
    return Complex(v_chain, e_chain, t_chain)
