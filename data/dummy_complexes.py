import torch

from data.data import Chain, Complex


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
    v_chain = Chain(dim=0, x=v_x, upper_index=v_up_index, shared_cofaces=v_shared_cofaces)

    e_up_index = torch.tensor([[2, 4, 2, 5, 4, 5],
                               [4, 2, 5, 2, 5, 4]], dtype=torch.long)
    e_shared_cofaces = torch.tensor([0, 0, 0, 0, 0, 0], dtype=torch.long)
    e_down_index = torch.tensor([[0, 1, 0, 3, 1, 2, 1, 5, 2, 3, 2, 4, 2, 5, 4, 5],
                                 [1, 0, 3, 0, 2, 1, 5, 1, 3, 2, 4, 2, 5, 2, 5, 4]],
                                dtype=torch.long)
    e_shared_faces = torch.tensor([1, 1, 0, 0, 2, 2, 2, 2, 3, 3, 3, 3, 2, 2, 4, 4],
                                  dtype=torch.long)
    e_x = torch.tensor([[1], [2], [3], [4], [5], [6]], dtype=torch.float)
    e_chain = Chain(dim=1, x=e_x, upper_index=e_up_index, lower_index=e_down_index,
                    shared_cofaces=e_shared_cofaces, shared_faces=e_shared_faces)

    t_x = torch.tensor([[1]], dtype=torch.float)
    t_chain = Chain(dim=2, x=t_x)
    return Complex(v_chain, e_chain, t_chain)