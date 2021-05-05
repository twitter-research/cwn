import torch

from data.complex import Chain, Complex

# TODO: make the features for these dummy complexes disjoint to stress tests even more


def get_testing_complex_list():
    return [get_pyramid_complex(), get_house_complex(), get_kite_complex(), get_square_complex(),
            get_square_dot_complex(), get_square_complex(), get_house_complex(),
            get_kite_complex(), get_pyramid_complex(), get_square_dot_complex()]


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

    e_faces = [[0, 1], [1, 2], [2, 3], [0, 3], [3, 4], [2, 4]]
    e_face_index = torch.stack([
        torch.LongTensor(e_faces).view(-1),
        torch.LongTensor([0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5]).view(-1)], 0)
    
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
                    shared_cofaces=e_shared_cofaces, shared_faces=e_shared_faces,
                    face_index=e_face_index, y=ye)

    t_faces = [[2, 4, 5]]
    t_face_index = torch.stack([
        torch.LongTensor(t_faces).view(-1),
        torch.LongTensor([0, 0, 0]).view(-1)], 0)
    t_x = torch.tensor([[1]], dtype=torch.float)
    yt = torch.tensor([2], dtype=torch.long)
    t_chain = Chain(dim=2, x=t_x, y=yt, face_index=t_face_index)
    return Complex(v_chain, e_chain, t_chain)


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
    v_shared_cofaces = torch.tensor([0, 0, 3, 3, 1, 1, 2, 2], dtype=torch.long)
    v_x = torch.tensor([[1], [2], [3], [4]], dtype=torch.float)
    yv = torch.tensor([0, 0, 0, 0], dtype=torch.long)
    v_chain = Chain(dim=0, x=v_x, upper_index=v_up_index, shared_cofaces=v_shared_cofaces, y=yv)

    e_faces = [[0, 1], [1, 2], [2, 3], [0, 3]]
    e_face_index = torch.stack([
        torch.LongTensor(e_faces).view(-1),
        torch.LongTensor([0, 0, 1, 1, 2, 2, 3, 3]).view(-1)], 0)
    e_down_index = torch.tensor([[0, 1, 0, 3, 1, 2, 2, 3],
                                 [1, 0, 3, 0, 2, 1, 3, 2]], dtype=torch.long)
    e_shared_faces = torch.tensor([1, 1, 0, 0, 2, 2, 3, 3], dtype=torch.long)
    e_x = torch.tensor([[1], [2], [3], [4]], dtype=torch.float)
    ye = torch.tensor([1, 1, 1, 1], dtype=torch.long)
    e_chain = Chain(dim=1, x=e_x, lower_index=e_down_index, shared_faces=e_shared_faces, y=ye,
                    face_index=e_face_index)

    return Complex(v_chain, e_chain)


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
    v_shared_cofaces = torch.tensor([0, 0, 3, 3, 1, 1, 2, 2], dtype=torch.long)
    v_x = torch.tensor([[1], [2], [3], [4], [5]], dtype=torch.float)
    yv = torch.tensor([0, 0, 0, 0, 0], dtype=torch.long)
    v_chain = Chain(dim=0, x=v_x, upper_index=v_up_index, shared_cofaces=v_shared_cofaces, y=yv)

    e_faces = [[0, 1], [1, 2], [2, 3], [0, 3]]
    e_face_index = torch.stack([
        torch.LongTensor(e_faces).view(-1),
        torch.LongTensor([0, 0, 1, 1, 2, 2, 3, 3]).view(-1)], 0)
    e_down_index = torch.tensor([[0, 1, 0, 3, 1, 2, 2, 3],
                                 [1, 0, 3, 0, 2, 1, 3, 2]], dtype=torch.long)
    e_shared_faces = torch.tensor([1, 1, 0, 0, 2, 2, 3, 3], dtype=torch.long)
    e_x = torch.tensor([[1], [2], [3], [4]], dtype=torch.float)
    ye = torch.tensor([1, 1, 1, 1], dtype=torch.long)
    e_chain = Chain(dim=1, x=e_x, lower_index=e_down_index, shared_faces=e_shared_faces, y=ye,
                    face_index=e_face_index)

    return Complex(v_chain, e_chain)


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
    v_up_index = torch.tensor(     [[0, 1, 0, 2, 1, 2, 1, 3, 2, 3, 3, 4],
                                    [1, 0, 2, 0, 2, 1, 3, 1, 3, 2, 4, 3]], dtype=torch.long)
    v_shared_cofaces = torch.tensor([0, 0, 2, 2, 1, 1, 3, 3, 4, 4, 5, 5], dtype=torch.long)
    v_x = torch.tensor([[1], [2], [3], [4], [5]], dtype=torch.float)
    yv = torch.tensor([0, 0, 0, 0, 0], dtype=torch.long)
    v_chain = Chain(dim=0, x=v_x, upper_index=v_up_index, shared_cofaces=v_shared_cofaces, y=yv)

    e_faces = [[0, 1], [1, 2], [0, 2], [1, 3], [2, 3], [3, 4]]
    e_face_index = torch.stack([
        torch.LongTensor(e_faces).view(-1),
        torch.LongTensor([0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5]).view(-1)], 0)
    
    e_down_index = torch.tensor([ [0, 1, 0, 3, 1, 3, 0, 2, 1, 2, 2, 4, 1, 4, 3, 4, 3, 5, 4, 5],
                                  [1, 0, 3, 0, 3, 1, 2, 0, 2, 1, 4, 2, 4, 1, 4, 3, 5, 3, 5, 4]],
                                dtype=torch.long)
    e_shared_faces = torch.tensor([1, 1, 1, 1, 1, 1, 0, 0, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3],
                                  dtype=torch.long)
    e_up_index = torch.tensor(     [[0, 1, 0, 2, 1, 2, 1, 3, 1, 4, 3, 4],
                                    [1, 0, 2, 0, 2, 1, 3, 1, 4, 1, 4, 3]], dtype=torch.long)
    e_shared_cofaces = torch.tensor([0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1], dtype=torch.long)
    
    e_x = torch.tensor([[1], [2], [3], [4], [5], [6]], dtype=torch.float)
    ye = torch.tensor([1, 1, 1, 1, 1, 1], dtype=torch.long)
    e_chain = Chain(dim=1, x=e_x, lower_index=e_down_index, shared_faces=e_shared_faces,
                    upper_index=e_up_index, shared_cofaces=e_shared_cofaces, y=ye,
                    face_index=e_face_index)

    t_faces = [[0, 1, 2], [1, 3, 4]]
    t_face_index = torch.stack([
        torch.LongTensor(t_faces).view(-1),
        torch.LongTensor([0, 0, 0, 1, 1, 1]).view(-1)], 0)
    
    t_down_index = torch.tensor( [[0, 1],
                                  [1, 0]], dtype=torch.long)
    t_shared_faces = torch.tensor([1, 1], dtype=torch.long)
    t_x = torch.tensor([[1], [2]], dtype=torch.float)
    yt = torch.tensor([2, 2], dtype=torch.long)
    t_chain = Chain(dim=2, x=t_x, lower_index=t_down_index, shared_faces=t_shared_faces, y=yt,
                    face_index=t_face_index)
    
    return Complex(v_chain, e_chain, t_chain)


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
    v_up_index = torch.tensor([     [0, 1, 0, 2, 0, 3, 1, 2, 1, 3, 2, 3],
                                    [1, 0, 2, 0, 3, 0, 2, 1, 3, 1, 3, 2]], dtype=torch.long)
    v_shared_cofaces = torch.tensor([0, 0, 2, 2, 5, 5, 1, 1, 3, 3, 4, 4], dtype=torch.long)
    v_x = torch.tensor([[1], [2], [3], [4]], dtype=torch.float)
    yv = torch.tensor([3, 3, 3, 3], dtype=torch.long)
    v_chain = Chain(dim=0, x=v_x, upper_index=v_up_index, shared_cofaces=v_shared_cofaces, y=yv)

    e_faces = [[0, 1], [1, 2], [0, 2], [1, 3], [2, 3], [0, 3]]
    e_face_index = torch.stack([
        torch.LongTensor(e_faces).view(-1),
        torch.LongTensor([0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5]).view(-1)], 0)
    
    e_up_index = torch.tensor([     [0, 1, 0, 2, 1, 2, 0, 5, 0, 3, 3, 5, 1, 3, 1, 4, 3, 4, 2, 4, 2, 5, 4, 5],
                                    [1, 0, 2, 0, 2, 1, 5, 0, 3, 0, 5, 3, 3, 1, 4, 1, 4, 3, 4, 2, 5, 2, 5, 4]], dtype=torch.long)
    e_shared_cofaces = torch.tensor([0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3], dtype=torch.long)
    e_down_index = torch.tensor([ [0, 1, 0, 2, 0, 3, 0, 5, 1, 2, 1, 3, 1, 4, 2, 4, 2, 5, 3, 4, 3, 5, 4, 5],
                                  [1, 0, 2, 0, 3, 0, 5, 0, 2, 1, 3, 1, 4, 1, 4, 2, 5, 2, 4, 3, 5, 3, 5, 4]], dtype=torch.long)
    e_shared_faces = torch.tensor([1, 1, 0, 0, 1, 1, 0, 0, 2, 2, 1, 1, 2, 2, 2, 2, 0, 0, 3, 3, 3, 3, 3, 3], dtype=torch.long)    
    e_x = torch.tensor([[1], [2], [3], [4], [5], [6]], dtype=torch.float)
    ye = torch.tensor([1, 1, 1, 1, 1, 1], dtype=torch.long)
    e_chain = Chain(dim=1, x=e_x, lower_index=e_down_index, upper_index=e_up_index,
                    shared_faces=e_shared_faces, shared_cofaces=e_shared_cofaces, y=ye,
                    face_index=e_face_index)

    t_faces = [[0, 1, 2], [0, 3, 5], [1, 3, 4], [2, 4, 5]]
    t_face_index = torch.stack([
        torch.LongTensor(t_faces).view(-1),
        torch.LongTensor([0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3]).view(-1)], 0)
    
    t_up_index = torch.tensor([     [0, 1, 0, 2, 1, 2, 0, 3, 1, 3, 2, 3],
                                    [1, 0, 2, 0, 2, 1, 3, 0, 3, 1, 3, 2]], dtype=torch.long)
    t_shared_cofaces = torch.tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=torch.long)
    t_down_index = torch.tensor([ [0, 1, 0, 2, 0, 3, 1, 2, 1, 3, 2, 3],
                                  [1, 0, 2, 0, 3, 0, 2, 1, 3, 1, 3, 2]], dtype=torch.long)
    t_shared_faces = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3, 5, 5, 4, 4], dtype=torch.long)
    t_x = torch.tensor([[1], [2], [3], [4]], dtype=torch.float)
    yt = torch.tensor([2, 2, 2, 2], dtype=torch.long)
    t_chain = Chain(dim=2, x=t_x, lower_index=t_down_index, upper_index=t_up_index,
                    shared_faces=t_shared_faces, shared_cofaces=t_shared_cofaces, y=yt,
                    face_index=t_face_index)

    p_faces = [[0, 1, 2, 3]]
    p_face_index = torch.stack([
        torch.LongTensor(p_faces).view(-1),
        torch.LongTensor([0, 0, 0, 0]).view(-1)], 0)
    p_x = torch.tensor([[1]], dtype=torch.float)
    yp = torch.tensor([3], dtype=torch.long)
    p_chain = Chain(dim=3, x=p_x, y=yp, face_index=p_face_index)
    
    return Complex(v_chain, e_chain, t_chain, p_chain)
