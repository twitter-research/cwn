import torch
import torch.optim as optim

from mp.layers import (
    DummyCellularMessagePassing, CINConv, OrientedConv, InitReduceConv, EmbedVEWithReduce)
from data.dummy_complexes import get_house_complex, get_molecular_complex
from torch import nn
from data.datasets.flow import load_flow_dataset


def test_dummy_cellular_message_passing_with_down_msg():
    house_complex = get_house_complex()
    v_params = house_complex.get_cochain_params(dim=0)
    e_params = house_complex.get_cochain_params(dim=1)
    t_params = house_complex.get_cochain_params(dim=2)

    dsmp = DummyCellularMessagePassing()
    v_x, e_x, t_x = dsmp.forward(v_params, e_params, t_params)

    expected_v_x = torch.tensor([[12], [9], [25], [25], [23]], dtype=torch.float)
    assert torch.equal(v_x, expected_v_x)

    expected_e_x = torch.tensor([[10], [20], [47], [22], [42], [37]], dtype=torch.float)
    assert torch.equal(e_x, expected_e_x)

    expected_t_x = torch.tensor([[1]], dtype=torch.float)
    assert torch.equal(t_x, expected_t_x)


def test_dummy_cellular_message_passing_with_boundary_msg():
    house_complex = get_house_complex()
    v_params = house_complex.get_cochain_params(dim=0)
    e_params = house_complex.get_cochain_params(dim=1)
    t_params = house_complex.get_cochain_params(dim=2)

    dsmp = DummyCellularMessagePassing(use_boundary_msg=True, use_down_msg=False)
    v_x, e_x, t_x = dsmp.forward(v_params, e_params, t_params)

    expected_v_x = torch.tensor([[12], [9], [25], [25], [23]], dtype=torch.float)
    assert torch.equal(v_x, expected_v_x)

    expected_e_x = torch.tensor([[4], [7], [23], [9], [25], [24]], dtype=torch.float)
    assert torch.equal(e_x, expected_e_x)

    expected_t_x = torch.tensor([[15]], dtype=torch.float)
    assert torch.equal(t_x, expected_t_x)


def test_dummy_cellular_message_passing_on_molecular_cell_complex():
    molecular_complex = get_molecular_complex()
    v_params = molecular_complex.get_cochain_params(dim=0)
    e_params = molecular_complex.get_cochain_params(dim=1)
    ring_params = molecular_complex.get_cochain_params(dim=2)

    dsmp = DummyCellularMessagePassing(use_boundary_msg=True, use_down_msg=True)
    v_x, e_x, ring_x = dsmp.forward(v_params, e_params, ring_params)

    expected_v_x = torch.tensor([[12], [24], [24], [15], [25], [31], [47], [24]],
        dtype=torch.float)
    assert torch.equal(v_x, expected_v_x)

    expected_e_x = torch.tensor([[35], [79], [41], [27], [66], [70], [92], [82], [53]],
        dtype=torch.float)
    assert torch.equal(e_x, expected_e_x)

    # The first cell feature is given by 1[x] + 0[up] + (2+2)[down] + (1+2+3+4)[boundaries] = 15
    # The 2nd cell is given by 2[x] + 0[up] + (1+2)[down] + (2+5+6+7+8)[boundaries] = 33
    expected_ring_x = torch.tensor([[15], [33]], dtype=torch.float)
    assert torch.equal(ring_x, expected_ring_x)


def test_cin_conv_training():
    msg_net = nn.Sequential(nn.Linear(2, 1))
    update_net = nn.Sequential(nn.Linear(1, 3))

    cin_conv = CINConv(1, 1, msg_net, msg_net, update_net, 0.05)

    all_params_before = []
    for p in cin_conv.parameters():
        all_params_before.append(p.clone().data)
    assert len(all_params_before) > 0

    house_complex = get_house_complex()

    v_params = house_complex.get_cochain_params(dim=0)
    e_params = house_complex.get_cochain_params(dim=1)
    t_params = house_complex.get_cochain_params(dim=2)

    yv = house_complex.get_labels(dim=0)
    ye = house_complex.get_labels(dim=1)
    yt = house_complex.get_labels(dim=2)
    y = torch.cat([yv, ye, yt])

    optimizer = optim.SGD(cin_conv.parameters(), lr=0.001)
    optimizer.zero_grad()

    out_v, out_e, out_t = cin_conv.forward(v_params, e_params, t_params)
    out = torch.cat([out_v, out_e, out_t], dim=0)

    criterion = nn.CrossEntropyLoss()
    loss = criterion(out, y)
    loss.backward()
    optimizer.step()

    all_params_after = []
    for p in cin_conv.parameters():
        all_params_after.append(p.clone().data)
    assert len(all_params_after) == len(all_params_before)

    # Check that parameters have been updated.
    for i, _ in enumerate(all_params_before):
        assert not torch.equal(all_params_before[i], all_params_after[i])


def test_orient_conv_on_flow_dataset():
    import numpy as np

    np.random.seed(4)
    update_up = nn.Sequential(nn.Linear(1, 4))
    update_down = nn.Sequential(nn.Linear(1, 4))
    update = nn.Sequential(nn.Linear(1, 4))

    train, _, G = load_flow_dataset(num_points=400, num_train=3, num_test=3)
    number_of_edges = G.number_of_edges()

    model = OrientedConv(1, 1, 1, update_up_nn=update_up, update_down_nn=update_down,
        update_nn=update, act_fn=torch.tanh)
    model.eval()

    out = model.forward(train[0])
    assert out.size(0) == number_of_edges
    assert out.size(1) == 4


def test_init_reduce_conv_on_house_complex():
    house_complex = get_house_complex()
    v_params = house_complex.get_cochain_params(dim=0)
    e_params = house_complex.get_cochain_params(dim=1)
    t_params = house_complex.get_cochain_params(dim=2)

    conv = InitReduceConv(reduce='add')

    ex = conv.forward(v_params.x, e_params.boundary_index)
    expected_ex = torch.tensor([[3], [5], [7], [5], [9], [8]], dtype=torch.float)
    assert torch.equal(expected_ex, ex)

    tx = conv.forward(e_params.x, t_params.boundary_index)
    expected_tx = torch.tensor([[14]], dtype=torch.float)
    assert torch.equal(expected_tx, tx)


def test_embed_with_reduce_layer_on_house_complex():
    house_complex = get_house_complex()
    cochains = house_complex.cochains
    params = house_complex.get_all_cochain_params()

    embed_layer = nn.Embedding(num_embeddings=32, embedding_dim=10)
    init_reduce = InitReduceConv()
    conv = EmbedVEWithReduce(embed_layer, None, init_reduce)

    # Simulate the lack of features in these dimensions.
    params[1].x = None
    params[2].x = None

    xs = conv.forward(*params)

    assert len(xs) == 3
    assert xs[0].dim() == 2
    assert xs[0].size(0) == cochains[0].num_cells
    assert xs[0].size(1) == 10
    assert xs[1].size(0) == cochains[1].num_cells
    assert xs[1].size(1) == 10
    assert xs[2].size(0) == cochains[2].num_cells
    assert xs[2].size(1) == 10


