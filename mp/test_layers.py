import torch
import torch.optim as optim

from mp.layers import DummySimplicialMessagePassing, SINConv, SINChainConv
from data.dummy_complexes import get_house_complex, get_square_dot_complex
from torch import nn


def test_dummy_simplicial_message_passing():
    house_complex = get_house_complex()
    v_params = house_complex.get_chain_params(dim=0)
    e_params = house_complex.get_chain_params(dim=1)
    t_params = house_complex.get_chain_params(dim=2)

    dsmp = DummySimplicialMessagePassing()
    v_x, e_x, t_x = dsmp.forward(v_params, e_params, t_params)

    expected_v_x = torch.tensor([[12], [9], [25], [25], [23]], dtype=torch.float)
    assert torch.equal(v_x, expected_v_x)

    expected_e_x = torch.tensor([[10], [20], [47], [22], [42], [37]], dtype=torch.float)
    assert torch.equal(e_x, expected_e_x)

    expected_t_x = torch.tensor([[1]], dtype=torch.float)
    assert torch.equal(t_x, expected_t_x)


def test_sin_conv_training():
    msg_net = nn.Sequential(nn.Linear(2, 1))
    update_net = nn.Sequential(nn.Linear(1, 3))

    sin_conv = SINConv(1, 1, msg_net, msg_net, update_net, 0.05)

    all_params_before = []
    for p in sin_conv.parameters():
        all_params_before.append(p.clone().data)
    assert len(all_params_before) > 0

    house_complex = get_house_complex()

    v_params = house_complex.get_chain_params(dim=0)
    e_params = house_complex.get_chain_params(dim=1)
    t_params = house_complex.get_chain_params(dim=2)

    yv = house_complex.get_labels(dim=0)
    ye = house_complex.get_labels(dim=1)
    yt = house_complex.get_labels(dim=2)
    y = torch.cat([yv, ye, yt])

    optimizer = optim.SGD(sin_conv.parameters(), lr=0.001)
    optimizer.zero_grad()

    out_v, out_e, out_t = sin_conv.forward(v_params, e_params, t_params)
    out = torch.cat([out_v, out_e, out_t], dim=0)

    criterion = nn.CrossEntropyLoss()
    loss = criterion(out, y)
    loss.backward()
    optimizer.step()

    all_params_after = []
    for p in sin_conv.parameters():
        all_params_after.append(p.clone().data)
    assert len(all_params_after) == len(all_params_before)

    # Check that parameters have been updated.
    for i, _ in enumerate(all_params_before):
        assert not torch.equal(all_params_before[i], all_params_after[i])
