import torch
import pytest

from mp.models import DummyChainMessagePassing, DummySimplicialMessagePassing
from data.dummy_complexes import get_house_complex


@pytest.fixture
def build_dummy_smp():
    return lambda: DummySimplicialMessagePassing(
        DummyChainMessagePassing(),
        DummyChainMessagePassing(),
        DummyChainMessagePassing(),
    )


def test_dummy_simplicial_message_passing(build_dummy_smp):
    house_complex = get_house_complex()
    v_params = house_complex.get_chain_params(dim=0)
    e_params = house_complex.get_chain_params(dim=1)
    t_params = house_complex.get_chain_params(dim=2)

    dsmp = build_dummy_smp()
    v_x, e_x, t_x = dsmp.forward(v_params, e_params, t_params)

    expected_v_x = torch.tensor([[11], [7], [22], [21], [18]], dtype=torch.float)
    assert torch.equal(v_x, expected_v_x)

    expected_e_x = torch.tensor([[9], [18], [44], [18], [37], [31]], dtype=torch.float)
    assert torch.equal(e_x, expected_e_x)

    expected_t_x = torch.tensor([[1]], dtype=torch.float)
    assert torch.equal(t_x, expected_t_x)
