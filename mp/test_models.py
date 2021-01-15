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
    #TODO(Cris): Finish this test by checking the new features are the expected ones.

