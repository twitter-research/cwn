import pytest

from mp.models import DummyChainMessagePassing, DummySimplicialMessagePassing


@pytest.fixture
def build_dummy_smp():
    return DummySimplicialMessagePassing(
        DummyChainMessagePassing(),
        DummyChainMessagePassing(),
        DummyChainMessagePassing(),
    )


def test_propagation_in_dummy_simplicial_message_passing(build_dummy_smp):
    pass
