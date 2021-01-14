import torch
import pytest

from mp.models import DummyChainMessagePassing, DummySimplicialMessagePassing


@pytest.fixture
def build_dummy_smp():
    return lambda: DummySimplicialMessagePassing(
        DummyChainMessagePassing(),
        DummyChainMessagePassing(),
        DummyChainMessagePassing(),
    )


def test_propagation_in_dummy_simplicial_message_passing(build_dummy_smp):
    # TODO(Cris): Use the data.py data structures for this test once they are ready.
    # TODO(Cris): Create a house dataset to be reused across all test.
    dsmp = build_dummy_smp()