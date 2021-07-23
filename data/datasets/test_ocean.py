import torch
import pytest
from data.datasets.ocean_utils import load_ocean_dataset


@pytest.mark.data
def test_ocean_dataset_generation():
    train, test, _ = load_ocean_dataset()
    assert len(train) == 160
    assert len(test) == 40

    for cochain in train + test:
        # checks the upper/lower orientation features are consistent
        # in shape with the upper/lower indices
        assert len(cochain.upper_orient) == cochain.upper_index.size(1)
        assert len(cochain.lower_orient) == cochain.lower_index.size(1)
        # checks the upper and lower indices are consistent with the number of edges
        assert cochain.upper_index.max() < cochain.x.size(0), print(cochain.upper_index.max(),
            cochain.x.size(0))
        assert cochain.lower_index.max() < cochain.x.size(0), print(cochain.lower_index.max(),
            cochain.x.size(0))

        # checks the values for orientations are either +1 (coherent) or -1 (not coherent)
        assert (torch.sum(cochain.upper_orient == 1)
                + torch.sum(cochain.upper_orient == -1) == cochain.upper_orient.numel())
        assert (torch.sum(cochain.lower_orient == 1)
                + torch.sum(cochain.lower_orient == -1) == cochain.lower_orient.numel())
