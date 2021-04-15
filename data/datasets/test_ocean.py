import torch
from data.datasets.ocean_utils import load_ocean_dataset


def test_ocean_dataset_generation():
    train, test, _ = load_ocean_dataset()
    assert len(train) == 160
    assert len(test) == 40

    for chain in train + test:
        # checks the upper/lower orientation features are consistent
        # in shape with the upper/lower indices
        assert len(chain.upper_orient) == chain.upper_index.size(1)
        assert len(chain.lower_orient) == chain.lower_index.size(1)
        # checks the upper and lower indices are consistent with the number of edges
        assert chain.upper_index.max() < chain.x.size(0), print(chain.upper_index.max(),
            chain.x.size(0))
        assert chain.lower_index.max() < chain.x.size(0), print(chain.lower_index.max(),
            chain.x.size(0))

        # checks the values for orientations are either +1 (coherent) or -1 (not coherent)
        assert (torch.sum(chain.upper_orient == 1)
                + torch.sum(chain.upper_orient == -1) == chain.upper_orient.numel())
        assert (torch.sum(chain.lower_orient == 1)
                + torch.sum(chain.lower_orient == -1) == chain.lower_orient.numel())
