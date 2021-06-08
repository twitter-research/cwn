import numpy as np
import torch

from scipy.spatial import Delaunay
from data.datasets.flow_utils import load_flow_dataset, create_hole, is_inside_rectangle


def test_create_hole():
    # This seed contains some edge cases.
    np.random.seed(4)
    points = np.random.uniform(size=(400, 2))
    tri = Delaunay(points)

    hole1 = np.array([[0.2, 0.2], [0.4, 0.4]])
    points, triangles = create_hole(points, tri.simplices, hole1)

    assert triangles.max() == len(points) - 1
    assert triangles.min() == 0

    # Check all points are outside the hole
    for i in range(len(points)):
        assert not is_inside_rectangle(points[i], hole1)

    # Double check each point appears in some triangle.
    for i in range(len(points)):
        assert np.sum(triangles == i) > 0


def test_flow_util_dataset_loading():
    # Fix seed for reproducibility
    np.random.seed(0)

    train, test, _ = load_flow_dataset(num_points=300, num_train=10, num_test=10)
    assert len(train) == 10
    assert len(test) == 10

    label_count = {0: 0, 1: 0, 2: 0}

    for chain in train + test:
        # checks x values (flow direction) are either +1 or -1
        assert (torch.sum(chain.x == 1) + torch.sum(chain.x == -1)
                == torch.count_nonzero(chain.x))

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

        label_count[chain.y.item()] += 1

    # checks distribution of labels
    assert label_count[0] == 10 // 2 + 10 // 2
    assert label_count[1] == 10 // 2 + 10 // 2
