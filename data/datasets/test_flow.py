import numpy as np
import matplotlib.pyplot as plt
import os

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
    train, test = load_flow_dataset(num_points=300, num_train=200, num_test=20)
    assert len(train) == 200
    assert len(test) == 20

    for chain in train + test:
        assert len(chain.upper_orient) == chain.upper_index.size(1)
        assert len(chain.lower_orient) == chain.lower_index.size(1)
        assert chain.upper_index.max() < chain.x.size(0), print(chain.upper_index.max(), chain.x.size(0))
        assert chain.lower_index.max() < chain.x.size(0)
