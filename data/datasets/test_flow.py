import numpy as np
import torch
import matplotlib.pyplot as plt
import os

from scipy.spatial import Delaunay
from data.datasets.flow_utils import load_flow_dataset, create_hole, is_inside_rectangle
from data.datasets import FlowDataset
from definitions import ROOT_DIR


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
    train, test, _ = load_flow_dataset(num_points=300, num_train=200, num_test=20)
    assert len(train) == 200
    assert len(test) == 20

    label_count = {0: 0, 1: 0, 2: 0}

    for chain in train + test:
        assert torch.sum(chain.x == 1) + torch.sum(chain.x == -1) == torch.count_nonzero(chain.x)

        assert len(chain.upper_orient) == chain.upper_index.size(1)
        assert len(chain.lower_orient) == chain.lower_index.size(1)
        assert chain.upper_index.max() < chain.x.size(0), print(chain.upper_index.max(),
            chain.x.size(0))
        assert chain.lower_index.max() < chain.x.size(0)

        assert (torch.sum(chain.upper_orient == 1) > 0)
        assert (torch.sum(chain.upper_orient == -1) > 0)
        assert (torch.sum(chain.upper_orient == 1)
                + torch.sum(chain.upper_orient == -1) == chain.upper_orient.numel())

        assert (torch.sum(chain.lower_orient == 1) > 0)
        assert (torch.sum(chain.lower_orient == -1) > 0)
        assert (torch.sum(chain.lower_orient == 1)
                + torch.sum(chain.lower_orient == -1) == chain.lower_orient.numel())

        label_count[chain.y.item()] += 1

    assert label_count[0] == 200 // 3 + 20 // 3
    assert label_count[1] == 200 // 3 + 20 // 3
    assert label_count[2] == 200 - 2 * (200 // 3) + 20 - 2 * (20 // 3)


def plot_arrow(p1, p2, color='red'):
    plt.arrow(p1[0], p1[1], p2[0] - p1[0], p2[1] - p1[1], color=color,
        shape='full', lw=3, length_includes_head=True, head_width=.01, zorder=10)


def test_visualise_flow_dataset():
    root = os.path.join(ROOT_DIR, 'datasets')
    name = 'FLOW'
    dataset = FlowDataset(os.path.join(root, name), name, num_points=300, train_samples=500,
            val_samples=100, load_graph=True)
    G = dataset.G
    edge_to_tuple = G.graph['edge_to_tuple']
    triangles = G.graph['triangles']
    points = G.graph['points']

    plt.figure(figsize=(10, 8))
    plt.triplot(points[:, 0], points[:, 1], triangles)
    plt.plot(points[:, 0], points[:, 1], 'o')

    chain = dataset[265]
    x = chain.x

    source_edge = 92
    source_points = edge_to_tuple[source_edge]
    plot_arrow(points[source_points[0]], points[source_points[1]], color='black')

    path_length = 0
    for i in range(len(x)):
        flow = x[i].item()
        if flow == 0:
            continue
        path_length += 1

        nodes1 = edge_to_tuple[i]
        if flow > 0:
            p1, p2 = points[nodes1[0]], points[nodes1[1]]
        else:
            p1, p2 = points[nodes1[1]], points[nodes1[0]],

        plt.arrow(p1[0], p1[1], p2[0] - p1[0], p2[1] - p1[1], color='red',
            shape='full', lw=3, length_includes_head=True, head_width=.01, zorder=10)

    # lower_index = chain.lower_index
    # for i in range(lower_index.size(1)):
    #     n1, n2 = lower_index[0, i].item(), lower_index[1, i].item()
    #     if n1 == source_edge:
    #         source_points = edge_to_tuple[n2]
    #         orient = chain.lower_orient[i].item()
    #         color = 'green' if orient == 1.0 else 'yellow'
    #         plot_arrow(points[source_points[0]], points[source_points[1]], color=color)

    upper_index = chain.upper_index
    for i in range(upper_index.size(1)):
        n1, n2 = upper_index[0, i].item(), upper_index[1, i].item()
        if n1 == source_edge:
            source_points = edge_to_tuple[n2]
            orient = chain.upper_orient[i].item()
            color = 'green' if orient == 1.0 else 'yellow'
            plot_arrow(points[source_points[0]], points[source_points[1]], color=color)

    plt.show()
