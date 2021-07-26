import seaborn as sns
import matplotlib.pyplot as plt
import os

from data.datasets import FlowDataset
from definitions import ROOT_DIR

sns.set_style('white')
sns.color_palette("tab10")


def plot_arrow(p1, p2, color='red'):
    plt.arrow(p1[0], p1[1], p2[0] - p1[0], p2[1] - p1[1], color=color,
        shape='full', lw=3, length_includes_head=True, head_width=.01, zorder=10)


def visualise_flow_dataset():
    root = os.path.join(ROOT_DIR, 'datasets')
    name = 'FLOW'
    dataset = FlowDataset(os.path.join(root, name), name, num_points=1000, train_samples=1000,
            val_samples=200, classes=3, load_graph=True)
    G = dataset.G
    edge_to_tuple = G.graph['edge_to_tuple']
    triangles = G.graph['triangles']
    points = G.graph['points']

    plt.figure(figsize=(10, 8))
    plt.triplot(points[:, 0], points[:, 1], triangles)
    plt.plot(points[:, 0], points[:, 1], 'o')

    for i, cochain in enumerate([dataset[180], dataset[480]]):
        colors = ['red', 'navy', 'purple']
        color = colors[i]

        x = cochain.x
        #
        # source_edge = 92
        # source_points = edge_to_tuple[source_edge]
        # plot_arrow(points[source_points[0]], points[source_points[1]], color='black')

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

            plt.arrow(p1[0], p1[1], p2[0] - p1[0], p2[1] - p1[1], color=color,
                shape='full', lw=3, length_includes_head=True, head_width=.01, zorder=10)

    # lower_index = cochain.lower_index
    # for i in range(lower_index.size(1)):
    #     n1, n2 = lower_index[0, i].item(), lower_index[1, i].item()
    #     if n1 == source_edge:
    #         source_points = edge_to_tuple[n2]
    #         orient = cochain.lower_orient[i].item()
    #         color = 'green' if orient == 1.0 else 'yellow'
    #         plot_arrow(points[source_points[0]], points[source_points[1]], color=color)

    # upper_index = cochain.upper_index
    # for i in range(upper_index.size(1)):
    #     n1, n2 = upper_index[0, i].item(), upper_index[1, i].item()
    #     if n1 == source_edge:
    #         source_points = edge_to_tuple[n2]
    #         orient = cochain.upper_orient[i].item()
    #         color = 'green' if orient == 1.0 else 'yellow'
    #         plot_arrow(points[source_points[0]], points[source_points[1]], color=color)

    plt.show()


if __name__ == "__main__":
    visualise_flow_dataset()
