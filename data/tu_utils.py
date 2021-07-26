"""
Based on code from https://github.com/weihua916/powerful-gnns/blob/master/util.py

MIT License

Copyright (c) 2021 Weihua Hu
Copyright (c) 2021 The CWN Project Authors

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

import networkx as nx
import numpy as np
import torch
import torch_geometric
from torch_geometric.data import Data
from sklearn.model_selection import StratifiedKFold


class S2VGraph(object):
    def __init__(self, g, label, node_tags=None, node_features=None):
        """
            g: a networkx graph
            label: an integer graph label
            node_tags: a list of integer node tags
            node_features: a torch float tensor, one-hot representation of the tag that is used as input to neural nets
            edge_mat: a torch long tensor, contain edge list, will be used to create torch sparse tensor
            neighbors: list of neighbors (without self-loop)
        """
        self.label = label
        self.g = g
        self.node_tags = node_tags
        self.neighbors = []
        self.node_features = 0
        self.edge_mat = 0

        self.max_neighbor = 0


def load_data(path, dataset, degree_as_tag):
    """
        dataset: name of dataset
        test_proportion: ratio of test train split
        seed: random seed for random splitting of dataset
    """

    print('loading data')
    g_list = []
    label_dict = {}
    feat_dict = {}

    with open('%s/%s.txt' % (path, dataset), 'r') as f:
        n_g = int(f.readline().strip())
        for i in range(n_g):
            row = f.readline().strip().split()
            n, l = [int(w) for w in row]
            if not l in label_dict:
                mapped = len(label_dict)
                label_dict[l] = mapped
            g = nx.Graph()
            node_tags = []
            node_features = []
            n_edges = 0
            for j in range(n):
                g.add_node(j)
                row = f.readline().strip().split()
                tmp = int(row[1]) + 2
                if tmp == len(row):
                    # no node attributes
                    row = [int(w) for w in row]
                    attr = None
                else:
                    row, attr = [int(w) for w in row[:tmp]], np.array([float(w) for w in row[tmp:]])
                if not row[0] in feat_dict:
                    mapped = len(feat_dict)
                    feat_dict[row[0]] = mapped
                node_tags.append(feat_dict[row[0]])

                if tmp > len(row):
                    node_features.append(attr)

                n_edges += row[1]
                for k in range(2, len(row)):
                    g.add_edge(j, row[k])

            if node_features != []:
                node_features = np.stack(node_features)
                node_feature_flag = True
            else:
                node_features = None
                node_feature_flag = False

            assert len(g) == n

            g_list.append(S2VGraph(g, l, node_tags))

    #add labels and edge_mat       
    for g in g_list:
        g.neighbors = [[] for i in range(len(g.g))]
        for i, j in g.g.edges():
            g.neighbors[i].append(j)
            g.neighbors[j].append(i)
        degree_list = []
        for i in range(len(g.g)):
            g.neighbors[i] = g.neighbors[i]
            degree_list.append(len(g.neighbors[i]))
        g.max_neighbor = max(degree_list)

        g.label = label_dict[g.label]

        edges = [list(pair) for pair in g.g.edges()]
        edges.extend([[i, j] for j, i in edges])

        deg_list = list(dict(g.g.degree(range(len(g.g)))).values())  # <- this might not be used...!?
        g.edge_mat = torch.LongTensor(edges).transpose(0,1)
    
    if degree_as_tag:
        for g in g_list:
            g.node_tags = list(dict(g.g.degree).values())
            # ^^^ !? it should probably be replaced by the following one:
            # g.node_tags = [g.g.degree[node] for node in range(len(g.g))]

    #Extracting unique tag labels   
    tagset = set([])
    for g in g_list:
        tagset = tagset.union(set(g.node_tags))

    tagset = list(tagset)
    tag2index = {tagset[i]:i for i in range(len(tagset))}

    for g in g_list:
        g.node_features = torch.zeros(len(g.node_tags), len(tagset))
        g.node_features[range(len(g.node_tags)), [tag2index[tag] for tag in g.node_tags]] = 1
        
    # ==================
    # Here we recompute degree encodings with external code,
    # as we observed some unexpected behaviors likely due to
    # incompatibilities w.r.t.  python versions
    # ==================
    def get_node_degrees(graph):
        edge_index = graph.edge_mat
        if edge_index.shape[1] == 0:  # just isolated nodes
            degrees = torch.zeros((graph.node_features.shape[0],1))
        else:
            degrees = torch_geometric.utils.degree(edge_index[0]).unsqueeze(1)
        return degrees
    if degree_as_tag:
        # 1. cumulate node degrees
        degs = torch.cat([get_node_degrees(graph) for graph in g_list], 0)
        # 2. compute unique values
        uniques, corrs = np.unique(degs, return_inverse=True, axis=0)
        # 3. encode
        pointer = 0
        for graph in g_list:
            n = graph.node_features.shape[0]
            hots = torch.LongTensor(corrs[pointer:pointer+n])
            graph.node_features = torch.nn.functional.one_hot(hots, len(uniques)).float()
            pointer += n
    # ====================

    print('# classes: %d' % len(label_dict))
    print('# maximum node tag: %d' % len(tagset))

    print("# data: %d" % len(g_list))

    return g_list, len(label_dict)


def S2V_to_PyG(data):
    
    new_data = Data()
    setattr(new_data, 'edge_index', data.edge_mat)
    setattr(new_data, 'x', data.node_features)
    setattr(new_data, 'num_nodes', data.node_features.shape[0])
    setattr(new_data, 'y', torch.tensor(data.label).unsqueeze(0).long())
    
    return new_data


def separate_data(graph_list, seed, fold_idx):
    assert 0 <= fold_idx and fold_idx < 10, "fold_idx must be from 0 to 9."
    skf = StratifiedKFold(n_splits=10, shuffle = True, random_state = seed)

    labels = [graph.label for graph in graph_list]
    idx_list = []
    for idx in skf.split(np.zeros(len(labels)), labels):
        idx_list.append(idx)
    train_idx, test_idx = idx_list[fold_idx]

    train_graph_list = [graph_list[i] for i in train_idx]
    test_graph_list = [graph_list[i] for i in test_idx]

    return train_graph_list, test_graph_list


def separate_data_given_split(graph_list, path, fold_idx):
    
    ### Splits data based on pre-computed splits
    assert -1 <= fold_idx and fold_idx < 10, "fold_idx must be from 0 to 9."
    
    train_filename = os.path.join(path, '10fold_idx', 'train_idx-{}.txt'.format(fold_idx+1))
    test_filename = os.path.join(path, '10fold_idx', 'test_idx-{}.txt'.format(fold_idx+1))
    train_idx = np.loadtxt(train_filename, dtype=int)
    test_idx = np.loadtxt(test_filename, dtype=int)
        
    train_graph_list = [graph_list[i] for i in train_idx]
    test_graph_list = [graph_list[i] for i in test_idx]

    return train_graph_list, test_graph_list


def get_fold_indices(complex_list, seed, fold_idx):
    assert 0 <= fold_idx and fold_idx < 10, "fold_idx must be from 0 to 9."
    skf = StratifiedKFold(n_splits=10, shuffle = True, random_state = seed)

    labels = [complex.y.item() for complex in complex_list]
    idx_list = []
    for idx in skf.split(np.zeros(len(labels)), labels):
        idx_list.append(idx)
    train_idx, test_idx = idx_list[fold_idx]

    return train_idx.tolist(), test_idx.tolist()
