import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from data.complex import Chain, Complex
from data.utils import get_nx_graph, compute_connectivity, get_adj_index, generate_complex
import itertools as it
import torch

def draw_ppa_ego(data, edge_feature_idx=0, with_labels=False, from_edge_list=False):
    if from_edge_list:
        G = nx.from_edgelist(data)
        pos = nx.spring_layout(G)
        plt.figure(figsize=(2,2), dpi=250)
        plt.box(False)
        nx.draw_networkx(G, pos=pos, with_labels=with_labels, node_size=1.5, node_color='indianred', width=0.5, font_size=3)
    else:
        G = get_nx_graph(data)
        pos = nx.spring_layout(G)
        plt.figure(figsize=(2,2), dpi=250)
        plt.box(False)
        edge_list = data.edge_index.numpy().T
        edge_feats = {(edge[0], edge[1]): data.edge_attr[e] for e, edge in enumerate(edge_list)}
        edge_labels = {edge: edge_feats[edge][edge_feature_idx].item() for edge in edge_feats}
        ecs = [np.asarray([1.0, 1.0, 1.0]) - edge_labels[edge] for edge in G.edges]
        nx.draw_networkx(G, pos=pos, with_labels=with_labels, node_size=1.5, node_color='indianred', edge_color=ecs, width=0.5, font_size=3)
    plt.show()
    plt.close()
    return  


def filter_out_edges(ego, thresholds, filt=np.all):
    edge_list = ego.edge_index.numpy().T
    edge_feats = ego.edge_attr.numpy()
    assert len(thresholds)==edge_feats.shape[1]
    mask = np.repeat(a=np.asarray(thresholds).reshape((1, edge_feats.shape[1])), repeats=edge_list.shape[0], axis=0)
    comp = edge_feats > mask
    to_keep = filt(comp, axis=1)
#     print(edge_feats[:10], '\n', mask[:10], '\n', comp[:10], '\n', to_keep)
    return edge_list[to_keep], to_keep


def extract_complex(ego, threshold_for_edges, threshold_for_cliques, max_size, keep_isolated_nodes=False):
    
    # filter out edges which are not strong enough
    edge_thresholds = threshold_for_edges * np.asarray([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
    filtered_edges, kept_edges = filter_out_edges(ego, edge_thresholds, np.any)
    
    # get the nodes associated with them, unless we want to retain isolated nodes
    G_edges = nx.from_edgelist(filtered_edges)
    if not keep_isolated_nodes:
        filtered_nodes = sorted(list(G_edges.nodes))
    else:
        filtered_nodes = list(range(ego.num_nodes)) 
    
    # filter out edges to be considered in cliques when they are not strong enough
    # NB: variable `temp` will be used to identify edges which are not maximal cliques
    clique_thresholds = threshold_for_cliques * np.asarray([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
    filtered_edges_for_cliques, _ = filter_out_edges(ego, clique_thresholds, np.any)
    temp = set([tuple(sorted(edge)) for edge in filtered_edges_for_cliques])
     
    # initialize facet variables
    all_facets_by_size = dict()
    for k in range(1, max_size+1):
        all_facets_by_size[k] = list()
    all_facets  = set()
    
    # find maximal cliques; those will be considered as the facets in the complex
    # import pdb; pdb.set_trace()
    G_clique = nx.from_edgelist(filtered_edges_for_cliques)
    for clique in nx.find_cliques(G_clique):
        clique = tuple(sorted(clique))
        k = len(clique)
        if k > max_size:
            all_facets_in_largest_clique = [tuple(sorted(comb)) for comb in it.combinations(clique, max_size)]
            for internal_facet in all_facets_in_largest_clique:
                kk = len(internal_facet)
                all_facets_by_size[kk].append(internal_facet)
                all_facets.add(internal_facet)
                if kk == 2:
                    temp.remove(internal_facet)
            continue
        all_facets_by_size[k].append(clique)
        all_facets.add(clique)
        if k == 2:
            temp.remove(clique)
            
    # manually add to the list of facets also edges which are kept
    # but where discarded to compute cliques
    # ... this happens when `threshold_for_edges` < `threshold_for_cliques`
    clique_edges = temp
    for edge in filtered_edges:
        edge = tuple(sorted(edge))
        if edge not in all_facets and edge not in clique_edges:
            all_facets.add(edge)
            all_facets_by_size[2].append(edge)
            
    # manually add to the list of facets also isolated nodes, if any
    if keep_isolated_nodes:
        non_isolated = set(list(G_edges.nodes))
        for node in filtered_nodes:
            if node not in non_isolated:
                node = (node,)
                all_facets.add(node)
                all_facets_by_size[1].append(node)
        
    # compute connectivity
    connectivity = compute_connectivity(all_facets, all_facets_by_size, max_size)
    upper_adjacencies, lower_adjacencies, all_simplices, all_simplices_by_size, upper_adjacencies_labeled = connectivity
    
    # construct chains
    attributes = dict()
    labels = dict()
    upper_indices = dict()
    lower_indices = dict()
    mappings = dict()
    for k in range(1, max_size+1):
        
        simplices = all_simplices_by_size[k]
        if k == 1:
            lower_index = None
        else:
            lower_index, index_to_simplex = get_adj_index(simplices, lower_adjacencies[k], k)
        if k == max_size:
            upper_index = None
        else:
            upper_index, index_to_simplex = get_adj_index(simplices, upper_adjacencies[k], k)

        # in the case of edges we do have feats!
        if k == 2:
            # take only those edges that have not been filtered out and the related features
            edges = (ego.edge_index.numpy().T)[kept_edges]
            feats = ego.edge_attr[kept_edges]
            # just for convenience, create mapping edge -> feats
            edge_to_feats = {tuple(sorted(edge)): feat for edge, feat in zip(edges, feats)}
            x = [edge_to_feats[tuple(index_to_simplex[index].numpy())] for index in range(len(index_to_simplex))]
            x = torch.stack(x)
        else:
            x = None
        
        attributes[k-1] = x
        labels[k-1] = None
        upper_indices[k-1] = upper_index
        lower_indices[k-1] = lower_index
        mappings[k-1] = index_to_simplex
        
    return generate_complex(attributes, labels, upper_indices, lower_indices, mappings, upper_adjacencies_labeled, 0, max_size-1), upper_adjacencies, lower_adjacencies