import os
import sys
import pickle

from data.data_loading import load_dataset, load_graph_dataset
from data.perm_utils import permute_graph, generate_permutation_matrices
from definitions import ROOT_DIR

__families__ = [
    'sr16622',
    'sr251256',
    'sr261034',
    'sr281264',
    'sr291467',
    'sr351668',
    'sr351899',
    'sr361446',
    'sr401224'
]

def prepare(family, jobs, max_ring_size, permute, init, seed):
    root = os.path.join(ROOT_DIR, 'datasets')
    raw_dir = os.path.join(root, 'SR_graphs', 'raw')
    _ = load_dataset(family, max_dim=2, max_ring_size=max_ring_size, n_jobs=jobs, init_method=init)
    if permute:
        graphs, _, _, _, _ = load_graph_dataset(family)
        permuted_graphs = list()
        for graph in graphs:
            perm = generate_permutation_matrices(graph.num_nodes, 1, seed=seed)[0]
            permuted_graph = permute_graph(graph, perm)
            permuted_graphs.append((permuted_graph.edge_index, permuted_graph.num_nodes))
        with open(os.path.join(raw_dir, f'{family}p{seed}.pkl'), 'wb') as handle:
            pickle.dump(permuted_graphs, handle)
        _ = load_dataset(f'{family}p{seed}', max_dim=2, max_ring_size=max_ring_size, n_jobs=jobs, init_method=init)

if __name__ == "__main__":
    
    # Standard args
    passed_args = sys.argv[1:]
    jobs = int(passed_args[0])
    max_ring_size = int(passed_args[1])
    permute = passed_args[2].lower()
    init_method = passed_args[3].lower()
    assert max_ring_size > 3

    # Execute
    for family in __families__:
        print('\n==============================================================')
        print(f'[i] Preprocessing on family {family}...')
        prepare(family, jobs, max_ring_size, permute=='y', init_method, 43)
