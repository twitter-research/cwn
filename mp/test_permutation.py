import numpy as np
import torch

from data.utils import compute_ring_2complex
from data.dummy_complexes import get_mol_testing_complex_list, convert_to_graph
from data.complex import ComplexBatch
from mp.models import SparseCIN
from scipy import sparse as sp
from torch_geometric.data import Data

def permute_graph(graph: Data, P: np.ndarray) -> Data:

    # TODO: support edge features and their permutation
    assert graph.edge_attr is None

    # Check validity of permutation matrix
    n = graph.x.size(0)
    assert P.ndim == 2
    assert P.shape[0] == n
    assert np.all(P.sum(0) == np.ones(n))
    assert np.all(P.sum(1) == np.ones(n))
    assert np.all(P.max(0) == np.ones(n))
    assert np.all(P.max(1) == np.ones(n))
    if n > 1:
        assert np.all(P.min(0) == np.zeros(n))
        assert np.all(P.min(1) == np.zeros(n))

    # Apply permutation to features
    x = graph.x.numpy()
    x_perm = torch.FloatTensor(P @ x)

    # Apply perm to labels, if per-node
    if graph.y.size(0) == n:
        y = graph.y.numpy()
        y_perm = torch.tensor(P @ y)
    else:
        y_perm = graph.y.clone().detach()

    # Apply permutation to adjacencies, if any
    if graph.edge_index.size(1) > 0:
        inps = (np.ones(graph.edge_index.size(1)), (graph.edge_index[0].numpy(), graph.edge_index[1].numpy()))
        A = sp.csr_matrix(inps, shape=(n,n))
        P = sp.csr_matrix(P)
        A_perm = P.dot(A).dot(P.transpose()).tocoo()
        edge_index_perm = torch.LongTensor(np.vstack((A_perm.row, A_perm.col)))
    else:
        edge_index_perm = graph.edge_index.clone().detach()

    # Instantiate new graph
    graph_perm = Data(x=x_perm, edge_index=edge_index_perm, y=y_perm)

    return graph_perm


def generate_permutation_matrices(size, amount=10, seed=43):

    Ps = list()
    random_state = np.random.RandomState(seed)
    for _ in range(amount):
        I = np.eye(size)
        perm = random_state.permutation(size)
        Ps.append(I[perm])
    
    return Ps


def test_sparse_cin0_perm_invariance_on_dummy_mol_complexes():

    # Generate reference graph list
    dummy_complexes = get_mol_testing_complex_list()
    dummy_graphs = [convert_to_graph(complex) for complex in dummy_complexes]
    for graph in dummy_graphs:
        graph.edge_attr = None
    # (We convert back to complexes to regenerate signals on edges and rings, fixing max_k to 7)
    dummy_complexes = [compute_ring_2complex(graph.x, graph.edge_index, None, graph.num_nodes, max_k=7, 
                            include_down_adj=False, init_method='sum', init_edges=True, init_rings=True)
                        for graph in dummy_graphs]

    # Instantiate model
    model = SparseCIN(num_input_features=1, num_classes=16, num_layers=3, hidden=32, use_coboundaries=True, nonlinearity='elu')
    model.eval()

    # Compute reference complex embeddings
    embeddings = [model.forward(ComplexBatch.from_complex_list([comp], max_dim=comp.dimension)) for comp in dummy_complexes]

    # Test invariance for multiple random permutations
    for comp_emb, graph in zip(embeddings, dummy_graphs):
        permutations = generate_permutation_matrices(graph.num_nodes, 5)
        print(graph.edge_index)
        for perm in permutations:
            permuted_graph = permute_graph(graph, perm)
            permuted_comp = compute_ring_2complex(permuted_graph.x, permuted_graph.edge_index, None, permuted_graph.num_nodes,
                            max_k=7, include_down_adj=False, init_method='sum', init_edges=True, init_rings=True)
            permuted_emb = model.forward(ComplexBatch.from_complex_list([permuted_comp], max_dim=permuted_comp.dimension))
            assert torch.allclose(comp_emb, permuted_emb, atol=1e-6)
