import torch

from data.utils import compute_ring_2complex
from data.perm_utils import permute_graph, generate_permutation_matrices
from data.dummy_complexes import get_mol_testing_complex_list, convert_to_graph
from data.complex import ComplexBatch
from mp.models import SparseCIN

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
        for perm in permutations:
            permuted_graph = permute_graph(graph, perm)
            permuted_comp = compute_ring_2complex(permuted_graph.x, permuted_graph.edge_index, None, permuted_graph.num_nodes,
                            max_k=7, include_down_adj=False, init_method='sum', init_edges=True, init_rings=True)
            permuted_emb = model.forward(ComplexBatch.from_complex_list([permuted_comp], max_dim=permuted_comp.dimension))
            assert torch.allclose(comp_emb, permuted_emb, atol=1e-6)
