import torch

from data.utils import compute_ring_2complex
from data.perm_utils import permute_graph, generate_permutation_matrices
from data.dummy_complexes import get_mol_testing_complex_list, convert_to_graph
from data.complex import ComplexBatch
from data.data_loading import DataLoader, load_dataset
from exp.prepare_sr_experiment import prepare
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


def _validate_iso_on_sr(family):

    eps = 0.0001
    hidden = 16
    num_layers = 3
    max_ring_size = 6
    nonlinearity = 'id'
    graph_norm = 'ln'
    readout = 'mean'
    final_readout = 'sum'
    jobs = 64
    seed = 43
    device = torch.device("cuda:" + str(0)) if torch.cuda.is_available() else torch.device("cpu")

    # Build and dump dataset if needed
    prepare(family, jobs, max_ring_size, True, seed)

    # Load reference dataset
    complexes = load_dataset(family, max_dim=2, max_ring_size=max_ring_size)
    permuted_complexes = load_dataset(f'{family}p{seed}', max_dim=2, max_ring_size=max_ring_size)

    # Instantiate model
    model = SparseCIN(num_input_features=1, num_classes=complexes.num_classes, num_layers=num_layers, hidden=hidden, 
                        use_coboundaries=True, nonlinearity=nonlinearity, graph_norm=graph_norm, 
                        readout=readout, final_readout=final_readout)
    model = model.to(device)
    model.eval()

    # Compute reference complex embeddings
    data_loader = DataLoader(complexes, batch_size=8, shuffle=False, num_workers=16, max_dim=2)
    data_loader_perm = DataLoader(permuted_complexes, batch_size=8, shuffle=False, num_workers=16, max_dim=2)

    with torch.no_grad():
        embeddings = [model.forward(batch.to(device)) for batch in data_loader]
        embeddings = torch.cat(embeddings, 0)  # n x d
        perm_embeddings = [model.forward(batch.to(device)) for batch in data_loader_perm]
        perm_embeddings = torch.cat(perm_embeddings, 0)  # n x d
    assert embeddings.size(0) == perm_embeddings.size(0)
    assert embeddings.size(1) == perm_embeddings.size(1)

    # Test iso between perms
    # assert False, embeddings[:10]
    dist = torch.sqrt(torch.sum(torch.square(embeddings - perm_embeddings), 1))
    assert torch.all(dist <= eps)
    
    # for i in range(embeddings.size(0)):
    #     preds = torch.stack((embeddings[i], perm_embeddings[i]), 0)
    #     assert preds.size(0) == 2
    #     assert preds.size(1) == complexes.num_classes
    #     dist = torch.pdist(preds, p=2).item()
    #     assert dist <= eps


def test_sparse_cin0_self_isomorphism_on_sr16622():
    _validate_iso_on_sr('sr16622')

def test_sparse_cin0_self_isomorphism_on_sr251256():
    _validate_iso_on_sr('sr251256')

def test_sparse_cin0_self_isomorphism_on_sr261034():
    _validate_iso_on_sr('sr261034')

def test_sparse_cin0_self_isomorphism_on_sr281264():
    _validate_iso_on_sr('sr281264')

def test_sparse_cin0_self_isomorphism_on_sr291467():
    _validate_iso_on_sr('sr291467')

# def test_sparse_cin0_self_isomorphism_on_sr351668():
#     _validate_iso_on_sr('sr351668')

def test_sparse_cin0_self_isomorphism_on_sr351899():
    _validate_iso_on_sr('sr351899')

def test_sparse_cin0_self_isomorphism_on_sr361446():
    _validate_iso_on_sr('sr361446')

# def test_sparse_cin0_self_isomorphism_on_sr401224():
#     _validate_iso_on_sr('sr401224')
