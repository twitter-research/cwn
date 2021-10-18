import torch
import numpy as np
import random

from data.utils import compute_ring_2complex
from data.perm_utils import permute_graph, generate_permutation_matrices
from data.dummy_complexes import get_mol_testing_complex_list, convert_to_graph
from data.complex import ComplexBatch
from data.data_loading import DataLoader, load_dataset
from exp.prepare_sr_experiment import prepare
from mp.models import MessagePassingAgnostic, SparseCIN

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


def _get_sr_embeddings(family, seed, baseline=False):

    # Set the seed for everything
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

    # Perform the check in double precision
    torch.set_default_dtype(torch.float64)

    # Please set the parameters below to the ones used in SR experiments.
    # If so, if tests pass then the experiments are deemed sound.
    hidden = 16
    num_layers = 3
    max_ring_size = 6
    use_coboundaries = True
    nonlinearity = 'elu'
    graph_norm = 'id'
    readout = 'sum'
    final_readout = 'sum'
    readout_dims = (0,1,2)
    init = 'sum'
    jobs = 64
    prepare_seed = 43
    device = torch.device("cuda:" + str(0)) if torch.cuda.is_available() else torch.device("cpu")

    # Build and dump dataset if needed
    prepare(family, jobs, max_ring_size, True, init, prepare_seed)

    # Load reference dataset
    complexes = load_dataset(family, max_dim=2, max_ring_size=max_ring_size, init_method=init)
    permuted_complexes = load_dataset(f'{family}p{prepare_seed}', max_dim=2, max_ring_size=max_ring_size, init_method=init)

    # Instantiate model
    if not baseline:
        model = SparseCIN(num_input_features=1, num_classes=complexes.num_classes, num_layers=num_layers, hidden=hidden, 
                            use_coboundaries=use_coboundaries, nonlinearity=nonlinearity, graph_norm=graph_norm, 
                            readout=readout, final_readout=final_readout, readout_dims=readout_dims)
    else:
        hidden = 256
        model = MessagePassingAgnostic(num_input_features=1, num_classes=complexes.num_classes, hidden=hidden,
                                        nonlinearity=nonlinearity, readout=readout)

    model = model.to(device)
    model.eval()

    # Compute reference complex embeddings
    data_loader = DataLoader(complexes, batch_size=8, shuffle=False, num_workers=16, max_dim=2)
    data_loader_perm = DataLoader(permuted_complexes, batch_size=8, shuffle=False, num_workers=16, max_dim=2)

    with torch.no_grad():
        embeddings = list()        
        perm_embeddings = list()        
        for batch in data_loader:
            batch.nodes.x = batch.nodes.x.double()
            batch.edges.x = batch.edges.x.double()
            batch.two_cells.x = batch.two_cells.x.double()
            out = model.forward(batch.to(device))
            embeddings.append(out)
        for batch in data_loader_perm:
            batch.nodes.x = batch.nodes.x.double()
            batch.edges.x = batch.edges.x.double()
            batch.two_cells.x = batch.two_cells.x.double()
            out = model.forward(batch.to(device))
            perm_embeddings.append(out)
        embeddings = torch.cat(embeddings, 0)            # n x d
        perm_embeddings = torch.cat(perm_embeddings, 0)  # n x d
    assert embeddings.size(0) == perm_embeddings.size(0)
    assert embeddings.size(1) == perm_embeddings.size(1) == complexes.num_classes

    return embeddings, perm_embeddings

def _validate_self_iso_on_sr(embeddings, perm_embeddings):
    eps = 0.01
    for i in range(embeddings.size(0)):
        preds = torch.stack((embeddings[i], perm_embeddings[i]), 0)
        assert preds.size(0) == 2
        assert preds.size(1) == embeddings.size(1)
        dist = torch.pdist(preds, p=2).item()
        assert dist <= eps

def _validate_magnitude_embeddings(embeddings):
    # At (5)e8, the fp64 granularity is still (2**29 - 2**28) / (2**52) ≈ 0.000000059604645
    # The fact that we work in such a safe range can also be verified by running the following:
    #   a = torch.DoubleTensor([2.5e8])
    #   d = torch.DoubleTensor([5.0e8])
    #   b = torch.nextafter(a, d)
    #   print(b - a)
    #   >>> tensor([2.9802e-08], dtype=torch.float64)
    thresh = np.array(5.0*1e8, dtype=np.float64)
    apex = torch.max(torch.abs(embeddings))
    print(apex)
    assert apex < thresh

def test_sparse_cin0_self_isomorphism_on_sr16622():
    for seed in range(5):
        embeddings, perm_embeddings = _get_sr_embeddings('sr16622', seed)
        _validate_magnitude_embeddings(embeddings)
        _validate_magnitude_embeddings(perm_embeddings)
        _validate_self_iso_on_sr(embeddings, perm_embeddings)

def test_sparse_cin0_self_isomorphism_on_sr251256():
    for seed in range(5):
        embeddings, perm_embeddings = _get_sr_embeddings('sr251256', seed)
        _validate_magnitude_embeddings(embeddings)
        _validate_magnitude_embeddings(perm_embeddings)
        _validate_self_iso_on_sr(embeddings, perm_embeddings)

def test_sparse_cin0_self_isomorphism_on_sr261034():
    for seed in range(5):
        embeddings, perm_embeddings = _get_sr_embeddings('sr261034', seed)
        _validate_magnitude_embeddings(embeddings)
        _validate_magnitude_embeddings(perm_embeddings)
        _validate_self_iso_on_sr(embeddings, perm_embeddings)

def test_sparse_cin0_self_isomorphism_on_sr281264():
    for seed in range(5):
        embeddings, perm_embeddings = _get_sr_embeddings('sr281264', seed)
        _validate_magnitude_embeddings(embeddings)
        _validate_magnitude_embeddings(perm_embeddings)
        _validate_self_iso_on_sr(embeddings, perm_embeddings)

def test_sparse_cin0_self_isomorphism_on_sr291467():
    for seed in range(5):
        embeddings, perm_embeddings = _get_sr_embeddings('sr291467', seed)
        _validate_magnitude_embeddings(embeddings)
        _validate_magnitude_embeddings(perm_embeddings)
        _validate_self_iso_on_sr(embeddings, perm_embeddings)

def test_sparse_cin0_self_isomorphism_on_sr351668():
    for seed in range(5):
        embeddings, perm_embeddings = _get_sr_embeddings('sr351668', seed)
        _validate_magnitude_embeddings(embeddings)
        _validate_magnitude_embeddings(perm_embeddings)
        _validate_self_iso_on_sr(embeddings, perm_embeddings)

def test_sparse_cin0_self_isomorphism_on_sr351899():
    for seed in range(5):
        embeddings, perm_embeddings = _get_sr_embeddings('sr351899', seed)
        _validate_magnitude_embeddings(embeddings)
        _validate_magnitude_embeddings(perm_embeddings)
        _validate_self_iso_on_sr(embeddings, perm_embeddings)

def test_sparse_cin0_self_isomorphism_on_sr361446():
    for seed in range(5):
        embeddings, perm_embeddings = _get_sr_embeddings('sr361446', seed)
        _validate_magnitude_embeddings(embeddings)
        _validate_magnitude_embeddings(perm_embeddings)
        _validate_self_iso_on_sr(embeddings, perm_embeddings)

def test_sparse_cin0_self_isomorphism_on_sr401224():
    for seed in range(5):
        embeddings, perm_embeddings = _get_sr_embeddings('sr401224', seed)
        _validate_magnitude_embeddings(embeddings)
        _validate_magnitude_embeddings(perm_embeddings)
        _validate_self_iso_on_sr(embeddings, perm_embeddings)

def test_baseline_self_isomorphism_on_sr16622():
    for seed in range(5):
        embeddings, perm_embeddings = _get_sr_embeddings('sr16622', seed, baseline=True)
        _validate_magnitude_embeddings(embeddings)
        _validate_magnitude_embeddings(perm_embeddings)
        _validate_self_iso_on_sr(embeddings, perm_embeddings)

def test_baseline_self_isomorphism_on_sr251256():
    for seed in range(5):
        embeddings, perm_embeddings = _get_sr_embeddings('sr251256', seed, baseline=True)
        _validate_magnitude_embeddings(embeddings)
        _validate_magnitude_embeddings(perm_embeddings)
        _validate_self_iso_on_sr(embeddings, perm_embeddings)

def test_baseline_self_isomorphism_on_sr261034():
    for seed in range(5):
        embeddings, perm_embeddings = _get_sr_embeddings('sr261034', seed, baseline=True)
        _validate_magnitude_embeddings(embeddings)
        _validate_magnitude_embeddings(perm_embeddings)
        _validate_self_iso_on_sr(embeddings, perm_embeddings)

def test_baseline_self_isomorphism_on_sr281264():
    for seed in range(5):
        embeddings, perm_embeddings = _get_sr_embeddings('sr281264', seed, baseline=True)
        _validate_magnitude_embeddings(embeddings)
        _validate_magnitude_embeddings(perm_embeddings)
        _validate_self_iso_on_sr(embeddings, perm_embeddings)

def test_baseline_self_isomorphism_on_sr291467():
    for seed in range(5):
        embeddings, perm_embeddings = _get_sr_embeddings('sr291467', seed, baseline=True)
        _validate_magnitude_embeddings(embeddings)
        _validate_magnitude_embeddings(perm_embeddings)
        _validate_self_iso_on_sr(embeddings, perm_embeddings)

def test_baseline_self_isomorphism_on_sr351668():
    for seed in range(5):
        embeddings, perm_embeddings = _get_sr_embeddings('sr351668', seed, baseline=True)
        _validate_magnitude_embeddings(embeddings)
        _validate_magnitude_embeddings(perm_embeddings)
        _validate_self_iso_on_sr(embeddings, perm_embeddings)

def test_baseline_self_isomorphism_on_sr351899():
    for seed in range(5):
        embeddings, perm_embeddings = _get_sr_embeddings('sr351899', seed, baseline=True)
        _validate_magnitude_embeddings(embeddings)
        _validate_magnitude_embeddings(perm_embeddings)
        _validate_self_iso_on_sr(embeddings, perm_embeddings)

def test_baseline_self_isomorphism_on_sr361446():
    for seed in range(5):
        embeddings, perm_embeddings = _get_sr_embeddings('sr361446', seed, baseline=True)
        _validate_magnitude_embeddings(embeddings)
        _validate_magnitude_embeddings(perm_embeddings)
        _validate_self_iso_on_sr(embeddings, perm_embeddings)

def test_baseline_self_isomorphism_on_sr401224():
    for seed in range(5):
        embeddings, perm_embeddings = _get_sr_embeddings('sr401224', seed, baseline=True)
        _validate_magnitude_embeddings(embeddings)
        _validate_magnitude_embeddings(perm_embeddings)
        _validate_self_iso_on_sr(embeddings, perm_embeddings)
