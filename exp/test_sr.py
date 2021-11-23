import torch
import numpy as np
import random
import pytest

from data.data_loading import DataLoader, load_dataset
from exp.prepare_sr_tests import prepare
from mp.models import MessagePassingAgnostic, SparseCIN

def _get_cwn_sr_embeddings(family, seed, baseline=False):

    # Set the seed for everything
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

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
    # The fact that we work in such a (safe) range can also be verified by running the following:
    #   a = torch.DoubleTensor([2.5e8])
    #   d = torch.DoubleTensor([5.0e8])
    #   b = torch.nextafter(a, d)
    #   print(b - a)
    #   >>> tensor([2.9802e-08], dtype=torch.float64)
    thresh = torch.DoubleTensor([5.0*1e8])
    apex = torch.max(torch.abs(embeddings)).cpu()
    print(apex)
    assert apex.dtype == torch.float64
    assert torch.all(apex < thresh)

@pytest.mark.slow
@pytest.mark.parametrize("family", ['sr16622', 'sr251256', 'sr261034', 'sr281264', 'sr291467', 'sr351668', 'sr351899', 'sr361446', 'sr401224'])
def test_sparse_cin0_self_isomorphism(family):
    # Perform the check in double precision
    torch.set_default_dtype(torch.float64)
    for seed in range(5):
        embeddings, perm_embeddings = _get_cwn_sr_embeddings(family, seed)
        _validate_magnitude_embeddings(embeddings)
        _validate_magnitude_embeddings(perm_embeddings)
        _validate_self_iso_on_sr(embeddings, perm_embeddings)
    # Revert back to float32 for other tests
    torch.set_default_dtype(torch.float32)

@pytest.mark.slow
@pytest.mark.parametrize("family", ['sr16622', 'sr251256', 'sr261034', 'sr281264', 'sr291467', 'sr351668', 'sr351899', 'sr361446', 'sr401224'])
def test_cwn_baseline_self_isomorphism(family):
    # Perform the check in double precision
    torch.set_default_dtype(torch.float64)
    for seed in range(5):
        embeddings, perm_embeddings = _get_cwn_sr_embeddings(family, seed, baseline=True)
        _validate_magnitude_embeddings(embeddings)
        _validate_magnitude_embeddings(perm_embeddings)
        _validate_self_iso_on_sr(embeddings, perm_embeddings)
    # Revert back to float32 for other tests
    torch.set_default_dtype(torch.float32)
