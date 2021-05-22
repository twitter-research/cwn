import torch
import itertools

from data.complex import ComplexBatch
from data.dummy_complexes import get_testing_complex_list
from mp.molec_models import EmbedSparseSIN, OGBEmbedSparseSIN
from data.data_loading import DataLoader, load_dataset


def test_zinc_sparse_sin0_model_with_batching():
    """Check this runs without errors and that batching and no batching produce the same output."""
    data_list = get_testing_complex_list()

    # Try multiple parameters
    dims = [1, 2]
    bs = list(range(2, len(data_list)+1))
    params = itertools.product(bs, dims, dims)
    torch.manual_seed(0)
    for batch_size, batch_max_dim, model_max_dim in params:
        if batch_max_dim > model_max_dim:
            continue

        data_loader = DataLoader(data_list, batch_size=batch_size, max_dim=batch_max_dim)
        model = EmbedSparseSIN(atom_types=32, bond_types=4, out_size=3, num_layers=3, hidden=5,
                               jump_mode='cat', max_dim=model_max_dim)
        # We use the model in eval mode to avoid problems with batch norm.
        model.eval()

        batched_res = {}
        for batch in data_loader:
            # Simulate no edge and triangle features to test init layer
            if len(batch.chains) >= 2:
                batch.chains[1].x = None
            if len(batch.chains) == 3:
                batch.chains[2].x = None

            batched_pred, res = model.forward(batch, include_partial=True)
            for key in res:
                if key not in batched_res:
                    batched_res[key] = []
                batched_res[key].append(res[key])

        for key in batched_res:
            batched_res[key] = torch.cat(batched_res[key], dim=0)

        unbatched_res = {}
        for complex in data_list:
            batch = ComplexBatch.from_complex_list([complex], max_dim=batch_max_dim)

            # Simulate no edge and triangle features to test init layer
            if len(batch.chains) >= 2:
                batch.chains[1].x = None
            if len(batch.chains) == 3:
                batch.chains[2].x = None

            pred, res = model.forward(batch, include_partial=True)
            for key in res:
                if key not in unbatched_res:
                    unbatched_res[key] = []
                unbatched_res[key].append(res[key])

        for key in unbatched_res:
            unbatched_res[key] = torch.cat(unbatched_res[key], dim=0)

        for key in set(list(unbatched_res.keys()) + list(batched_res.keys())):
            assert torch.allclose(unbatched_res[key], batched_res[key], atol=1e-6), (
                    print(key, torch.max(torch.abs(unbatched_res[key] - batched_res[key]))))


def test_zinc_sparse_sin0_model_with_batching_on_proteins():
    """Check this runs without errors and that batching and no batching produce the same output."""
    dataset = load_dataset('PROTEINS', max_dim=2, fold=0, init_method='mean')
    assert len(dataset) == 1113
    split_idx = dataset.get_idx_split()
    dataset = dataset[split_idx['valid']]
    assert len(dataset) == 111

    max_dim = 2
    torch.manual_seed(0)
    data_loader = DataLoader(dataset, batch_size=32, max_dim=max_dim)
    model = EmbedSparseSIN(atom_types=64, bond_types=4, out_size=3, num_layers=3, hidden=5,
                           jump_mode='cat', max_dim=max_dim)
    model.eval()

    batched_res = {}
    for batch in data_loader:
        # Simulate no edge and triangle features to test init layer
        batch.chains[1].x = None
        if len(batch.chains) == 3:
            batch.chains[2].x = None
        # ZincSparseSin assumes features are unidimensional like in ZINC
        batch.chains[0].x = batch.chains[0].x[:, :1]

        batched_pred, res = model.forward(batch, include_partial=True)
        for key in res:
            if key not in batched_res:
                batched_res[key] = []
            batched_res[key].append(res[key])

    for key in batched_res:
        batched_res[key] = torch.cat(batched_res[key], dim=0)

    unbatched_res = {}
    for complex in dataset:
        batch = ComplexBatch.from_complex_list([complex], max_dim=max_dim)
        # Simulate no edge and triangle features to test init layer
        batch.chains[1].x = None
        if len(batch.chains) == 3:
            batch.chains[2].x = None
        # ZincSparseSin assumes features are unidimensional like in ZINC
        batch.chains[0].x = batch.chains[0].x[:, :1]

        pred, res = model.forward(batch, include_partial=True)
        for key in res:
            if key not in unbatched_res:
                unbatched_res[key] = []
            unbatched_res[key].append(res[key])

    for key in unbatched_res:
        unbatched_res[key] = torch.cat(unbatched_res[key], dim=0)

    for key in set(list(unbatched_res.keys()) + list(batched_res.keys())):
        assert torch.allclose(unbatched_res[key], batched_res[key], atol=1e-6), (
                print(key, torch.max(torch.abs(unbatched_res[key] - batched_res[key]))))



def test_ogb_sparse_sin0_model_with_batching():
    """Check this runs without errors and that batching and no batching produce the same output."""
    data_list = get_testing_complex_list()

    # Try multiple parameters
    dims = [1, 2]
    bs = list(range(2, len(data_list)+1))
    params = itertools.product(bs, dims, dims)
    torch.manual_seed(0)
    for batch_size, batch_max_dim, model_max_dim in params:
        if batch_max_dim > model_max_dim:
            continue

        data_loader = DataLoader(data_list, batch_size=batch_size, max_dim=batch_max_dim)
        model = OGBEmbedSparseSIN(out_size=3, num_layers=3, hidden=5,
                                  jump_mode=None, max_dim=model_max_dim)
        # We use the model in eval mode to avoid problems with batch norm.
        model.eval()

        batched_res = {}
        for batch in data_loader:
            # Simulate no edge and triangle features to test init layer
            if len(batch.chains) >= 2:
                batch.chains[1].x = None
            if len(batch.chains) == 3:
                batch.chains[2].x = None

            batched_pred, res = model.forward(batch, include_partial=True)
            for key in res:
                if key not in batched_res:
                    batched_res[key] = []
                batched_res[key].append(res[key])

        for key in batched_res:
            batched_res[key] = torch.cat(batched_res[key], dim=0)


        unbatched_res = {}
        for complex in data_list:
            batch = ComplexBatch.from_complex_list([complex], max_dim=batch_max_dim)

            # Simulate no edge and triangle features to test init layer
            if len(batch.chains) >= 2:
                batch.chains[1].x = None
            if len(batch.chains) == 3:
                batch.chains[2].x = None

            pred, res = model.forward(batch, include_partial=True)

            for key in res:
                if key not in unbatched_res:
                    unbatched_res[key] = []
                unbatched_res[key].append(res[key])

        for key in unbatched_res:
            unbatched_res[key] = torch.cat(unbatched_res[key], dim=0)

        for key in set(list(unbatched_res.keys()) + list(batched_res.keys())):
            assert torch.allclose(unbatched_res[key], batched_res[key], atol=1e-6), (
                    print(key, torch.max(torch.abs(unbatched_res[key] - batched_res[key]))))
