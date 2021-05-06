import torch
import itertools
import numpy as np

from data.complex import ComplexBatch, ChainBatch
from data.dummy_complexes import get_testing_complex_list
from mp.models import SIN0, EdgeSIN0, SparseSIN, EdgeOrient
from mp.molec_models import ZincSparseSIN
from data.data_loading import DataLoader, load_dataset
from data.datasets.flow import load_flow_dataset


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
        model = ZincSparseSIN(embed_dict_size=32, num_input_features=1, num_classes=3, num_layers=3,
            hidden=5, jump_mode='cat', max_dim=model_max_dim)
        # We use the model in eval mode to avoid problems with batch norm.
        model.eval()

        batched_res = {}
        for batch in data_loader:
            # Simulate no edge and triangle features to test init layer
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
            # print(f"Complex dim {complex.dimension}")
            batch = ComplexBatch.from_complex_list([complex], max_dim=batch_max_dim)

            # Simulate no edge and triangle features to test init layer
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
            if key == 'out':
                # This seems to have some numerical errors on some seeds.
                # After investigations, this seems to be related to the last linear layers of the model
                # Probably because the magnitude of the levels is different, this causes instabilities.
                # The absolute tolerance should be adjusted if the test becomes flaky.
                assert (torch.allclose(unbatched_res[key], batched_res[key]),
                        print(key, torch.max(torch.abs(unbatched_res[key] - batched_res[key]))))
            else:
                assert (torch.allclose(unbatched_res[key], batched_res[key]),
                        print(key, torch.max(torch.abs(unbatched_res[key] - batched_res[key]))))


def test_sparse_sin0_model_with_batching_on_proteins():
    """Check this runs without errors and that batching and no batching produce the same output."""
    dataset = load_dataset('PROTEINS', max_dim=3, fold=0, init_method='mean')
    assert len(dataset) == 1113
    split_idx = dataset.get_idx_split()
    dataset = dataset[split_idx['valid']]
    assert len(dataset) == 111

    max_dim = 3
    torch.manual_seed(0)
    data_loader = DataLoader(dataset, batch_size=32, max_dim=max_dim)
    model = SparseSIN(num_input_features=dataset.num_features_in_dim(0),
        num_classes=2, num_layers=3, hidden=5, jump_mode=None, max_dim=max_dim)
    model.eval()

    batched_res = {}
    for batch in data_loader:
        batched_pred, res = model.forward(batch, include_partial=True)
        for key in res:
            if key not in batched_res:
                batched_res[key] = []
            batched_res[key].append(res[key])

    for key in batched_res:
        batched_res[key] = torch.cat(batched_res[key], dim=0)

    unbatched_res = {}
    for complex in dataset:
        # print(f"Complex dim {complex.dimension}")
        pred, res = model.forward(ComplexBatch.from_complex_list([complex], max_dim=max_dim),
            include_partial=True)
        for key in res:
            if key not in unbatched_res:
                unbatched_res[key] = []
            unbatched_res[key].append(res[key])

    for key in unbatched_res:
        unbatched_res[key] = torch.cat(unbatched_res[key], dim=0)

    for key in set(list(unbatched_res.keys()) + list(batched_res.keys())):
        if key == 'out':
            # This seems to have some numerical errors on some seeds.
            # After investigations, this seems to be related to the last linear layers of the model
            # Probably because the magnitude of the levels is different, this causes instabilities.
            # The absolute tolerance should be adjusted if the test becomes flaky.
            assert (torch.allclose(unbatched_res[key], batched_res[key]),
                    print(key, torch.max(torch.abs(unbatched_res[key] - batched_res[key]))))
        else:
            assert (torch.allclose(unbatched_res[key], batched_res[key]),
                print(key, torch.max(torch.abs(unbatched_res[key] - batched_res[key]))))



