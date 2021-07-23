import torch
import pytest
import itertools

from data.complex import ComplexBatch
from data.dummy_complexes import get_testing_complex_list
from mp.models import CIN0, EdgeCIN0, SparseCIN
from data.data_loading import DataLoader, load_dataset


def test_cin_model_with_batching():
    """Check this runs without errors and that batching and no batching produce the same output."""
    data_list = get_testing_complex_list()

    # Try multiple parameters
    dims = [1, 2, 3]
    bs = list(range(2, len(data_list)+1))
    params = itertools.product(bs, dims, dims)
    for batch_size, batch_max_dim, model_max_dim in params:
        if batch_max_dim > model_max_dim:
            continue

        data_loader = DataLoader(data_list, batch_size=batch_size, max_dim=batch_max_dim)
        model = CIN0(num_input_features=1, num_classes=3, num_layers=3, hidden=5, jump_mode='cat',
                     max_dim=model_max_dim)
        # We use the model in eval mode to avoid problems with batch norm.
        model.eval()

        batched_preds = []
        for batch in data_loader:
            batched_pred = model.forward(batch)
            batched_preds.append(batched_pred)
        batched_preds = torch.cat(batched_preds, dim=0)

        preds = []
        for complex in data_list:
            pred = model.forward(ComplexBatch.from_complex_list([complex], max_dim=batch_max_dim))
            preds.append(pred)
        preds = torch.cat(preds, dim=0)

        # Atol was reduced from 1e-6 to 1e-5 to remove flakiness.
        assert (preds.size() == batched_preds.size())
        assert torch.allclose(preds, batched_preds, atol=1e-5)


def test_edge_cin0_model_with_batching():
    """Check this runs without errors and that batching and no batching produce the same output."""
    data_list = get_testing_complex_list()

    for top_features in [True, False]:
        data_loader = DataLoader(data_list, batch_size=4)
        model = EdgeCIN0(num_input_features=1, num_classes=3, num_layers=3, hidden=5,
                         jump_mode='cat', include_top_features=top_features)
        # We use the model in eval mode to avoid problems with batch norm.
        model.eval()

        batched_preds = []
        for batch in data_loader:
            batched_pred = model.forward(batch)
            batched_preds.append(batched_pred)
        batched_preds = torch.cat(batched_preds, dim=0)

        preds = []
        for complex in data_list:
            pred = model.forward(ComplexBatch.from_complex_list([complex]))
            preds.append(pred)
        preds = torch.cat(preds, dim=0)

        assert torch.allclose(preds, batched_preds, atol=1e-6)


def test_edge_cin0_model_with_batching_while_including_top_features_and_max_dim_one():
    """Check this runs without errors and that batching and no batching produce the same output."""
    data_list = get_testing_complex_list()

    data_loader = DataLoader(data_list, batch_size=4)

    model1 = EdgeCIN0(num_input_features=1, num_classes=3, num_layers=3, hidden=5,
                      jump_mode='cat', include_top_features=True)
    # We use the model in eval mode to avoid problems with batch norm.
    model1.eval()

    batched_preds = []
    for batch in data_loader:
        batched_pred = model1.forward(batch)
        batched_preds.append(batched_pred)
    batched_preds1 = torch.cat(batched_preds, dim=0)

    model2 = EdgeCIN0(num_input_features=1, num_classes=3, num_layers=3, hidden=5,
                      jump_mode='cat', include_top_features=False)
    # We use the model in eval mode to avoid problems with batch norm.
    model2.eval()

    batched_preds = []
    for batch in data_loader:
        batched_pred = model2.forward(batch)
        batched_preds.append(batched_pred)
    batched_preds2 = torch.cat(batched_preds, dim=0)

    # Check excluding the top features providea a different
    # output compared to the model that includes them.
    assert not torch.equal(batched_preds1, batched_preds2)


def test_cin_model_with_batching_over_complexes_missing_two_cells():
    """Check this runs without errors"""
    data_list = get_testing_complex_list()
    data_loader = DataLoader(data_list, batch_size=2)

    # Run using a model that works up to two_cells.
    model = CIN0(num_input_features=1, num_classes=3, num_layers=3, hidden=5, max_dim=2,
                 jump_mode='max')
    # We use the model in eval mode to avoid problems with batch norm.
    model.eval()

    preds1 = []
    for batch in data_loader:
        out = model.forward(batch)
        preds1.append(out)
    preds1 = torch.cat(preds1, dim=0)

    # Run using a model that works up to edges.
    model = CIN0(num_input_features=1, num_classes=3, num_layers=3, hidden=5, max_dim=1,
                 jump_mode='max')
    model.eval()

    data_loader = DataLoader(data_list, batch_size=2, max_dim=1)
    preds2 = []
    for batch in data_loader:
        out = model.forward(batch)
        preds2.append(out)
    preds2 = torch.cat(preds2, dim=0)

    # Make sure the two outputs are different. The model using two_cells set the two_cell outputs
    # to zero, so the output of the readout should also be different.
    assert not torch.equal(preds1, preds2)


def test_sparse_cin0_model_with_batching():
    """Check this runs without errors and that batching and no batching produce the same output."""
    data_list = get_testing_complex_list()

    # Try multiple parameters
    dims = [1, 2, 3]
    bs = list(range(2, len(data_list)+1))
    params = itertools.product(bs, dims, dims)
    torch.manual_seed(0)
    for batch_size, batch_max_dim, model_max_dim in params:
        if batch_max_dim > model_max_dim:
            continue

        data_loader = DataLoader(data_list, batch_size=batch_size, max_dim=batch_max_dim)
        model = SparseCIN(num_input_features=1, num_classes=3, num_layers=3, hidden=5,
                          jump_mode='cat', max_dim=model_max_dim)
        # We use the model in eval mode to avoid problems with batch norm.
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
        for complex in data_list:
            # print(f"Complex dim {complex.dimension}")
            pred, res = model.forward(ComplexBatch.from_complex_list([complex],
                max_dim=batch_max_dim),
                include_partial=True)
            for key in res:
                if key not in unbatched_res:
                    unbatched_res[key] = []
                unbatched_res[key].append(res[key])

        for key in unbatched_res:
            unbatched_res[key] = torch.cat(unbatched_res[key], dim=0)

        for key in set(list(unbatched_res.keys()) + list(batched_res.keys())):
            assert torch.allclose(unbatched_res[key], batched_res[key], atol=1e-6), (
                    print(key, torch.max(torch.abs(unbatched_res[key] - batched_res[key]))))


@pytest.mark.data
def test_sparse_cin0_model_with_batching_on_proteins():
    """Check this runs without errors and that batching and no batching produce the same output."""
    dataset = load_dataset('PROTEINS', max_dim=3, fold=0, init_method='mean')
    assert len(dataset) == 1113
    split_idx = dataset.get_idx_split()
    dataset = dataset[split_idx['valid']]
    assert len(dataset) == 111

    max_dim = 3
    torch.manual_seed(0)
    data_loader = DataLoader(dataset, batch_size=32, max_dim=max_dim)
    model = SparseCIN(num_input_features=dataset.num_features_in_dim(0),
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
        assert torch.allclose(unbatched_res[key], batched_res[key], atol=1e-6), (
            print(key, torch.max(torch.abs(unbatched_res[key] - batched_res[key]))))


