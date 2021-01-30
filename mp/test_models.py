import torch
import itertools

from data.complex import ComplexBatch
from data.dummy_complexes import get_testing_complex_list
from data.data_loading import DataLoader
from mp.models import SIN0


def test_sin_model_with_batching():
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
        model = SIN0(num_input_features=1, num_classes=3, num_layers=3, hidden=5, jump_mode='cat',
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

        # This is flaky when using equal. I suspect it's because of numerical errors.
        assert (preds.size() == batched_preds.size())
        assert torch.allclose(preds, batched_preds)


def test_sin_model_with_batching_over_complexes_missing_triangles():
    """Check this runs without errors"""
    data_list = get_testing_complex_list()
    data_loader = DataLoader(data_list, batch_size=2)

    # Run using a model that works up to triangles.
    model = SIN0(num_input_features=1, num_classes=3, num_layers=3, hidden=5, max_dim=2,
                 jump_mode='max')
    # We use the model in eval mode to avoid problems with batch norm.
    model.eval()

    preds1 = []
    for batch in data_loader:
        out = model.forward(batch)
        preds1.append(out)
    preds1 = torch.cat(preds1, dim=0)

    # Run using a model that works up to edges.
    model = SIN0(num_input_features=1, num_classes=3, num_layers=3, hidden=5, max_dim=1,
                 jump_mode='max')
    model.eval()

    data_loader = DataLoader(data_list, batch_size=2, max_dim=1)
    preds2 = []
    for batch in data_loader:
        out = model.forward(batch)
        preds2.append(out)
    preds2 = torch.cat(preds2, dim=0)

    # Make sure the two outputs are different. The model using triangles set the triangle outputs
    # to zero, so the output of the readout should also be different.
    assert not torch.equal(preds1, preds2)



