import torch

from data.complex import ComplexBatch
from data.dummy_complexes import (get_house_complex, get_square_complex, get_pyramid_complex,
                                  get_kite_complex, get_square_dot_complex)
from data.data_loading import DataLoader
from mp.models import SIN0


def test_sin_model_with_batching():
    """Check this runs without errors and that batching and no batching produce the same output."""
    data_list = [get_house_complex(), get_kite_complex(), get_square_complex(),
                 get_square_dot_complex(), get_square_complex(),
                 get_house_complex(), get_pyramid_complex(),
                 get_kite_complex(), get_square_dot_complex()]

    data_loader = DataLoader(data_list, batch_size=3)

    model = SIN0(num_input_features=1, num_classes=3, num_layers=3, hidden=5, jump_mode='cat')
    # We use the model in eval mode to avoid problems with batch norm.
    model.eval()

    batched_preds = []
    i = 1
    for batch in data_loader:
        print(i)
        i += 1
        batched_pred = model.forward(batch)
        batched_preds.append(batched_pred)
    batched_preds = torch.cat(batched_preds, dim=0)

    preds = []
    for complex in data_list:
        pred = model.forward(ComplexBatch.from_complex_list([complex]))
        preds.append(pred)
    preds = torch.cat(preds, dim=0)

    # This is flaky when using equal. I suspect it's because of numerical errors.
    assert torch.allclose(preds, batched_preds)


def test_sin_model_with_batching_over_complexes_missing_triangles():
    """Check this runs without errors"""
    data_list = [get_house_complex(), get_kite_complex(), get_square_complex(),
                 get_square_dot_complex(), get_square_complex(),
                 get_house_complex(), get_pyramid_complex(),
                 get_kite_complex(), get_square_dot_complex()]
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



