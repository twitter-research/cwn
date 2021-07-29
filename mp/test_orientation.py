import torch
import numpy as np

from data.datasets.flow import load_flow_dataset
from mp.models import EdgeOrient, EdgeMPNN
from mp.layers import OrientedConv
from data.complex import CochainBatch
from data.data_loading import DataLoader
from data.datasets.flow_utils import build_cochain


def generate_oriented_flow_pair():
    # This is the complex from slide 19 of https://crisbodnar.github.io/files/mml_talk.pdf
    B1 = np.array([
        [-1, -1,  0,  0,  0,  0],
        [+1,  0, -1,  0,  0, +1],
        [ 0, +1,  0, -1,  0, -1],
        [ 0,  0, +1, +1, -1,  0],
        [ 0,  0,  0,  0, +1,  0],
    ])

    B2 = np.array([
        [-1,  0],
        [+1,  0],
        [ 0, +1],
        [ 0, -1],
        [ 0,  0],
        [+1, +1],
    ])

    x = np.array([[1.0], [0.0], [0.0], [1.0], [1.0], [-1.0]])
    id = np.identity(x.shape[0])
    T2 = np.diag([+1.0, +1.0, +1.0, +1.0, -1.0, -1.0])

    cochain1 = build_cochain(B1, B2, id, x, 0)
    cochain2 = build_cochain(B1, B2, T2, x, 0)
    return cochain1, cochain2, torch.tensor(T2, dtype=torch.float)


def test_edge_orient_model_on_flow_dataset_with_batching():
    dataset, _, _ = load_flow_dataset(num_points=100, num_train=20, num_test=2)

    np.random.seed(4)
    data_loader = DataLoader(dataset, batch_size=16)
    model = EdgeOrient(num_input_features=1, num_classes=2, num_layers=2, hidden=5)
    # We use the model in eval mode to test its inference behavior.
    model.eval()

    batched_preds = []
    for batch in data_loader:
        batched_pred = model.forward(batch)
        batched_preds.append(batched_pred)
    batched_preds = torch.cat(batched_preds, dim=0)

    preds = []
    for cochain in dataset:
        pred = model.forward(CochainBatch.from_cochain_list([cochain]))
        preds.append(pred)
    preds = torch.cat(preds, dim=0)

    assert (preds.size() == batched_preds.size())
    assert torch.allclose(preds, batched_preds, atol=1e-5)


def test_edge_orient_conv_is_orientation_equivariant():
    cochain1, cochain2, T2 = generate_oriented_flow_pair()
    assert torch.equal(cochain1.lower_index, cochain2.lower_index)
    assert torch.equal(cochain1.upper_index, cochain2.upper_index)

    layer = OrientedConv(dim=1, up_msg_size=1, down_msg_size=1, update_up_nn=None,
        update_down_nn=None, update_nn=None, act_fn=None)

    out_up1, out_down1, _ = layer.propagate(cochain1.upper_index, cochain1.lower_index, None, x=cochain1.x,
            up_attr=cochain1.upper_orient.view(-1, 1), down_attr=cochain1.lower_orient.view(-1, 1))
    out_up2, out_down2, _ = layer.propagate(cochain2.upper_index, cochain2.lower_index, None, x=cochain2.x,
            up_attr=cochain2.upper_orient.view(-1, 1), down_attr=cochain2.lower_orient.view(-1, 1))

    assert torch.equal(T2 @ out_up1, out_up2)
    assert torch.equal(T2 @ out_down1, out_down2)
    assert torch.equal(T2 @ (cochain1.x + out_up1 + out_down1), cochain2.x + out_up2 + out_down2)


def test_edge_orient_model_with_tanh_is_orientation_equivariant_and_invariant_at_readout():
    cochain1, cochain2, T2 = generate_oriented_flow_pair()
    assert torch.equal(cochain1.lower_index, cochain2.lower_index)
    assert torch.equal(cochain1.upper_index, cochain2.upper_index)

    model = EdgeOrient(num_input_features=1, num_classes=2, num_layers=2, hidden=5,
        nonlinearity='tanh', dropout_rate=0.0)
    model.eval()

    final1, pred1 = model.forward(CochainBatch.from_cochain_list([cochain1]), include_partial=True)
    final2, pred2 = model.forward(CochainBatch.from_cochain_list([cochain2]), include_partial=True)
    # Check equivariant.
    assert torch.equal(T2 @ pred1, pred2)
    # Check invariant after readout.
    assert torch.equal(final1, final2)


def test_edge_orient_model_with_id_is_orientation_equivariant_and_invariant_at_readout():
    cochain1, cochain2, T2 = generate_oriented_flow_pair()
    assert torch.equal(cochain1.lower_index, cochain2.lower_index)
    assert torch.equal(cochain1.upper_index, cochain2.upper_index)

    model = EdgeOrient(num_input_features=1, num_classes=2, num_layers=2, hidden=5,
        nonlinearity='id', dropout_rate=0.0)
    model.eval()

    final1, pred1 = model.forward(CochainBatch.from_cochain_list([cochain1]), include_partial=True)
    final2, pred2 = model.forward(CochainBatch.from_cochain_list([cochain2]), include_partial=True)
    # Check equivariant.
    assert torch.equal(T2 @ pred1, pred2)
    # Check invariant after readout.
    assert torch.equal(final1, final2)


def test_edge_orient_model_with_relu_is_not_orientation_equivariant_or_invariant():
    cochain1, cochain2, T2 = generate_oriented_flow_pair()
    assert torch.equal(cochain1.lower_index, cochain2.lower_index)
    assert torch.equal(cochain1.upper_index, cochain2.upper_index)

    model = EdgeOrient(num_input_features=1, num_classes=2, num_layers=2, hidden=5,
        nonlinearity='relu', dropout_rate=0.0)
    model.eval()

    _, pred1 = model.forward(CochainBatch.from_cochain_list([cochain1]), include_partial=True)
    _, pred2 = model.forward(CochainBatch.from_cochain_list([cochain2]), include_partial=True)
    # Check not equivariant.
    assert not torch.equal(T2 @ pred1, pred2)
    # Check not invariant.
    assert not torch.equal(pred1, pred2)


def test_edge_mpnn_model_is_orientation_invariant():
    cochain1, cochain2, T2 = generate_oriented_flow_pair()
    assert torch.equal(cochain1.lower_index, cochain2.lower_index)
    assert torch.equal(cochain1.upper_index, cochain2.upper_index)

    model = EdgeMPNN(num_input_features=1, num_classes=2, num_layers=2, hidden=5,
        nonlinearity='id', dropout_rate=0.0)
    model.eval()

    _, pred1 = model.forward(CochainBatch.from_cochain_list([cochain1]), include_partial=True)
    _, pred2 = model.forward(CochainBatch.from_cochain_list([cochain2]), include_partial=True)

    # Check the model is orientation invariant.
    assert torch.equal(pred1, pred2)
