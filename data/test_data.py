import torch

from data.dummy_complexes import get_house_complex


def test_up_and_down_feature_extraction_on_house_complex():
    house_complex = get_house_complex()

    v_cochain_params = house_complex.get_cochain_params(dim=0)
    v_up_attr = v_cochain_params.kwargs['up_attr']
    expected_v_up_attr = torch.tensor([[1], [1], [4], [4], [2], [2], [3], [3], [6], [6], [5], [5]],
                                      dtype=torch.float)
    assert torch.equal(expected_v_up_attr, v_up_attr)

    e_cochain_params = house_complex.get_cochain_params(dim=1)
    e_up_attr = e_cochain_params.kwargs['up_attr']
    expected_e_up_attr = torch.tensor([[1], [1], [1], [1], [1], [1]], dtype=torch.float)
    assert torch.equal(expected_e_up_attr, e_up_attr)

    e_down_attr = e_cochain_params.kwargs['down_attr']
    expected_e_down_attr = torch.tensor([[2], [2], [1], [1], [3], [3], [3], [3], [4], [4], [4], [4],
                                         [3], [3], [4], [4], [5], [5]], dtype=torch.float)
    assert torch.equal(expected_e_down_attr, e_down_attr)

    t_cochain_params = house_complex.get_cochain_params(dim=2)
    t_up_attr = t_cochain_params.kwargs['up_attr']
    assert t_up_attr is None

    t_down_attr = t_cochain_params.kwargs['down_attr']
    assert t_down_attr is None


def test_get_all_cochain_params_with_max_dim_one_and_no_top_features():
    house_complex = get_house_complex()

    params = house_complex.get_all_cochain_params(max_dim=1, include_top_features=False)
    assert len(params) == 2

    v_cochain_params, e_cochain_params = params

    v_up_attr = v_cochain_params.kwargs['up_attr']
    expected_v_up_attr = torch.tensor([[1], [1], [4], [4], [2], [2], [3], [3], [6], [6], [5], [5]],
                                      dtype=torch.float)
    assert torch.equal(expected_v_up_attr, v_up_attr)

    e_up_attr = e_cochain_params.kwargs['up_attr']
    assert e_up_attr is None
    assert e_cochain_params.up_index is not None
    assert e_cochain_params.up_index.size(1) == 6

    e_down_attr = e_cochain_params.kwargs['down_attr']
    expected_e_down_attr = torch.tensor([[2], [2], [1], [1], [3], [3], [3], [3], [4], [4], [4], [4],
                                         [3], [3], [4], [4], [5], [5]], dtype=torch.float)
    assert torch.equal(expected_e_down_attr, e_down_attr)
