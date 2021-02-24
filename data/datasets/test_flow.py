from data.datasets.flow import load_flow_dataset


def test_flow_dataset_loading():
    train, test = load_flow_dataset(num_points=100, num_train=200, num_test=20)
    assert len(train) == 200
    assert len(test) == 20
