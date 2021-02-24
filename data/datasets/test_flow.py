from data.datasets.flow import load_flow_dataset
import numpy as np


def test_flow_dataset_loading():
    np.random.seed(0)
    train, test = load_flow_dataset(num_points=100, num_train=200, num_test=20)
    assert len(train) == 200
    assert len(test) == 20

    for chain in train + test:
        assert len(chain.upper_orient) == chain.upper_index.size(1)
        assert len(chain.lower_orient) == chain.lower_index.size(1)
        assert chain.upper_index.max() < chain.x.size(0), print(chain.upper_index.max(), chain.x.size(0))
        assert chain.lower_index.max() < chain.x.size(0)
