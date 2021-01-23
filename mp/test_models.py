import torch

from data.complex import ComplexBatch
from data.dummy_complexes import get_house_complex, get_square_complex, get_pyramid_complex
from data.data_loading import DataLoader
from mp.models import SIN0


def test_sin_model_with_batching():
    data_list = [get_house_complex(), get_square_complex(),
                 get_square_complex(), get_house_complex()]

    data_loader = DataLoader(data_list, batch_size=2)

    model = SIN0(num_input_features=1, num_classes=3, num_layers=3, hidden=5)
    model.eval()
    
    for batch in data_loader:
        model.forward(batch)


