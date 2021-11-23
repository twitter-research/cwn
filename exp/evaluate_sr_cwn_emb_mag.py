import os
import sys
import torch
import numpy as np
import random
from definitions import ROOT_DIR

from exp.prepare_sr_tests import prepare
from mp.models import MessagePassingAgnostic, SparseCIN
from data.data_loading import DataLoader, load_dataset

__families__ = [
    'sr16622',
    'sr251256',
    'sr261034',
    'sr281264',
    'sr291467',
    'sr351668',
    'sr351899',
    'sr361446',
    'sr401224'
]

def compute_embeddings(family, baseline, seed):

    # Set the seed for everything
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

    # Perform the check in double precision
    torch.set_default_dtype(torch.float64)

    # Please set the parameters below to the ones used in SR experiments.
    hidden = 16
    num_layers = 3
    max_ring_size = 6
    use_coboundaries = True
    nonlinearity = 'elu'
    graph_norm = 'id'
    readout = 'sum'
    final_readout = 'sum'
    readout_dims = (0,1,2)
    init = 'sum'
    jobs = 64
    device = torch.device("cuda:" + str(0)) if torch.cuda.is_available() else torch.device("cpu")

    # Build and dump dataset if needed
    prepare(family, jobs, max_ring_size, False, init, None)

    # Load reference dataset
    complexes = load_dataset(family, max_dim=2, max_ring_size=max_ring_size, init_method=init)
    data_loader = DataLoader(complexes, batch_size=8, shuffle=False, num_workers=16, max_dim=2)
    
    # Instantiate model
    if not baseline:
        model = SparseCIN(num_input_features=1, num_classes=complexes.num_classes, num_layers=num_layers, hidden=hidden, 
                            use_coboundaries=use_coboundaries, nonlinearity=nonlinearity, graph_norm=graph_norm, 
                            readout=readout, final_readout=final_readout, readout_dims=readout_dims)
    else:
        hidden = 256
        model = MessagePassingAgnostic(num_input_features=1, num_classes=complexes.num_classes, hidden=hidden,
                                        nonlinearity=nonlinearity, readout=readout)
    model = model.to(device)
    model.eval()

    # Compute complex embeddings
    with torch.no_grad():
        embeddings = list()        
        for batch in data_loader:
            batch.nodes.x = batch.nodes.x.double()
            batch.edges.x = batch.edges.x.double()
            batch.two_cells.x = batch.two_cells.x.double()
            out = model.forward(batch.to(device))
            embeddings.append(out)
        embeddings = torch.cat(embeddings, 0)  # n x d
    assert embeddings.size(1) ==  complexes.num_classes

    return embeddings

if __name__ == "__main__":
    
    # Standard args
    passed_args = sys.argv[1:]
    baseline = (passed_args[0].lower() == 'true')
    max_ring_size = int(passed_args[1])
    assert max_ring_size > 3

    # Execute
    msg = f'Model: {"CIN" if not baseline else "MLP-sum"}({max_ring_size})'
    print(msg)
    for family in __families__:
        text = f'\n======================== {family}'
        msg += text+'\n'
        print(text)
        for seed in range(5):
            embeddings = compute_embeddings(family, baseline, seed)
            text = f'seed {seed}: {torch.max(torch.abs(embeddings)):.2f}'
            msg += text+'\n'
            print(text)
    path = os.path.join(ROOT_DIR, 'exp', 'results')
    if baseline:
        path = os.path.join(path, f'sr-base-{max_ring_size}.txt')
    else:
        path = os.path.join(path, f'sr-{max_ring_size}.txt')
    with open(path, 'w') as handle:
        handle.write(msg)
