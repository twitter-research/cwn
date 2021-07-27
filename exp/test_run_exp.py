from exp.parser import get_parser
from exp.run_exp import main

def get_args_for_dummym():
    args = list()
    args += ['--use_coboundaries', 'True']
    args += ['--graph_norm', 'id']
    args += ['--lr_scheduler', 'None']
    args += ['--num_layers', '3']
    args += ['--emb_dim', '8']
    args += ['--batch_size', '3']
    args += ['--epochs', '1']
    args += ['--dataset', 'DUMMYM']
    args += ['--max_ring_size', '5']
    args += ['--exp_name', 'dummym_test']
    args += ['--readout_dims', '0', '2']
    return args

def test_run_exp_on_dummym():
    parser = get_parser()
    args = get_args_for_dummym()
    parsed_args = parser.parse_args(args)
    curves = main(parsed_args)
    # On this dataset the splits all coincide; we assert
    # that the final performance is the same on all of them.
    assert curves['last_train'] == curves['last_val']
    assert curves['last_train'] == curves['last_test']