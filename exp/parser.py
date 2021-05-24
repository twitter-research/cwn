import os
import time
import argparse

from definitions import ROOT_DIR


def get_parser():
    parser = argparse.ArgumentParser(description='SCN experiment.')
    parser.add_argument('--seed', type=int, default=43,
                        help='random seed to set (default: 43, i.e. the non-meaning of life))')
    parser.add_argument('--start_seed', type=int, default=0,
                        help='The initial seed when evaluating on multiple seeds.')
    parser.add_argument('--stop_seed', type=int, default=0,
                        help='The final seed when evaluating on multiple seeds.')
    parser.add_argument('--device', type=int, default=9,
                        help='which gpu to use if any (default: 0)')
    parser.add_argument('--model', type=str, default='sparse_sin',
                        help='model, possible choices: sin, dummy, ... (default: sin)')
    parser.add_argument('--use_cofaces', type=str, default='False',
                        help='whether to use coface features for up-messages in sparse_sin (default: False)')
    # ^^^ here we explicitly pass it as string as easier to handle in tuning
    parser.add_argument('--indrop_rate', type=float, default=0.0,
                        help='inputs dropout rate for molec models(default: 0.0)')
    parser.add_argument('--drop_rate', type=float, default=0.5,
                        help='dropout rate (default: 0.5)')
    parser.add_argument('--drop_position', type=str, default='lin2',
                        help='where to apply the final dropout (default: lin2, i.e. _before_ lin2)')
    parser.add_argument('--nonlinearity', type=str, default='relu',
                        help='activation function (default: relu)')
    parser.add_argument('--readout', type=str, default='sum',
                        help='readout function (default: sum)')
    parser.add_argument('--final_readout', type=str, default='sum',
                        help='final readout function (default: sum)')
    parser.add_argument('--jump_mode', type=str, default=None,
                        help='Mode for JK (default: None, i.e. no JK)')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='learning rate (default: 0.001)')
    parser.add_argument('--lr_scheduler', type=str, default='StepLR',
                        help='learning rate decay scheduler (default: StepLR)')
    parser.add_argument('--lr_scheduler_decay_steps', type=int, default=50,
                        help='number of epochs between lr decay (default: 50)')
    parser.add_argument('--lr_scheduler_decay_rate', type=float, default=0.5,
                        help='strength of lr decay (default: 0.5)')
    parser.add_argument('--lr_scheduler_patience', type=float, default=10,
                        help='patience for `ReduceLROnPlateau` lr decay (default: 10)')
    parser.add_argument('--lr_scheduler_min', type=float, default=0.00001,
                        help='min LR for `ReduceLROnPlateau` lr decay (default: 1e-5)')
    parser.add_argument('--num_layers', type=int, default=5,
                        help='number of message passing layers (default: 5)')
    parser.add_argument('--emb_dim', type=int, default=64,
                        help='dimensionality of hidden units in models (default: 300)')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='input batch size for training (default: 32)')
    parser.add_argument('--epochs', type=int, default=100,
                        help='number of epochs to train (default: 100)')
    parser.add_argument('--num_workers', type=int, default=0,
                        help='number of workers (default: 0)')
    parser.add_argument('--dataset', type=str, default="PROTEINS",
                        help='dataset name (default: PROTEINS)')
    parser.add_argument('--task_type', type=str, default='classification',
                        help='task type, either (bin)classification, regression or isomorphism (default: classification)')    
    parser.add_argument('--eval_metric', type=str, default='accuracy',
                        help='evaluation metric (default: accuracy)')
    parser.add_argument('--minimize', action='store_true',
                        help='whether to minimize evaluation metric or not')
    parser.add_argument('--max_dim', type=int, default="2",
                        help='maximum simplicial dimension (default: 2, i.e. triangles)')
    parser.add_argument('--max_ring_size', type=int, default=None,
                        help='maximum ring size to look for (default: None, i.e. do not look for rings)')
    parser.add_argument('--result_folder', type=str, default=os.path.join(ROOT_DIR, 'exp', 'results'),
                        help='filename to output result (default: None, will use `scn/exp/results`)')
    parser.add_argument('--exp_name', type=str, default=str(time.time()),
                        help='name for specific experiment; if not provided, a name based on unix timestamp will be '+\
                        'used. (default: None)')
    parser.add_argument('--dump_curves', action='store_true',
                        help='whether to dump the training curves to disk')
    parser.add_argument('--untrained', action='store_true',
                        help='whether to skip training')
    parser.add_argument('--fold', type=int, default=None,
                        help='fold index for k-fold cross-validation experiments')
    parser.add_argument('--folds', type=int, default=None,
                        help='The number of folds to run on in cross validation experiments')
    parser.add_argument('--init_method', type=str, default='sum',
                        help='How to initialise features at higher levels (sum, mean)')
    parser.add_argument('--train_eval_period', type=int, default=10,
                        help='How often to evaluate on train.')
    parser.add_argument('--tune', action='store_true', help='Use the tuning indexes')
    parser.add_argument('--flow_points',  type=int, default=400,
                        help='Number of points to use for the flow experiment')
    parser.add_argument('--flow_classes',  type=int, default=3,
                        help='Number of classes for the flow experiment')
    parser.add_argument('--use_edge_features', action='store_true',
                        help="Use edge features for molecular graphs")
    parser.add_argument('--simple_features', action='store_true',
                        help="Whether to use only a subset of original features, specific to ogb-mol*")
    parser.add_argument('--early_stop', action='store_true', help='Stop when minimum LR is reached.')
    parser.add_argument('--paraid',  type=int, default=0,
                        help='model id')
    parser.add_argument('--preproc_jobs',  type=int, default=2,
                        help='Jobs to use for the dataset preprocessing. For all jobs use "-1".'
                             'For sequential processing (no parallelism) use "1"')
    return parser


def validate_args(args):
    # TODO(Cris): Add more detailed validation for more datasets in the future
    if args.dataset == 'CSL':
        assert args.model == 'embed_sparse_sin'
        assert args.task_type == 'classification'
        assert not args.minimize
        assert args.lr_scheduler == 'ReduceLROnPlateau'
        assert args.eval_metric == 'accuracy'
        assert args.fold is not None
        assert not args.simple_features
    elif args.dataset == 'RINGTREE':
        assert args.model == 'ringtree_sparse_sin' or args.model == 'gin_ringtree'
        assert args.task_type == 'classification'
        assert not args.minimize
        assert args.lr_scheduler == 'None'
        assert args.eval_metric == 'accuracy'
        assert args.fold is None
        assert not args.simple_features
        assert args.max_ring_size is not None and args.max_ring_size > 3
    elif args.dataset.startswith('ZINC'):
        assert args.model == 'embed_sparse_sin'
        assert args.task_type == 'regression'
        assert args.minimize
        assert args.eval_metric == 'mae'
        assert args.lr_scheduler == 'ReduceLROnPlateau'
        assert not args.simple_features
    elif args.dataset in ['MOLHIV', 'MOLPCBA', 'MOLTOX21', 'MOLTOXCAST', 'MOLMUV',
                          'MOLBACE', 'MOLBBBP', 'MOLCLINTOX', 'MOLSIDER', 'MOLESOL',
                          'MOLFREESOLV', 'MOLLIPO']:
        assert args.model == 'ogb_embed_sparse_sin'
        assert args.eval_metric == 'ogbg-'+args.dataset.lower()
        assert args.jump_mode is None
        if args.dataset in ['MOLESOL', 'MOLFREESOLV', 'MOLLIPO']:
            assert args.task_type == 'mse_regression'
            assert args.minimize
        else:
            assert args.task_type == 'bin_classification'
            assert not args.minimize
