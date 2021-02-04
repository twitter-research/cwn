import argparse

def get_parser():
    parser = argparse.ArgumentParser(description='SCN experiment.')
    parser.add_argument('--device', type=int, default=0,
                        help='which gpu to use if any (default: 0)')
    parser.add_argument('--model', type=str, default='sin',
                        help='model, possible choices: sin, dummy, ... (default: sin)')
    parser.add_argument('--drop_rate', type=float, default=0.5,
                        help='dropout rate (default: 0.5)')
    parser.add_argument('--nonlinearity', type=str, default='relu',
                        help='activation function (default: relu)')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='learning rate (default: 0.001)')
    parser.add_argument('--lr_scheduler', type=str, default='None',
                        help='learning rate decay scheduler (default: None)')
    parser.add_argument('--lr_scheduler_decay_steps', type=int, default=50,
                        help='number of epochs between lr decay (default: 50)')
    parser.add_argument('--lr_scheduler_decay_rate', type=float, default=0.5,
                        help='strength of lr decay (default: 0.5)')
    parser.add_argument('--lr_scheduler_patience', type=float, default=10,
                        help='patience for `ReduceLROnPlateau` lr decay (default: 10)')
    parser.add_argument('--num_layers', type=int, default=5,
                        help='number of message passing layers (default: 5)')
    parser.add_argument('--emb_dim', type=int, default=300,
                        help='dimensionality of hidden units in models (default: 300)')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='input batch size for training (default: 32)')
    parser.add_argument('--epochs', type=int, default=100,
                        help='number of epochs to train (default: 100)')
    parser.add_argument('--num_workers', type=int, default=0,
                        help='number of workers (default: 0)')
    parser.add_argument('--dataset', type=str, default="IMDBBINARY",
                        help='dataset name (default: IMDBBINARY)')
    parser.add_argument('--task_type', type=str, default='classification',
                        help='task type, either classification, regression or isomorphism (default: classification)')    
    parser.add_argument('--eval_metric', type=str, default='accuracy',
                        help='evaluation metric (default: accuracy)')
    parser.add_argument('--minimize', action='store_true',
                        help='whether to minimize evaluation metric or not')
    parser.add_argument('--max_dim', type=int, default="2",
                        help='maximum simplicial dimension (default: 2, i.e. triangles)')
    parser.add_argument('--result_folder', type=str, default=None,
                        help='filename to output result (default: None, will use `scn/exp/results`)')
    parser.add_argument('--exp_name', type=str, default=None,
                        help='name for specific experiment; if not provided, a name based on unix timestamp will be '+\
                        'used. (default: None)')
    parser.add_argument('--dump_curves', action='store_true',
                        help='whether to dump the training curves to disk')
    parser.add_argument('--untrained', action='store_true',
                        help='whether to skip training')
    parser.add_argument('--fold', type=int, default=None,
                        help='fold index for k-fold cross-validation experiments')
    return parser
