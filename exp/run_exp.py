import os

import torch
import torch.optim as optim

from data.data_loading import DataLoader, load_dataset
from exp.train_utils import train, eval, Evaluator
from mp.models import SIN0, Dummy

from definitions import ROOT_DIR

import argparse
import time
import numpy as np

# run isomorphism test on sr251256:
# python3 -m exp.run_exp --model dummy --num_layers 1 --dataset sr251256 --untrained

def main():
    
    # training settings
    parser = argparse.ArgumentParser(description='SCN experiment.')
    
    parser.add_argument('--device', type=int, default=0,
                        help='which gpu to use if any (default: 0)')
    parser.add_argument('--model', type=str, default='sin',
                        help='model, possible choices: sin, ... (default: sin)')
    parser.add_argument('--drop_rate', type=float, default=0.5,
                        help='dropout rate (default: 0.5)')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='learning rate (default: 0.001)')
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
    parser.add_argument('--dataset', type=str, default="sr251256",
                        help='dataset name (default: sr251256)')
    parser.add_argument('--result_folder', type=str, default=None,
                        help='filename to output result (default: None, will use `scn/exp/results`)')
    parser.add_argument('--dump_curves', action='store_true',
                        help='whether to dump the training curves to disk.')
    parser.add_argument('--untrained', action='store_true',
                        help='whether to skip training')
    args = parser.parse_args()

    # set device
    device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")
    
    # get timestamp for results and set result directory
    ts = time.time()
    result_folder = args.result_folder if args.result_folder is not None else os.path.join(ROOT_DIR, 'exp', 'results')
    result_folder = os.path.join(result_folder, '{}-{}'.format(args.dataset, ts))
    if not os.path.exists(result_folder):
        os.makedirs(result_folder)
    filename = os.path.join(result_folder, 'results.txt')
    
    # data loading
    dataset = load_dataset(args.dataset)
    split_idx = dataset.get_idx_split()

    # automatic evaluator, takes dataset name as input
    evaluator = Evaluator(dataset.eval_metric)

    # instantiate data loaders
    train_loader = DataLoader(dataset[split_idx["train"]], batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, max_dim=dataset.max_dim)
    valid_loader = DataLoader(dataset[split_idx["valid"]], batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, max_dim=dataset.max_dim)
    test_loader = DataLoader(dataset[split_idx["test"]], batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, max_dim=dataset.max_dim)

    # instantiate model
    # NB: here we assume to have the same number of features per dim
    linear_output = (not dataset.task_type=='classification')
    if args.model == 'sin':
        model = SIN0(dataset.num_features(0),                 # num_input_features
                     dataset.num_classes,                     # num_classes
                     args.num_layers,                         # num_layers
                     args.emb_dim,                            # hidden
                     dropout_rate=args.drop_rate,             # dropout rate
                     max_dim=dataset.max_dim,                 # max_dim
                     linear_output=linear_output
                    ).to(device)
    elif args.model == 'dummy':
        model = Dummy(dataset.num_features(0),
                      dataset.num_classes,
                      args.num_layers,
                      max_dim=dataset.max_dim,
                      linear_output=linear_output
                     ).to(device)
    else:
        raise ValueError('Invalid model type {}.'.format(args.model))
        
    # instantiate optimiser
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # (!) start training/evaluation
    valid_curve = []
    test_curve = []
    train_curve = []
    train_loss_curve = []
    if not args.untrained:
        assert dataset.maximize is not None
        for epoch in range(1, args.epochs + 1):

            print("=====Epoch {}".format(epoch))
            print('Training...')
            train_loss_curve += train(model, device, train_loader, optimizer, dataset.task_type)

            print('Evaluating...')
            train_perf = eval(model, device, train_loader, evaluator)
            valid_perf = eval(model, device, valid_loader, evaluator)
            test_perf = eval(model, device, test_loader, evaluator)

            print({'Train': train_perf, 'Validation': valid_perf, 'Test': test_perf})

            train_curve.append(train_perf[dataset.eval_metric])
            valid_curve.append(valid_perf[dataset.eval_metric])
            test_curve.append(test_perf[dataset.eval_metric])

        if dataset.maximize:
            best_val_epoch = np.argmax(np.array(valid_curve))
        else:
            best_val_epoch = np.argmin(np.array(valid_curve))
        print('Finished training!')
        print('Best validation score: {}'.format(valid_curve[best_val_epoch]))
        print('Test score: {}'.format(test_curve[best_val_epoch]))
        
    else:
        print('Evaluating...')
        train_curve.append(eval(model, device, train_loader, evaluator))
        valid_curve.append(eval(model, device, valid_loader, evaluator))
        test_curve.append(eval(model, device, test_loader, evaluator))
        best_val_epoch = 0
        train_loss_curve.append(np.nan)

    # save results
    msg = (
        'Dataset:        {0}\n'
        'Validation:     {1}\n'
        'Test:           {2}\n'
        'Train:          {3}\n'
        'Best epoch:     {4}\n'
        '-------------------------------\n')
    msg = msg.format(args.dataset, valid_curve[best_val_epoch], test_curve[best_val_epoch], train_curve[best_val_epoch], best_val_epoch)
    msg += str(args)
    with open(filename, 'w') as handle:
        handle.write(msg)
    if args.dump_curves:
        with open(result_path+'curves.pkl', 'wb') as handle:
            pickle.dump(
                {
                    'train_loss': train_loss_curve,
                    'train': train_curve,
                    'val': valid_curve,
                    'test': test_curve,
                    'best': best_val_epoch
                }, handle)

if __name__ == "__main__":
    main()