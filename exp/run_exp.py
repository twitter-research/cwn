import os

import pickle
import torch
import torch.optim as optim

from data.data_loading import DataLoader, load_dataset
from exp.train_utils import train, eval, Evaluator
from exp.parser import get_parser
from mp.models import SIN0, Dummy

from definitions import ROOT_DIR

import time
import numpy as np

# run isomorphism test on sr251256:
# python3 -m exp.run_exp --model dummy --num_layers 1 --dataset sr251256 --untrained

def main(args):

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
    dataset = load_dataset(args.dataset, max_dim=args.max_dim, fold=args.fold)
    split_idx = dataset.get_idx_split()

    # automatic evaluator, takes dataset name as input
    evaluator = Evaluator(args.eval_metric)

    # instantiate data loaders
    train_loader = DataLoader(dataset[split_idx["train"]], batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, max_dim=dataset.max_dim)
    valid_loader = DataLoader(dataset[split_idx["valid"]], batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, max_dim=dataset.max_dim)
    test_split = split_idx.get("test", None)
    if test_split is not None:
        test_loader = DataLoader(dataset[test_split], batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, max_dim=dataset.max_dim)
    else:
        test_loader = None

    # instantiate model
    # NB: here we assume to have the same number of features per dim
    linear_output = (not args.task_type=='classification')
    if args.model == 'sin':
        model = SIN0(dataset.num_features_in_dim(0),          # num_input_features
                     dataset.num_classes,                     # num_classes
                     args.num_layers,                         # num_layers
                     args.emb_dim,                            # hidden
                     dropout_rate=args.drop_rate,             # dropout rate
                     max_dim=dataset.max_dim,                 # max_dim
                     linear_output=linear_output
                    ).to(device)
    elif args.model == 'dummy':
        import pdb; pdb.set_trace()
        model = Dummy(dataset.num_features_in_dim(0),
                      dataset.num_classes,
                      args.num_layers,
                      max_dim=dataset.max_dim,
                      linear_output=linear_output
                     ).to(device)
    else:
        raise ValueError('Invalid model type {}.'.format(args.model))
        
    # instantiate optimiser
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    # instantiate learning rate decay
    if args.lr_scheduler == 'ReduceLROnPlateau':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=args.lr_scheduler_decay_rate, patience=args.lr_scheduler_patience, verbose=True)
    elif args.lr_scheduler == 'StepLR':
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.args.lr_scheduler_decay_steps, gamma=args.lr_scheduler_decay_rate)
    elif args.lr_scheduler == 'None':
        scheduler = None
    else:
        raise NotImplementedError('Scheduler {} is not currently supported.'.format(args.lr_scheduler))

    # (!) start training/evaluation
    valid_curve = []
    test_curve = []
    train_curve = []
    train_loss_curve = []
    if not args.untrained:
        for epoch in range(1, args.epochs + 1):

            # perform one epoch
            print("=====Epoch {}".format(epoch))
            print('Training...')
            train_loss_curve += train(model, device, train_loader, optimizer, args.task_type)
            
            # evaluate model
            print('Evaluating...')
            train_perf = eval(model, device, train_loader, evaluator)
            train_curve.append(train_perf)
            valid_perf = eval(model, device, valid_loader, evaluator)
            valid_curve.append(valid_perf)
            if test_loader is not None:
                test_perf = eval(model, device, test_loader, evaluator)
            else:
                test_perf = np.nan
            test_curve.append(test_perf)
            print({'Train': train_perf, 'Validation': valid_perf, 'Test': test_perf})
            
            # decay learning rate
            if scheduler is not None:
                if args.lr_scheduler == 'ReduceLROnPlateau':
                    scheduler.step(valid_perf)
                else:
                    scheduler.step()

        if not args.minimize:
            best_val_epoch = np.argmax(np.array(valid_curve))
        else:
            best_val_epoch = np.argmin(np.array(valid_curve))
        print('Finished training!')
        print('Best validation score: {}'.format(valid_curve[best_val_epoch]))
        if test_loader is not None:
            print('Test score: {}'.format(test_curve[best_val_epoch]))
        
    else:
        print('Evaluating...')
        train_curve.append(eval(model, device, train_loader, evaluator))
        valid_curve.append(eval(model, device, valid_loader, evaluator))
        if test_loader is not None:
            test_curve.append(eval(model, device, test_loader, evaluator))
        else:
            test_curve.append(np.nan)
        best_val_epoch = 0
        train_loss_curve.append(np.nan)

    # save results
    curves = {
        'train_loss': train_loss_curve,
        'train': train_curve,
        'val': valid_curve,
        'test': test_curve,
        'best': best_val_epoch}
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
            pickle.dump(curves, handle)
            
    return curves

if __name__ == "__main__":

    parser = get_parser()
    args = parser.parse_args()
    
    main(args)
