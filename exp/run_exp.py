import torch
import torch.optim as optim

from mp.models import SIN
from data.data_loading import DataLoader
from exp.train_utils import load_data, train, eval, Evaluator

from definitions import ROOT_DIR

import argparse
import time
import numpy as np

def main():
    
    # Training settings
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
    parser.add_argument('--dataset', type=str, default="synth-SR",
                        help='dataset name (default: synth-SR)')
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
    result_folder = args.result_folder if result_folder is non None else os.path.join(ROOT_DIR, 'exp', 'results')
    result_folder = os.path.join(result_folder, '{}-{}'.format(dataset_name, ts))
    if not os.path.exists(result_folder):
        os.makedirs(result_folder)
    filename = os.path.join(result_folder, 'results.txt')
    
    # data loading
    dataset = load_dataset(name=args.dataset)
    # TODO: shall we just load the dataset into the three different splits or keep the following?
    split_idx = dataset.get_idx_split()

    ### automatic evaluator. takes dataset name as input
    # TODO: here we could just instantiate a std one, say one that computes PR-AUC
    evaluator = Evaluator(dataset.eval_metric)

    # instantiate data loaders
    train_loader = DataLoader(dataset[split_idx["train"]], batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    valid_loader = DataLoader(dataset[split_idx["valid"]], batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    test_loader = DataLoader(dataset[split_idx["test"]], batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    # instantiate model
    if args.gnn == 'sin':
        model = SIN(num_classes=dataset.num_classes,
                    num_layers=args.num_layers,
                    emb_dim=args.emb_dim,
                    drop_rate=args.drop_rate).to(device)
    else:
        raise ValueError('Invalid model type')
        
    # instantiate optimiser
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # (!) start training/evaluation
    valid_curve = []
    test_curve = []
    train_curve = []
    train_loss_curve = []
    if not args.untrained:
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
        'Dataset:        {0}\n',
        'Validation:     {1}\n',
        'Test:           {2}\n',
        'Train:          {3}\n',
        'Best epoch:     {4}\n',
        '-------------------------------\n')
    msg.format(valid_curve[best_val_epoch], test_curve[best_val_epoch], train_curve[best_val_epoch], best_val_epoch)
    msg += str(args)
    with open(filename, 'wb') as handle:
        handle.write(msg)
    if args.dump_curves:
        with open(result_path+'curves.pkl', 'wb') ad handle:
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