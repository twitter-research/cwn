import os

import pickle
import torch
import torch.optim as optim
import random

from data.data_loading import DataLoader, load_dataset
from exp.train_utils import train, eval, Evaluator
from exp.parser import get_parser
from mp.molec_models import ZincSparseSIN

from definitions import ROOT_DIR

import time
import numpy as np
import copy


def main(args):
    # set device
    device = torch.device(
        "cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")

    if args.result_folder is None:
        result_folder = os.path.join(ROOT_DIR, 'exp', 'results')
    else:
        result_folder = args.result_folder

    print("==========================================================================")
    print("Using device", str(device))
    print("======================== Args ===========================")
    print(args)

    # Set the seed for everything
    torch.manual_seed(43)
    np.random.seed(43)
    random.seed(43)

    exp_name = time.time() if args.exp_name is None else args.exp_name
    result_folder = os.path.join(result_folder, '{}-{}'.format(args.dataset, exp_name))

    if not os.path.exists(result_folder):
        os.makedirs(result_folder)
    filename = os.path.join(result_folder, 'results.txt')

    # Data loading
    dataset = load_dataset(args.dataset, max_dim=args.max_dim, fold=args.fold,
                           init_method=args.init_method, emb_dim=args.emb_dim,
                           flow_points=args.flow_points, flow_classes=args.flow_classes,
                           max_ring_size=args.max_ring_size,
                           use_edge_features=args.use_edge_features)
    split_idx = dataset.get_idx_split()

    # Instantiate data loaders
    train_loader = DataLoader(dataset[split_idx["train"]], batch_size=args.batch_size,
                              shuffle=True, num_workers=args.num_workers, max_dim=dataset.max_dim)
    valid_loader = DataLoader(dataset[split_idx["valid"]], batch_size=args.batch_size,
                              shuffle=False, num_workers=args.num_workers, max_dim=dataset.max_dim)
    test_loader = DataLoader(dataset[split_idx["test"]], batch_size=args.batch_size,
                             shuffle=False, num_workers=args.num_workers, max_dim=dataset.max_dim)

    # automatic evaluator, takes dataset name as input
    evaluator = Evaluator(args.eval_metric)

    # instantiate model
    # NB: here we assume to have the same number of features per dim
    if args.model == 'zinc_sparse_sin':
        assert args.dataset == 'ZINC'
        assert args.task_type == 'regression'
        assert args.minimize
        assert args.lr_scheduler == 'ReduceLROnPlateau'
        model = ZincSparseSIN(28,  # The number of atomic types
                              1,  # num_classes
                              args.num_layers,  # num_layers
                              args.emb_dim,  # hidden
                              dropout_rate=args.drop_rate,  # dropout rate
                              max_dim=dataset.max_dim,  # max_dim
                              jump_mode=args.jump_mode,  # jump mode
                              nonlinearity=args.nonlinearity,  # nonlinearity
                              readout=args.readout,  # readout
                              final_readout=args.final_readout,  # final readout
                              apply_dropout_before=args.drop_position,  # where to apply dropout
                              ).to(device)
    else:
        raise ValueError('Invalid model type {}.'.format(args.model))

    print("============= Model Parameters =================")
    trainable_params = 0
    total_params = 0
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(name, param.size())
            trainable_params += param.numel()
        total_params += param.numel()
    print("============= Params stats ==================")
    print(f"Trainable params: {trainable_params}")
    print(f"Total params    : {total_params}")

    # instantiate optimiser
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # instantiate learning rate decay
    if args.lr_scheduler == 'ReduceLROnPlateau':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
                                                               factor=args.lr_scheduler_decay_rate,
                                                               patience=args.lr_scheduler_patience,
                                                               min_lr=args.lr_scheduler_min,
                                                               verbose=True)
    elif args.lr_scheduler == 'StepLR':
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.lr_scheduler_decay_steps,
                                                    gamma=args.lr_scheduler_decay_rate)
    elif args.lr_scheduler == 'None':
        scheduler = None
    else:
        raise NotImplementedError(
            'Scheduler {} is not currently supported.'.format(args.lr_scheduler))

    # Start training and evaluation
    train_curve = []
    valid_curve = []
    test_curve = []
    train_loss_curve = []
    params = []
    for epoch in range(1, args.epochs + 1):

        # Perform one epoch of training
        print("=====Epoch {}".format(epoch))
        print('Training...')
        epoch_train_curve = train(model, device, train_loader, optimizer, args.task_type)
        train_loss_curve += epoch_train_curve
        epoch_train_loss = float(np.mean(epoch_train_curve))

        # evaluate model
        print('Evaluating...')
        if epoch == 1 or epoch % args.train_eval_period == 0:
            train_perf, _ = eval(model, device, train_loader, evaluator, args.task_type)
        train_curve.append(train_perf)
        valid_perf, epoch_val_loss = eval(model, device, valid_loader, evaluator, args.task_type)
        valid_curve.append(valid_perf)

        test_perf, epoch_test_loss = eval(model, device, test_loader, evaluator, args.task_type)
        test_curve.append(test_perf)

        print(f'Train: {train_perf:.3f} | Validation: {valid_perf:.3f} | Test: {test_perf:.3f}'
              f' | Train Loss {epoch_train_loss:.3f} | Val Loss {epoch_val_loss:.3f}'
              f' | Test Loss {epoch_test_loss:.3f}')

        # decay learning rate
        if scheduler is not None:
            if args.lr_scheduler == 'ReduceLROnPlateau':
                scheduler.step(valid_perf)
            else:
                scheduler.step()

        i = 0
        new_params = []
        if epoch % args.train_eval_period == 0:
            print("====== Slowly changing params ======= ")
        for name, param in model.named_parameters():
            # print(f"Param {name}: {param.data.view(-1)[0]}")
            # new_params.append(param.data.detach().clone().view(-1)[0])
            new_params.append(param.data.detach().mean().item())
            if len(params) > 0 and epoch % args.train_eval_period == 0:
                if abs(params[i] - new_params[i]) < 1e-6:
                    print(f"Param {name}: {params[i] - new_params[i]}")
            i += 1
        params = copy.copy(new_params)

    if not args.minimize:
        best_val_epoch = np.argmax(np.array(valid_curve))
    else:
        best_val_epoch = np.argmin(np.array(valid_curve))
    print('Finished training!')
    print('Best validation score: {}'.format(valid_curve[best_val_epoch]))
    print('Test score: {}'.format(test_curve[best_val_epoch]))

    # Save the results
    curves = {
        'train_loss': train_loss_curve,
        'train': train_curve,
        'val': valid_curve,
        'test': test_curve,
        'best': best_val_epoch}
    msg = (
        f'Dataset:        {args.dataset}\n'
        f'Validation:     {valid_curve[best_val_epoch]}\n'
        f'Train:          {train_curve[best_val_epoch]}\n'
        f'Test:           {test_curve[best_val_epoch]}\n'
        f'Best epoch:     {best_val_epoch}\n'
        '-------------------------------\n')
    msg += str(args)
    with open(filename, 'w') as handle:
        handle.write(msg)
    if args.dump_curves:
        with open(os.path.join(result_folder, 'curves.pkl'), 'wb') as handle:
            pickle.dump(curves, handle)

    return curves


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()

    main(args)
