import os

import pickle
import torch
import torch.optim as optim
import random

from data.data_loading import DataLoader, load_dataset, load_graph_dataset
from torch_geometric.data import DataLoader as PyGDataLoader
from exp.train_utils import train, eval, Evaluator
from exp.parser import get_parser
from mp.graph_models import GIN0, GINWithJK
from mp.models import SIN0, Dummy, SparseSIN, EdgeOrient, EdgeMPNN

from definitions import ROOT_DIR

import time
import numpy as np
import copy


def main(args):

    # set device
    device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")

    if args.result_folder is None:
        result_folder = os.path.join(ROOT_DIR, 'exp', 'results')
    else:
        result_folder = args.result_folder

    print("==========================================================================")
    print("Using device", str(device))
    print(f"Fold: {args.fold}")
    print("======================== Args ===========================")
    print(args)

    # Set the seed for everything
    torch.manual_seed(43)
    np.random.seed(43)
    random.seed(43)

    if args.exp_name is None:
        # get timestamp for results and set result directory
        exp_name = time.time()
    else:
        exp_name = args.exp_name
    result_folder = os.path.join(result_folder, '{}-{}'.format(args.dataset, exp_name))
    if args.fold is not None:
        result_folder = os.path.join(result_folder, 'fold-{}'.format(args.fold))
    if not os.path.exists(result_folder):
        os.makedirs(result_folder)
    filename = os.path.join(result_folder, 'results.txt')
    
    if args.model.startswith('gin'):  # load graph dataset
        
        graph_list, train_ids, val_ids, test_ids, num_classes = load_graph_dataset(args.dataset, fold=args.fold)
        train_graphs = [graph_list[i] for i in train_ids]
        val_graphs = [graph_list[i] for i in val_ids]
        train_loader = PyGDataLoader(train_graphs, batch_size=args.batch_size,
                                  shuffle=True, num_workers=args.num_workers)
        valid_loader = PyGDataLoader(val_graphs, batch_size=args.batch_size,
                                   shuffle=False, num_workers=args.num_workers)
        if test_ids is not None:
            test_graphs = [graph_list[i] for i in test_ids]
            test_loader = PyGDataLoader(test_graphs, batch_size=args.batch_size,
                                       shuffle=False, num_workers=args.num_workers)
        else:
            test_loader = None
        if args.dataset.startswith('sr'):
            num_features = 1
            num_classes = args.emb_dim
        else:
            num_features = graph_list[0].x.shape[1]
    
    else:
        
        # Data loading
        dataset = load_dataset(args.dataset, max_dim=args.max_dim, fold=args.fold,
            init_method=args.init_method, emb_dim=args.emb_dim, flow_points=args.flow_points,
            flow_classes=args.flow_classes, max_ring_size=args.max_ring_size)
        if args.tune:
            split_idx = dataset.get_tune_idx_split()
        else:
            split_idx = dataset.get_idx_split()

        # Instantiate data loaders
        train_loader = DataLoader(dataset[split_idx["train"]], batch_size=args.batch_size,
            shuffle=True, num_workers=args.num_workers, max_dim=dataset.max_dim)
        valid_loader = DataLoader(dataset[split_idx["valid"]], batch_size=args.batch_size,
            shuffle=False, num_workers=args.num_workers, max_dim=dataset.max_dim)
        test_split = split_idx.get("test", None)
        if test_split is not None:
            test_loader = DataLoader(dataset[test_split], batch_size=args.batch_size,
                shuffle=False, num_workers=args.num_workers, max_dim=dataset.max_dim)
        else:
            test_loader = None
            
    # automatic evaluator, takes dataset name as input
    evaluator = Evaluator(args.eval_metric)

    # instantiate model
    # NB: here we assume to have the same number of features per dim
    if args.model == 'sin':
        model = SIN0(dataset.num_features_in_dim(0),          # num_input_features
                     dataset.num_classes,                     # num_classes
                     args.num_layers,                         # num_layers
                     args.emb_dim,                            # hidden
                     dropout_rate=args.drop_rate,             # dropout rate
                     max_dim=dataset.max_dim,                 # max_dim
                     jump_mode=args.jump_mode,                # jump mode
                     nonlinearity=args.nonlinearity,          # nonlinearity
                     readout=args.readout,                    # readout
                    ).to(device)
    elif args.model == 'sparse_sin':
        model = SparseSIN(dataset.num_features_in_dim(0),     # num_input_features
                     dataset.num_classes,                     # num_classes
                     args.num_layers,                         # num_layers
                     args.emb_dim,                            # hidden
                     dropout_rate=args.drop_rate,             # dropout rate
                     max_dim=dataset.max_dim,                 # max_dim
                     jump_mode=args.jump_mode,                # jump mode
                     nonlinearity=args.nonlinearity,          # nonlinearity
                     readout=args.readout,                    # readout
                     final_readout=args.final_readout,        # final readout
                     apply_dropout_before=args.drop_position, # where to apply dropout
                    ).to(device)
    elif args.model == 'gin':
        model = GIN0(num_features,                            # num_input_features
                     args.num_layers,                         # num_layers
                     args.emb_dim,                            # hidden
                     num_classes,                             # num_classes
                     dropout_rate=args.drop_rate,             # dropout rate
                     nonlinearity=args.nonlinearity,          # nonlinearity
                     readout=args.readout,                    # readout
                    ).to(device)
    elif args.model == 'gin_jk':
        model = GINWithJK(num_features,                       # num_input_features
                     args.num_layers,                         # num_layers
                     args.emb_dim,                            # hidden
                     num_classes,                             # num_classes
                     dropout_rate=args.drop_rate,             # dropout rate
                     nonlinearity=args.nonlinearity,          # nonlinearity
                     readout=args.readout,                    # readout
                    ).to(device)
    elif args.model == 'dummy':
        model = Dummy(dataset.num_features_in_dim(0),
                      dataset.num_classes,
                      args.num_layers,
                      max_dim=dataset.max_dim,
                      readout=args.readout,
                     ).to(device)
    elif args.model == 'edge_orient':
        model = EdgeOrient(1,
                      dataset.num_classes,
                      args.num_layers,
                      args.emb_dim,  # hidden
                      readout=args.readout,
                     ).to(device)
    elif args.model == 'edge_mpnn':
        model = EdgeMPNN(1,
                      dataset.num_classes,
                      args.num_layers,
                      args.emb_dim,  # hidden
                      readout=args.readout,
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
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=args.lr_scheduler_decay_rate, patience=args.lr_scheduler_patience, verbose=True)
    elif args.lr_scheduler == 'StepLR':
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.lr_scheduler_decay_steps, gamma=args.lr_scheduler_decay_rate)
    elif args.lr_scheduler == 'None':
        scheduler = None
    else:
        raise NotImplementedError('Scheduler {} is not currently supported.'.format(args.lr_scheduler))

    # (!) start training/evaluation
    valid_curve = []
    test_curve = []
    train_curve = []
    train_loss_curve = []
    params = []
    if not args.untrained:
        for epoch in range(1, args.epochs + 1):

            # perform one epoch
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
            valid_perf, epoch_val_loss = eval(model, device,
                valid_loader, evaluator, args.task_type)#, dataset[split_idx["valid"]])
            valid_curve.append(valid_perf)
            if test_loader is not None:
                test_perf, _ = eval(model, device, test_loader, evaluator, args.task_type)
            else:
                test_perf = np.nan
            test_curve.append(test_perf)
            print(f'Train: {train_perf:.3f} | Validation: {valid_perf:.3f} | Test: {test_perf:.3f}'
                  f' | Train Loss {epoch_train_loss:.3f} | Val Loss {epoch_val_loss:.3f}')
            
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
        with open(os.path.join(result_folder,'curves.pkl'), 'wb') as handle:
            pickle.dump(curves, handle)
            
    return curves


if __name__ == "__main__":

    parser = get_parser()
    args = parser.parse_args()
    
    main(args)
