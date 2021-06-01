import os
import numpy as np
import copy
import pickle
import torch
import torch.optim as optim
import random
import time

from data.data_loading import DataLoader, load_dataset, load_graph_dataset
from torch_geometric.data import DataLoader as PyGDataLoader
from exp.train_utils import train, eval, Evaluator
from exp.parser import get_parser, validate_args
from mp.graph_models import GIN0, GINWithJK
from mp.models import SIN0, Dummy, SparseSIN, EdgeOrient, EdgeMPNN, MessagePassingAgnostic
from mp.molec_models import EmbedSparseSIN, OGBEmbedSparseSIN, EmbedSparseSINNoRings, EmbedGIN

__timing_warmup__ = 1

def main(args):

    # set device
    device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")

    print("==========================================================")
    print("Using device", str(device))
    print(f"Fold: {args.fold}")
    print(f"Seed: {args.seed}")
    print("======================== Args ===========================")
    print(args)
    print("===================================================")

    # Set the seed for everything
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    # Create results folder
    result_folder = os.path.join(
        args.result_folder, f'{args.dataset}-{args.exp_name}', f'seed-{args.seed}')
    if args.fold is not None:
        result_folder = os.path.join(result_folder, f'fold-{args.fold}')
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
                               init_method=args.init_method, emb_dim=args.emb_dim,
                               flow_points=args.flow_points, flow_classes=args.flow_classes,
                               max_ring_size=args.max_ring_size,
                               use_edge_features=args.use_edge_features,
                               simple_features=args.simple_features, n_jobs=args.preproc_jobs)
        if args.tune:
            split_idx = dataset.get_tune_idx_split()
        else:
            split_idx = dataset.get_idx_split()

        # Instantiate data loaders
        train_loader = DataLoader(dataset.get_split('train'), batch_size=args.batch_size,
            shuffle=True, num_workers=args.num_workers, max_dim=dataset.max_dim)
        valid_loader = DataLoader(dataset.get_split('valid'), batch_size=args.batch_size,
            shuffle=False, num_workers=args.num_workers, max_dim=dataset.max_dim)
        test_split = split_idx.get("test", None)
        test_loader = None
        if test_split is not None:
            test_loader = DataLoader(dataset.get_split('test'), batch_size=args.batch_size,
                shuffle=False, num_workers=args.num_workers, max_dim=dataset.max_dim)
            
    # automatic evaluator, takes dataset name as input
    evaluator = Evaluator(args.eval_metric)
    
    # use cofaces?
    use_cofaces = args.use_cofaces.lower() == 'true'

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
                     use_cofaces=use_cofaces,                 # whether to use cofaces in up-msg
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
    elif args.model == 'mp_agnostic':    
        model = MessagePassingAgnostic(
                     dataset.num_features_in_dim(0),          # num_input_features
                     dataset.num_classes,                     # num_classes
                     args.emb_dim,                            # hidden
                     dropout_rate=args.drop_rate,             # dropout rate
                     max_dim=dataset.max_dim,                 # max_dim
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
    elif args.model == 'embed_sparse_sin':
        model = EmbedSparseSIN(dataset.num_node_type,  # The number of atomic types
                               dataset.num_edge_type,  # The number of bond types
                               dataset.num_classes,  # num_classes
                               args.num_layers,  # num_layers
                               args.emb_dim,  # hidden
                               dropout_rate=args.drop_rate,  # dropout rate
                               max_dim=dataset.max_dim,  # max_dim
                               jump_mode=args.jump_mode,  # jump mode
                               nonlinearity=args.nonlinearity,  # nonlinearity
                               readout=args.readout,  # readout
                               final_readout=args.final_readout,  # final readout
                               apply_dropout_before=args.drop_position,  # where to apply dropout
                               use_cofaces=use_cofaces,
                               embed_edge=args.use_edge_features
                               ).to(device)
    elif args.model == 'embed_sparse_sin_no_rings':
        model = EmbedSparseSINNoRings(dataset.num_node_type,  # The number of atomic types
                                      dataset.num_edge_type,  # The number of bond types
                                      dataset.num_classes,  # num_classes
                                      args.num_layers,  # num_layers
                                      args.emb_dim,  # hidden
                                      dropout_rate=args.drop_rate,  # dropout rate
                                      nonlinearity=args.nonlinearity,  # nonlinearity
                                      readout=args.readout,  # readout
                                      final_readout=args.final_readout,  # final readout
                                      apply_dropout_before=args.drop_position,  # where to apply dropout
                                      use_cofaces=use_cofaces,
                                      embed_edge=args.use_edge_features
                                      ).to(device)
    elif args.model == 'embed_gin':
        model = EmbedGIN(dataset.num_node_type,  # The number of atomic types
                         dataset.num_edge_type,  # The number of bond types
                         dataset.num_classes,  # num_classes
                         args.num_layers,  # num_layers
                         args.emb_dim,  # hidden
                         dropout_rate=args.drop_rate,  # dropout rate
                         nonlinearity=args.nonlinearity,  # nonlinearity
                         readout=args.readout,  # readout
                         final_readout=args.final_readout,  # final readout
                         apply_dropout_before=args.drop_position,  # where to apply dropout
                         embed_edge=args.use_edge_features
                        ).to(device)
    # TODO: handle this as above
    elif args.model == 'ogb_embed_sparse_sin':
        model = OGBEmbedSparseSIN(dataset.num_tasks,                       # out_size
                                  args.num_layers,                         # num_layers
                                  args.emb_dim,                            # hidden
                                  dropout_rate=args.drop_rate,             # dropout_rate
                                  indropout_rate=args.indrop_rate,         # in-dropout_rate
                                  max_dim=dataset.max_dim,                 # max_dim
                                  jump_mode=args.jump_mode,                # jump_mode
                                  nonlinearity=args.nonlinearity,          # nonlinearity
                                  readout=args.readout,                    # readout
                                  final_readout=args.final_readout,        # final readout
                                  apply_dropout_before=args.drop_position, # where to apply dropout
                                  use_cofaces=use_cofaces,                 # whether to use cofaces
                                  embed_edge=args.use_edge_features        # whether to use edge feats
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
        mode = 'min' if args.minimize else 'max'
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode=mode,
                                                               factor=args.lr_scheduler_decay_rate,
                                                               patience=args.lr_scheduler_patience,
                                                               verbose=True)
    elif args.lr_scheduler == 'StepLR':
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.lr_scheduler_decay_steps,
                                                    gamma=args.lr_scheduler_decay_rate)
    elif args.lr_scheduler == 'None':
        scheduler = None
    else:
        raise NotImplementedError(f'Scheduler {args.lr_scheduler} is not currently supported.')

    # (!) start training/evaluation
    best_val_epoch = 0
    valid_curve = []
    test_curve = []
    train_curve = []
    train_loss_curve = []
    params = []
    t_times = []
    e_trn_times = []
    e_val_times = []
    e_test_times = []
    if not args.untrained:
        for epoch in range(1, args.epochs + 1):

            # perform one epoch
            print("=====Epoch {}".format(epoch))
            print('Training...')
            ts_start = time.time()
            epoch_train_curve = train(model, device, train_loader, optimizer, args.task_type)
            ts_stop = time.time()
            if epoch > __timing_warmup__:
                print(epoch)
                t_times.append(ts_stop - ts_start)
            train_loss_curve += epoch_train_curve
            epoch_train_loss = float(np.mean(epoch_train_curve))
            
            # evaluate model
            print('Evaluating...')
            if epoch == 1 or epoch % args.train_eval_period == 0:
                ts_start = time.time()
                train_perf, _ = eval(model, device, train_loader, evaluator, args.task_type)
                ts_stop = time.time()
                if epoch > __timing_warmup__:
                    e_trn_times.append(ts_stop - ts_start)
            train_curve.append(train_perf)
            
            ts_start = time.time()
            valid_perf, epoch_val_loss = eval(model, device,
                valid_loader, evaluator, args.task_type)#, dataset[split_idx["valid"]])
            ts_stop = time.time()
            if epoch > __timing_warmup__:
                e_val_times.append(ts_stop - ts_start)
            valid_curve.append(valid_perf)

            if test_loader is not None:
                ts_start = time.time()
                test_perf, epoch_test_loss = eval(model, device, test_loader, evaluator,
                                                  args.task_type)
                ts_stop = time.time()
                if epoch > __timing_warmup__:
                    e_test_times.append(ts_stop - ts_start)
            else:
                test_perf = np.nan
                epoch_test_loss = np.nan
            test_curve.append(test_perf)

            print(f'Train: {train_perf:.3f} | Validation: {valid_perf:.3f} | Test: {test_perf:.3f}'
                  f' | Train Loss {epoch_train_loss:.3f} | Val Loss {epoch_val_loss:.3f}'
                  f' | Test Loss {epoch_test_loss:.3f}')
            
            # decay learning rate
            if scheduler is not None:
                if args.lr_scheduler == 'ReduceLROnPlateau':
                    scheduler.step(valid_perf)
                    # We use a strict inequality here like in the benchmarking GNNs paper code
                    # https://github.com/graphdeeplearning/benchmarking-gnns/blob/master/main_molecules_graph_regression.py#L217
                    if args.early_stop and optimizer.param_groups[0]['lr'] < args.lr_scheduler_min:
                        print("\n!! The minimum learning rate has been reached.")
                        break
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

    print('Final Evaluation...')
    final_train_perf, _ = eval(model, device, train_loader, evaluator, args.task_type)
    final_val_perf, _ = eval(model, device, valid_loader, evaluator, args.task_type)
    final_test_perf = np.nan
    if test_loader is not None:
        final_test_perf, _ = eval(model, device, test_loader, evaluator, args.task_type)

    # save results
    curves = {
        'train_loss': train_loss_curve,
        'train': train_curve,
        'val': valid_curve,
        'test': test_curve,
        'last_val': final_val_perf,
        'last_test': final_test_perf,
        'last_train': final_train_perf,
        'best': best_val_epoch,
        'timing_training': t_times,
        'timing_trainset_eval': e_trn_times,
        'timing_valset_eval': e_val_times,
        'timing_testset_eval': e_test_times}
    
    # Compute timing stats
    t_times_mean = np.nan
    t_times_std = np.nan
    if len(t_times) > 0:
        t_times_mean = np.mean(t_times)
        t_times_std = np.std(t_times)    
    e_trn_times_mean = np.nan
    e_trn_times_std = np.nan
    if len(e_trn_times) > 0:
        e_trn_times_mean = np.mean(e_trn_times)
        e_trn_times_std = np.std(e_trn_times)
    e_val_times_mean = np.nan
    e_val_times_std = np.nan
    if len(e_val_times) > 0:
        e_val_times_mean = np.mean(e_val_times)
        e_val_times_std = np.std(e_val_times)
    e_test_times_mean = np.nan
    e_test_times_std = np.nan
    if len(e_test_times) > 0:
        e_test_times_mean = np.mean(e_test_times)
        e_test_times_std = np.std(e_test_times)

    msg = (
       f'========== Result ============\n'
       f'Dataset:        {args.dataset}\n'
       f'------------ Best epoch -----------\n'
       f'Train:          {train_curve[best_val_epoch]}\n'
       f'Validation:     {valid_curve[best_val_epoch]}\n'
       f'Test:           {test_curve[best_val_epoch]}\n'
       f'Best epoch:     {best_val_epoch}\n'
       '------------ Last epoch -----------\n'
       f'Train:          {final_train_perf}\n'
       f'Validation:     {final_val_perf}\n'
       f'Test:           {final_test_perf}\n'
       '-------------- Timings -------------\n'
       f'Training:       {t_times_mean} ± {t_times_std} ({len(t_times)})\n'
       f'Eval (Train):   {e_trn_times_mean} ± {e_trn_times_std} ({len(e_trn_times)})\n'
       f'Eval (Val):     {e_val_times_mean} ± {e_val_times_std} ({len(e_val_times)})\n'
       f'Eval (Test):    {e_test_times_mean} ± {e_test_times_std} ({len(e_test_times)})\n'
       '-------------------------------\n\n')
    print(msg)

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
    validate_args(args)
    
    main(args)
