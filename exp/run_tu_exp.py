import sys
import os
import copy
import time
import numpy as np
from definitions import ROOT_DIR
from exp.parser import get_parser
from exp.run_exp import main

# python3 -m exp.run_tu_exp --dataset IMDBBINARY --model sin --drop_rate 0.0 --lr 0.0001 --max_dim 2 --emb_dim 32 --dump_curves --epochs 30 --num_layers 1 --lr_scheduler StepLR --lr_scheduler_decay_steps 5

__num_folds__ = 10


def print_summary(summary):
    msg = ''
    for k, v in summary.items():
        msg += f'Fold {k:1d}:  {v:.3f}\n'
    print(msg)
    
    
def exp_main(passed_args):
    
    parser = get_parser()
    # run each experiment separately and gather results
    results = list()
    for i in range(__num_folds__):
        current_args = copy.copy(passed_args) + ['--fold', str(i)]
        parsed_args = parser.parse_args(current_args)
        curves = main(parsed_args)
        results.append(curves)
        
    # aggregate results
    train_curves = np.asarray([curves['train'] for curves in results])
    val_curves = np.asarray([curves['val'] for curves in results])
    avg_train_curve = train_curves.mean(axis=0)
    avg_val_curve = val_curves.mean(axis=0)
    best_index = np.argmax(avg_val_curve)
    mean_perf = avg_val_curve[best_index]
    std_perf = val_curves.std(axis=0)[best_index]

    print(" ===== Mean performance per fold ======")
    perf_per_fold = val_curves.mean(1)
    perf_per_fold = {i: perf_per_fold[i] for i in range(len(perf_per_fold))}
    print_summary(perf_per_fold)

    print(" ===== Max performance per fold ======")
    perf_per_fold = np.max(val_curves, axis=1)
    perf_per_fold = {i: perf_per_fold[i] for i in range(len(perf_per_fold))}
    print_summary(perf_per_fold)

    print(" ===== Median performance per fold ======")
    perf_per_fold = np.median(val_curves, axis=1)
    perf_per_fold = {i: perf_per_fold[i] for i in range(len(perf_per_fold))}
    print_summary(perf_per_fold)

    print(" ===== Performance on best epoch ======")
    perf_per_fold = val_curves[:, best_index]
    perf_per_fold = {i: perf_per_fold[i] for i in range(len(perf_per_fold))}
    print_summary(perf_per_fold)

    print(" ===== Final result ======")
    msg = (
        'Dataset:        {0}\n'
        'Accuracy:       {1} ± {2}\n'
        'Best epoch:     {3}\n'
        '-------------------------------\n').format(parsed_args.dataset,
                                                    mean_perf, std_perf, best_index)
    print(msg)
    
    # additionally write msg and configuration on file
    exp_name = None
    result_folder = None
    for a, arg in enumerate(passed_args):
        if not arg.startswith('--'):
            continue
        key = arg[2:]
        if key == 'result_folder':
            result_folder = passed_args[a+1]
        if key == 'exp_name':
            exp_name = passed_args[a+1]
    if exp_name is None:
        exp_name = str(time.time())
    if result_folder is None:
        result_folder = os.path.join(ROOT_DIR, 'exp', 'results')
    filename = os.path.join(result_folder, '{}-{}/result.txt'.format(parsed_args.dataset, exp_name))
    print('Writing results at: {}'.format(filename))
    with open(filename, 'w') as handle:
        handle.write(msg)
        for arg in passed_args:
            if arg.startswith('--'):
                handle.write(arg+': ')
            else:
                handle.write(arg+'\n')

if __name__ == "__main__":
    
    # standard args
    passed_args = sys.argv[1:]
    assert 'fold' not in passed_args
    if '--result_folder' not in passed_args:
        result_folder = os.path.join(ROOT_DIR, 'exp', 'results')
        passed_args += ['--result_folder', result_folder]
    if '--exp_name' not in passed_args:
        passed_args += ['--exp_name', str(time.time())]

    exp_main(passed_args)