import sys
import os
import copy
import numpy as np
import subprocess

from exp.parser import get_parser
from exp.run_exp import main
from itertools import product


def exp_main(passed_args):
    # Extract the commit sha so we can check the code that was used for each experiment
    sha = subprocess.check_output(["git", "describe", "--always"]).strip().decode()

    parser = get_parser()
    args = parser.parse_args(copy.copy(passed_args))
    assert args.stop_seed >= args.start_seed

    # run each experiment separately and gather results
    results = list()
    if args.folds is None:
        for seed in range(args.start_seed, args.stop_seed + 1):
            current_args = copy.copy(passed_args) + ['--seed', str(seed)]
            parsed_args = parser.parse_args(current_args)
            curves = main(parsed_args)
            results.append(curves)
    else:
        # Used by CSL only to run experiments across both seeds and folds
        assert args.dataset == 'CSL'
        for seed, fold in product(range(args.start_seed, args.stop_seed + 1), range(args.folds)):
            current_args = copy.copy(passed_args) + ['--seed', str(seed)] + ['--fold', str(fold)]
            parsed_args = parser.parse_args(current_args)
            curves = main(parsed_args)
            results.append(curves)

    # Extract results
    train_curves = [curves['train'] for curves in results]
    val_curves = [curves['val'] for curves in results]
    test_curves = [curves['test'] for curves in results]
    best_idx = [curves['best'] for curves in results]
    last_train = [curves['last_train'] for curves in results]
    last_val = [curves['last_val'] for curves in results]
    last_test = [curves['last_test'] for curves in results]

    # Extract results at the best validation epoch.
    best_epoch_train_results = [train_curves[i][best] for i, best in enumerate(best_idx)]
    best_epoch_train_results = np.array(best_epoch_train_results, dtype=np.float)
    best_epoch_val_results = [val_curves[i][best] for i, best in enumerate(best_idx)]
    best_epoch_val_results = np.array(best_epoch_val_results, dtype=np.float)
    best_epoch_test_results = [test_curves[i][best] for i, best in enumerate(best_idx)]
    best_epoch_test_results = np.array(best_epoch_test_results, dtype=np.float)

    # Compute stats for the best validation epoch
    mean_train_perf = np.mean(best_epoch_train_results)
    std_train_perf = np.std(best_epoch_train_results, ddof=1)  # ddof=1 makes the estimator unbiased
    mean_val_perf = np.mean(best_epoch_val_results)
    std_val_perf = np.std(best_epoch_val_results, ddof=1)  # ddof=1 makes the estimator unbiased
    mean_test_perf = np.mean(best_epoch_test_results)
    std_test_perf = np.std(best_epoch_test_results, ddof=1)  # ddof=1 makes the estimator unbiased
    min_perf = np.min(best_epoch_test_results)
    max_perf = np.max(best_epoch_test_results)

    # Compute stats for the last epoch
    mean_final_train_perf = np.mean(last_train)
    std_final_train_perf = np.std(last_train, ddof=1)
    mean_final_val_perf = np.mean(last_val)
    std_final_val_perf = np.std(last_val, ddof=1)
    mean_final_test_perf = np.mean(last_test)
    std_final_test_perf = np.std(last_test, ddof=1)
    final_test_min = np.min(last_test)
    final_test_max = np.max(last_test)

    msg = (
        f"========= Final result ==========\n"
        f'Dataset:                {args.dataset}\n'
        f'SHA:                    {sha}\n'
        f'----------- Best epoch ----------\n'
        f'Train:                  {mean_train_perf} ± {std_train_perf}\n'
        f'Valid:                  {mean_val_perf} ± {std_val_perf}\n'
        f'Test:                   {mean_test_perf} ± {std_test_perf}\n'
        f'Test Min:               {min_perf}\n'
        f'Test Max:               {max_perf}\n'
        f'----------- Last epoch ----------\n'
        f'Train:                  {mean_final_train_perf} ± {std_final_train_perf}\n'
        f'Valid:                  {mean_final_val_perf} ± {std_final_val_perf}\n'
        f'Test:                   {mean_final_test_perf} ± {std_final_test_perf}\n'
        f'Test Min:               {final_test_min}\n'
        f'Test Max:               {final_test_max}\n'
        f'---------------------------------\n\n')
    print(msg)
    
    # additionally write msg and configuration on file
    msg += str(args)
    filename = os.path.join(args.result_folder, f'{args.dataset}-{args.exp_name}/result.txt')
    print('Writing results at: {}'.format(filename))
    with open(filename, 'w') as handle:
        handle.write(msg)


if __name__ == "__main__":
    passed_args = sys.argv[1:]
    assert '--seed' not in passed_args
    assert '--fold' not in passed_args
    exp_main(passed_args)
