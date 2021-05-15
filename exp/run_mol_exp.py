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

    # run each experiment separately and gather results
    results = list()
    if args.folds is None:
        for seed in range(args.seeds):
            current_args = copy.copy(passed_args) + ['--seed', str(seed)]
            parsed_args = parser.parse_args(current_args)
            curves = main(parsed_args)
            results.append(curves)
    else:
        # Used by CSL only to run experiments across both seeds and folds
        for seed, fold in product(range(args.seeds), range(args.folds)):
            current_args = copy.copy(passed_args) + ['--seed', str(seed)] + ['--fold', str(fold)]
            parsed_args = parser.parse_args(current_args)
            curves = main(parsed_args)
            results.append(curves)

    # Extract results
    val_curves = [curves['val'] for curves in results]
    test_curves = [curves['test'] for curves in results]
    best_idx = [curves['best'] for curves in results]

    test_results = [test_curves[i][best] for i, best in enumerate(best_idx)]
    test_results = np.array(test_results, dtype=np.float)

    mean_perf = np.mean(test_results)
    std_perf = np.std(test_results, ddof=1)  # ddof=1 makes the estimator unbiased
    min_perf = np.min(test_results)
    max_perf = np.max(test_results)

    print(" ===== Final result ======")
    msg = (
        f'Dataset:           {args.dataset}\n'
        f'Mean:              {mean_perf} ± {std_perf}\n'
        f'Min:               {min_perf}\n'
        f'Max:               {max_perf}\n'
        f'SHA:               {sha}\n'
        f'-------------------------------\n')
    print(msg)
    
    # additionally write msg and configuration on file
    filename = os.path.join(args.result_folder, f'{args.dataset}-{args.exp_name}/result.txt')
    print('Writing results at: {}'.format(filename))
    with open(filename, 'w') as handle:
        handle.write(msg)
        for arg in passed_args:
            if arg.startswith('--'):
                handle.write(arg+': ')
            else:
                handle.write(arg+'\n')


if __name__ == "__main__":
    passed_args = sys.argv[1:]
    assert 'seed' not in passed_args
    assert 'fold' not in passed_args
    exp_main(passed_args)
