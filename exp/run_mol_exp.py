import sys
import os
import copy
import time
import numpy as np
from exp.parser import get_parser
from exp.run_exp import main
    
    
def exp_main(passed_args):
    
    parser = get_parser()
    args = parser.parse_args(copy.copy(passed_args))

    # run each experiment separately and gather results
    results = list()
    for i in range(args.seeds):
        current_args = copy.copy(passed_args) + ['--seed', str(i)]
        parsed_args = parser.parse_args(current_args)
        curves = main(parsed_args)
        results.append(curves)
        
    # Extract results
    train_curves = np.asarray([curves['train'] for curves in results])
    val_curves = np.asarray([curves['val'] for curves in results])
    test_curves = np.asarray([curves['test']] for curves in results)

    best_val_idx = np.argmin(val_curves, axis=-1)
    test_results = test_curves[:, best_val_idx]

    mean_perf = np.mean(test_results)
    std_perf = np.std(test_results)

    print(" ===== Final result ======")
    msg = (
        f'Dataset:           {args.dataset}\n'
        f'Performance:       {mean_perf} ± {std_perf}\n'
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
    
    # standard args
    passed_args = sys.argv[1:]
    assert 'seed' not in passed_args
    if '--exp_name' not in passed_args:
        passed_args += ['--exp_name', str(time.time())]

    exp_main(passed_args)
