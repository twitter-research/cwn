import os
import sys
import copy
import time
import numpy as np
from definitions import ROOT_DIR
from exp.parser import get_parser
from exp.run_exp import main

__families__ = [
    'sr16622',
    'sr251256',
    'sr261034',
    'sr281264',
    'sr291467',
    'sr351668',
    'sr351899',
    'sr361446',
    'sr401224'
]

__max_dim__ = [
    3,
    4,
    3,
    6,
    4,
    4,
    6,
    3,
    3]

if __name__ == "__main__":
    
    # standard args
    passed_args = sys.argv[1:]
    assert '--seed' not in passed_args
    assert '--dataset' not in passed_args
    assert '--exp_name' not in passed_args
    parser = get_parser()
    args = parser.parse_args(copy.copy(passed_args))
    assert args.model in ['sparse_sin', 'mp_agnostic']
    ts = str(time.time())
    if args.model == 'mp_agnostic':
        result_folder = os.path.join(args.result_folder, 'sr-base')
    else:
        result_folder = os.path.join(args.result_folder, 'sr')
    if '--max_ring_size' in passed_args:
        result_folder += f'-{args.max_ring_size}'
    passed_args += ['--result_folder', result_folder]
    # run each experiment separately and gather results
    results = [list() for _ in __families__]
    for f, family in enumerate(__families__):
        for seed in range(args.start_seed, args.stop_seed + 1):
            print(f'[i] family {family}, seed {seed}')
            current_args = copy.copy(passed_args) + ['--dataset', family, '--exp_name', family, '--seed', str(seed)]
            if '--max_dim' not in passed_args:
                if '--max_ring_size' not in passed_args:
                    current_args += ['--max_dim', str(__max_dim__[f])]
                else:
                    current_args += ['--max_dim', str(2)]
            else:
                assert '--max_ring_size' not in passed_args
            parsed_args = parser.parse_args(current_args)
            curves = main(parsed_args)
            results[f].append(curves)
            
    msg = ''
    for f, family in enumerate(__families__):
        curves = results[f]
        test_perfs = [curve['last_test'] for curve in curves]
        assert len(test_perfs) == args.stop_seed + 1 - args.start_seed
        mean = np.mean(test_perfs)
        std_err = np.std(test_perfs) / float(len(test_perfs))
        minim = np.min(test_perfs)
        maxim = np.max(test_perfs)
        msg += (
            f'Dataset:               {family}\n'
            f'Mean failure rate:     {mean}\n'
            f'StdErr failure rate:   {std_err}\n'
            f'Min failure rate:      {minim}\n'
            f'Max failure rate:      {maxim}\n'
            '-----------------------------------------------\n')
    print(msg)
    
    # additionally write msg and configuration on file
    msg += str(args)
    filename = os.path.join(result_folder, 'result.txt')
    print('Writing results at: {}'.format(filename))
    with open(filename, 'w') as handle:
        handle.write(msg)
