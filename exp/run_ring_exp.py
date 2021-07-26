import os
import sys
import copy
import subprocess
import numpy as np

from exp.parser import get_parser
from exp.run_exp import main

RING_SIZES = list(range(10, 32, 2))


def exp_main(passed_args):
    # Extract the commit sha so we can check the code that was used for each experiment
    sha = subprocess.check_output(["git", "describe", "--always"]).strip().decode()

    parser = get_parser()
    args = parser.parse_args(copy.copy(passed_args))
    assert args.max_ring_size is None

    # run each experiment separately and gather results
    train_results = {fold: [] for fold in range(len(RING_SIZES))}
    val_results = {fold: [] for fold in range(len(RING_SIZES))}
    for seed in range(args.start_seed, args.stop_seed + 1):
        # We use the ring_size as a "fold" for the dataset.
        # This is just a hack to save the results properly using our usual infrastructure.
        for fold in range(len(RING_SIZES)):
            max_ring_size = RING_SIZES[fold]
            num_layers = 3 if args.model == 'ring_sparse_cin' else max_ring_size // 2
            current_args = (copy.copy(passed_args) + ['--fold', str(fold)] +
                            ['--max_ring_size', str(max_ring_size)] +
                            ['--num_layers', str(num_layers)] +
                            ['--seed', str(seed)])
            parsed_args = parser.parse_args(current_args)
            # Check that the default parameter value (5) was overwritten
            assert parsed_args.num_layers == num_layers
            curves = main(parsed_args)

            # Extract results
            train_results[fold].append(curves['last_train'])
            val_results[fold].append(curves['last_val'])

    msg = (
        f"========= Final result ==========\n"
        f'Dataset:                {args.dataset}\n'
        f'SHA:                    {sha}\n'
        f'----------- Train ----------\n')

    for fold, results in train_results.items():
        mean = np.mean(results)
        std = np.std(results)
        msg += f'Ring size: {RING_SIZES[fold]}            {mean}+-{std}\n'

    msg += f'----------- Test ----------\n'

    for fold, results in val_results.items():
        mean = np.mean(results)
        std = np.std(results)
        msg += f'Ring size: {RING_SIZES[fold]}            {mean}+-{std}\n'

    print(msg)

    # additionally write msg and configuration on file
    msg += str(args)
    filename = os.path.join(args.result_folder, f'{args.dataset}-{args.exp_name}/result.txt')
    print('Writing results at: {}'.format(filename))
    with open(filename, 'w') as handle:
        handle.write(msg)


if __name__ == "__main__":
    passed_args = sys.argv[1:]
    assert '--fold' not in passed_args
    assert '--seed' not in passed_args
    exp_main(passed_args)
