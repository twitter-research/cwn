import sys
import os
import copy
import numpy as np
import subprocess

from exp.parser import get_parser
from exp.run_exp import main

ring_size = list(range(10, 32, 2))


def exp_main(passed_args):
    # Extract the commit sha so we can check the code that was used for each experiment
    sha = subprocess.check_output(["git", "describe", "--always"]).strip().decode()

    parser = get_parser()
    args = parser.parse_args(copy.copy(passed_args))
    assert args.max_ring_size is None
    assert args.num_layers is None

    # run each experiment separately and gather results
    results = list()
    # We use the ring_size as a "fold" for the dataset. This is just a trick to save the results.
    for fold in range(len(ring_size)):
        max_ring_size = ring_size[fold]
        num_layers = 3 if args.model == 'ring_sparse_sin' else max_ring_size // 2
        current_args = (copy.copy(passed_args) + ['--fold', str(fold)] +
                        ['--max_ring_size', str(max_ring_size)] +
                        ['--num_layers', str(num_layers)])
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

    print(sha)

    print("Train")
    print(last_train)

    print("Val")
    print(last_val)


if __name__ == "__main__":
    passed_args = sys.argv[1:]
    assert '--fold' not in passed_args
    exp_main(passed_args)
