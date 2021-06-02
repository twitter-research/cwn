import sys
import os
import os.path as osp
import errno
import numpy as np
import argparse
import time

from data.data_loading import load_graph_dataset
from data.utils import convert_graph_dataset_with_rings
from definitions import ROOT_DIR

def makedirs(path):
    try:
        os.makedirs(osp.expanduser(osp.normpath(path)))
    except OSError as e:
        if e.errno != errno.EEXIST and osp.isdir(path):
            raise e

parser = argparse.ArgumentParser(description='Dataset conversion experiment.')
parser.add_argument('--dataset', type=str, default="ZINC",
                    help='dataset name (default: ZINC)')
parser.add_argument('--n_jobs', type=int, default=1,
                    help='Number of jobs to use')
parser.add_argument('--max_ring_size', type=int, default=18,
                    help='maximum ring size to look for')
parser.add_argument('--include_down_adj', action='store_true',
                    help='whether to include down adjs')
parser.add_argument('--init_edges', action='store_true',
                    help='whether to init edges')
parser.add_argument('--init_rings', action='store_true',
                    help='whether to init rings')
parser.add_argument('--num_runs', type=int, default=5,
                    help='number of iterations')
parser.add_argument('--result_folder', type=str,
                    default=osp.join(ROOT_DIR, 'exp', 'results'),
                    help='folder where to write results')
parser.add_argument('--exp_name', type=str,
                    default=None, help='experiment identifier/name')


def exp_main(passed_args):

    args = parser.parse_args(passed_args)
    exp_name = str(time.ts()) if args.exp_name is None else args.exp_name
    result_folder = osp.join(args.result_folder, f'dataset_timings_{args.dataset}-{exp_name}')
    makedirs(result_folder)
    result_filename = osp.join(result_folder, 'results.txt')
    
    print('\n----==== {} ====----'.format(args.dataset))
    dataset, _, _, _, _ = load_graph_dataset(args.dataset)
    
    times = list()
    for i in range(args.num_runs + 1):
        print(f'Iteration {i}...')
        ts_start = time.time()
        _, _, _ = convert_graph_dataset_with_rings(dataset, max_ring_size=args.max_ring_size,
                                                   include_down_adj=args.include_down_adj,
                                                   init_edges=args.init_edges, init_rings=args.init_rings,
                                                   n_jobs=args.n_jobs)
        ts_stop = time.time()
        times.append(ts_stop - ts_start)
    
    msg  = f'Conversion time:   {np.mean(times[1:])} ± {np.std(times[1:])} ({args.num_runs}, {args.n_jobs}x)\n'
    msg += f'Dry run:           {times[0]} ({args.n_jobs}x)\n'
    msg += "--------------------------------------------------------------\n"
    msg += str(args)
    
    print(msg)
    print(f'\nWriting results at {result_filename}...')
    with open(result_filename, 'w') as handle:
        handle.write(msg)

if __name__ == "__main__":
    passed_args = sys.argv[1:]
    exp_main(passed_args)
