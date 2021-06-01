import sys
import os.path as osp
import numpy as np
import argparse
import time

from data.data_loading import load_graph_dataset
from data.utils import convert_graph_dataset_with_rings
from definitions import ROOT_DIR

parser = argparse.ArgumentParser(description='Dataset conversion experiment.')
parser.add_argument('--dataset', type=str, default="ZINC",
                    help='dataset name (default: ZINC)')
parser.add_argument('--n_jobs', type=int, default=1,
                    help='Number of jobs to use')
parser.add_argument('--max_ring_size', type=int, default=18,
                    help='maximum ring size to look for')
parser.add_argument('--include_down_adj', type=bool, default=False,
                    help='whether to include down adjs')
parser.add_argument('--init_edges', type=bool, default=True,
                    help='whether to init edges')
parser.add_argument('--init_rings', type=bool, default=False,
                    help='whether to init rings')
parser.add_argument('--num_runs', type=int, default=5,
                    help='number of iterations')

def exp_main(passed_args):

    args = parser.parse_args(passed_args)
    print('\n----==== {} ====----'.format(args.dataset))
    dataset, _, _, _, _ = load_graph_dataset(args.dataset)
    
    times = list()
    for i in range(args.num_runs):
        print(f'Iteration {i}...')
        ts_start = time.time()
        _, _, _ = convert_graph_dataset_with_rings(dataset, max_ring_size=args.max_ring_size,
                                                   include_down_adj=args.include_down_adj,
                                                   init_edges=args.init_edges, init_rings=args.init_rings,
                                                   n_jobs=args.n_jobs)
        ts_stop = time.time()
        times.append(ts_stop - ts_start)

    print(f'Conversion time:   {np.mean(times)} ± {np.std(times)} ({args.num_runs}, {args.n_jobs}x)')
    print("--------------------------------------------------------------")
    print(str(args))

if __name__ == "__main__":
    passed_args = sys.argv[1:]
    exp_main(passed_args)
