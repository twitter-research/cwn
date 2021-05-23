import sys
import numpy as np
import argparse
import time

from data.parallel import ProgressParallel
from data.data_loading import load_graph_dataset
from data.utils import get_rings
from joblib import delayed

parser = argparse.ArgumentParser(description='Ring counting experiment.')
parser.add_argument('--dataset', type=str, default="ZINC",
                    help='dataset name (default: ZINC)')
parser.add_argument('--n_jobs', type=int, default=4,
                    help='Number of jobs to use')
parser.add_argument('--max_ring_size', type=int, default=12,
                    help='maximum ring size to look for')


def get_ring_count_for_graph(edge_index, max_ring, keys):
    rings = get_rings(edge_index, max_k=max_ring)
    rings_per_graph = {key: 0 for key in keys}
    for ring in rings:
        k = len(ring)
        rings_per_graph[k] += 1
    return rings_per_graph


def combine_all_cards(*cards):
    keys = cards[0].keys()
    ring_cards = {key: [] for key in keys}

    for card in cards:
        for k in keys:
            ring_cards[k].append(card[k])
    return ring_cards


def get_ring_counts(dataset, max_ring, jobs):
    start = time.time()
    keys = list(range(3, max_ring+1))

    parallel = ProgressParallel(n_jobs=jobs, use_tqdm=True, total=len(dataset))
    # It is important we supply a numpy array here. tensors seem to slow joblib down significantly.
    cards = parallel(delayed(get_ring_count_for_graph)(
        graph.edge_index.numpy(), max_ring, keys) for graph in dataset)

    end = time.time()
    print(f'Done ({end - start:.2f} secs).')
    return combine_all_cards(*cards)


def combine_all_counts(*stats):
    all_stats = dict()

    for k in stats[0].keys():
        all_stats[k] = []

    for stat in stats:
        for k, v in stat.items():
            # Extend the list
            all_stats[k] += v
    return all_stats


def print_stats(stats):
    for k in stats:
        min = np.min(stats[k])
        max = np.max(stats[k])
        mean = np.mean(stats[k])
        med = np.median(stats[k])
        sum = np.sum(stats[k])
        nz = np.count_nonzero(stats[k])
        print(
            f'Ring {k:02d} => Min: {min:.3f}, Max: {max:.3f}, Mean:{mean:.3f}, Median: {med:.3f}, '
            f'Sum: {sum:05d}, Non-zero: {nz:05d}')


def exp_main(passed_args):
    args = parser.parse_args(passed_args)

    print('----==== {} ====----'.format(args.dataset))
    graph_list, train_ids, val_ids, test_ids, _ = load_graph_dataset(args.dataset)
    graph_list = list(graph_list)  # Needed to bring OGB in the right format

    train = [graph_list[i] for i in train_ids]
    val = [graph_list[i] for i in val_ids]
    test = None
    if test_ids is not None:
        test = [graph_list[i] for i in test_ids]

    print("Counting rings on the training set ....")
    print("First, it will take a while to set up the processes...")
    train_stats = get_ring_counts(train, args.max_ring_size, args.n_jobs)
    print("Counting rings on the validation set ....")
    val_stats = get_ring_counts(val, args.max_ring_size, args.n_jobs)

    test_stats = None
    if test is not None:
        print("Counting rings on the test set ....")
        test_stats = get_ring_counts(test, args.max_ring_size, args.n_jobs)
        all_stats = combine_all_counts(train_stats, val_stats, test_stats)
    else:
        all_stats = combine_all_counts(train_stats, val_stats)

    print("=============== Train ================")
    print_stats(train_stats)
    print("=============== Validation ================")
    print_stats(val_stats)
    if test is not None:
        print("=============== Test ================")
        print_stats(test_stats)
    print("=============== Whole Dataset ================")
    print_stats(all_stats)


if __name__ == "__main__":
    passed_args = sys.argv[1:]
    exp_main(passed_args)
